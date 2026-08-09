"""POST /api/v1/documents/extract over the real ASGI stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

ENDPOINT = "/api/v1/documents/extract"

META_KEYS = {"patient_name", "age", "sex", "report_date", "lab_name", "reference_no"}
ITEM_KEYS = {"test_name", "value", "unit", "reference_range", "flag", "raw_line"}


def _post(client: TestClient, name: str, data: bytes, content_type: str = "image/png") -> Any:
    return client.post(ENDPOINT, files={"file": (name, data, content_type)})


def test_extracts_a_report_matching_the_documented_contract(
    client: TestClient, image_bytes: Any
) -> None:
    response = _post(client, "cbc_report.png", image_bytes("cbc_report"))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"meta", "results", "provider", "ocr_mean_confidence", "warnings"}
    assert set(body["meta"]) == META_KEYS
    assert body["meta"]["patient_name"] == "MD. RAFIQUL ISLAM"
    assert body["meta"]["age"] == "45 Y"
    assert body["meta"]["reference_no"] == "PDC-2024-0098871"
    assert len(body["results"]) == 11
    assert all(set(item) == ITEM_KEYS for item in body["results"])
    assert body["ocr_mean_confidence"] == pytest.approx(88.7)


def test_raw_line_survives_serialization_byte_for_byte(
    client: TestClient, image_bytes: Any, testdata_dir: Path
) -> None:
    """The audit trail must reach the client unaltered -- including its whitespace,
    which is what a reader compares against the paper report."""
    source = {
        line["text"]
        for line in json.loads(
            (testdata_dir / "ocr" / "responses" / "cbc_report.json").read_text(encoding="utf-8")
        )["lines"]
    }

    body = _post(client, "cbc_report.png", image_bytes("cbc_report")).json()

    for item in body["results"]:
        assert item["raw_line"] in source
    haemoglobin = next(i for i in body["results"] if i["test_name"] == "Haemoglobin")
    assert haemoglobin["raw_line"] == (
        "Haemoglobin          11.2        g/dl          13.0 - 17.0"
    )


@pytest.mark.parametrize(
    ("test_name", "expected_value", "unit", "flag"),
    [
        ("Haemoglobin", {"kind": "single", "parsed": True, "value": 11.2}, "g/dL", "low"),
        (
            "CRP",
            {"kind": "bounded", "parsed": True, "value": 0.5, "comparator": "<"},
            "mg/L",
            "normal",
        ),
        (
            "Platelet Count",
            {"kind": "single", "parsed": True, "value": 180000.0},
            "/µL",
            "normal",
        ),
    ],
)
def test_values_are_structured_objects_not_flattened_scalars(
    client: TestClient,
    image_bytes: Any,
    test_name: str,
    expected_value: dict[str, Any],
    unit: str,
    flag: str,
) -> None:
    """Flattening to a float would erase the difference between "0.5" and "<0.5"."""
    body = _post(client, "cbc_report.png", image_bytes("cbc_report")).json()
    item = next(i for i in body["results"] if i["test_name"] == test_name)

    for field, value in expected_value.items():
        assert item["value"][field] == pytest.approx(value) if isinstance(
            value, float
        ) else item["value"][field] == value
    assert item["unit"] == unit
    assert item["flag"] == flag


def test_preserves_an_ocr_garble_without_repairing_it(
    client: TestClient, image_bytes: Any
) -> None:
    """"2S.4" is a capital S misread for a 5. It must arrive as unparsed with the raw
    text intact -- never silently corrected to 25.4."""
    body = _post(client, "lipid_profile.png", image_bytes("lipid_profile")).json()
    item = next(i for i in body["results"] if "Vitamin D" in i["test_name"])

    assert item["value"]["raw"] == "2S.4"
    assert item["value"]["parsed"] is False
    assert item["value"]["value"] is None
    assert item["flag"] == "unknown"
    # The readable fields are still normalized.
    assert item["unit"] == "ng/mL"
    assert item["reference_range"]["high"] == pytest.approx(100.0)
    assert any("could not be parsed" in warning for warning in body["warnings"])


def test_respects_an_inverted_reference_bound(client: TestClient, image_bytes: Any) -> None:
    """HDL 38 against "> 40" is LOW. Reading it as HIGH inverts the clinical meaning."""
    body = _post(client, "lipid_profile.png", image_bytes("lipid_profile")).json()
    hdl = next(i for i in body["results"] if i["test_name"] == "HDL Cholesterol")

    assert hdl["flag"] == "low"


def test_accepts_a_cropped_report_and_nulls_its_metadata(
    client: TestClient, image_bytes: Any
) -> None:
    body = _post(client, "partial_crop.png", image_bytes("partial_crop")).json()

    assert len(body["results"]) == 2
    assert all(body["meta"][key] is None for key in META_KEYS)
    assert any("patient name" in warning for warning in body["warnings"])


def test_refuses_a_non_lab_image_without_fabricating_results(
    client: TestClient, image_bytes: Any
) -> None:
    response = _post(client, "non_lab_receipt.png", image_bytes("non_lab_receipt"))

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "not_a_lab_report"
    assert body["error"]["details"]["lines_detected"] == 13
    assert body["error"]["details"]["result_rows_detected"] == 0
    # Nothing from the receipt leaks into the response as a result.
    assert "Basmati" not in response.text


def test_distinguishes_an_unreadable_image_from_a_wrong_document(
    client: TestClient, image_bytes: Any
) -> None:
    """Different fixes: retake the photograph vs photograph something else."""
    response = _post(client, "blank_page.png", image_bytes("blank_page"))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unreadable_image"


@pytest.mark.parametrize(
    ("content_type", "status"),
    [
        ("image/png", 200),
        ("image/jpeg", 200),
        # Browsers really do send this for valid images, so it must be accepted.
        ("application/octet-stream", 200),
        # An allowed extension contradicted by the content type (D8).
        ("audio/wav", 400),
        ("application/pdf", 400),
    ],
)
def test_media_type_policy_at_the_boundary(
    client: TestClient, image_bytes: Any, content_type: str, status: int
) -> None:
    response = _post(client, "cbc_report.png", image_bytes("cbc_report"), content_type)
    assert response.status_code == status
