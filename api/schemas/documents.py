"""POST /api/v1/documents/extract response models.

ExtractResponse -> meta: ReportMeta, results: list[LabResultItem]
ReportMeta      -> patient_name, age, sex, report_date, lab_name, reference_no
LabResultItem   -> test_name, value, unit, reference_range, flag, raw_line

``value`` and ``reference_range`` are **objects, not scalars**. A lab value is not
always a number -- ``<0.5``, ``0.8 - 1.2`` and ``Not Done`` all appear on real
reports -- so the wire format carries the structure that was recovered along with
the verbatim source text, instead of flattening everything to a float and losing
the distinction between "0.5" and "less than 0.5".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from services.domain import (
    Comparator,
    ExtractedReport,
    LabResult,
    NumericValue,
    ReportMeta,
    ResultFlag,
    ValueKind,
)


class NumericValueModel(BaseModel):
    """A measurement, or an explicit admission that it could not be parsed."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"raw": "11.2", "kind": "single", "parsed": True, "value": 11.2},
                {
                    "raw": "<0.5",
                    "kind": "bounded",
                    "parsed": True,
                    "value": 0.5,
                    "comparator": "<",
                },
                {"raw": "0.8 - 1.2", "kind": "range", "parsed": True, "low": 0.8, "high": 1.2},
                {"raw": "2S.4", "kind": "unparsed", "parsed": False},
            ]
        }
    )

    raw: str = Field(
        description="The source text, verbatim. Always present, even when nothing parsed."
    )
    kind: ValueKind = Field(
        description=(
            "Which fields below are populated. 'single' -> value; 'bounded' -> "
            "comparator + value; 'range' -> low + high; 'unparsed' -> none of them."
        )
    )
    parsed: bool = Field(
        description=(
            "False when the text could not be read as any known numeric form. The "
            "value is preserved in `raw` and no number is guessed -- an ambiguous "
            "clinical value is reported as unparsed rather than repaired."
        )
    )
    value: float | None = Field(default=None, description="Set for 'single' and 'bounded'.")
    comparator: Comparator | None = Field(
        default=None, description="Set for 'bounded': one of <, <=, >, >=."
    )
    low: float | None = Field(default=None, description="Lower endpoint, set for 'range'.")
    high: float | None = Field(default=None, description="Upper endpoint, set for 'range'.")

    @classmethod
    def from_domain(cls, value: NumericValue) -> NumericValueModel:
        return cls(
            raw=value.raw,
            kind=value.kind,
            parsed=value.is_parsed,
            value=value.value,
            comparator=value.comparator,
            low=value.low,
            high=value.high,
        )


class ReportMetaModel(BaseModel):
    """Header fields of the report. Every field is optional.

    Phone photographs crop and letterheads vary, so a field that was not found is
    reported as ``null`` rather than inferred from surrounding text.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_name": "MD. RAFIQUL ISLAM",
                "age": "45 Y",
                "sex": "Male",
                "report_date": "12/03/2024",
                "lab_name": "POPULAR DIAGNOSTIC CENTRE LTD.",
                "reference_no": "PDC-2024-0098871",
            }
        }
    )

    patient_name: str | None = None
    age: str | None = Field(
        default=None,
        description=(
            "Verbatim as printed -- '45 Y', '6 M', '3 Days'. Deliberately a string: "
            "coercing '3 M' to the integer 3 would turn a three-month-old infant "
            "into a three-year-old child."
        ),
    )
    sex: str | None = Field(default=None, description="Verbatim as printed.")
    report_date: str | None = Field(
        default=None,
        description=(
            "Verbatim as printed. Not parsed into a date, because 05-04-2024 is "
            "ambiguous between day-first and month-first conventions."
        ),
    )
    lab_name: str | None = None
    reference_no: str | None = None

    @classmethod
    def from_domain(cls, meta: ReportMeta) -> ReportMetaModel:
        return cls(
            patient_name=meta.patient_name,
            age=meta.age,
            sex=meta.sex,
            report_date=meta.report_date,
            lab_name=meta.lab_name,
            reference_no=meta.reference_no,
        )


class LabResultItem(BaseModel):
    """One row of the report."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_name": "Haemoglobin",
                "value": {"raw": "11.2", "kind": "single", "parsed": True, "value": 11.2},
                "unit": "g/dL",
                "reference_range": {
                    "raw": "13.0 - 17.0",
                    "kind": "range",
                    "parsed": True,
                    "low": 13.0,
                    "high": 17.0,
                },
                "flag": "low",
                "raw_line": "Haemoglobin          11.2        g/dl          13.0 - 17.0",
            }
        }
    )

    test_name: str
    value: NumericValueModel
    unit: str | None = Field(
        default=None,
        description=(
            "Standardized spelling where recognized ('gm/dl' -> 'g/dL', '/cumm' -> "
            "'/µL'), otherwise passed through unchanged. Only identities with a "
            "conversion factor of exactly 1 are ever applied -- magnitudes are "
            "never rescaled."
        ),
    )
    reference_range: NumericValueModel | None = Field(
        default=None, description="Null when the report printed no reference range."
    )
    flag: ResultFlag = Field(
        description=(
            "Position against the reference range: 'high', 'low', 'normal', or "
            "'unknown'. 'unknown' whenever the comparison is not certain -- never "
            "defaulted to 'normal', which is the answer least likely to be "
            "double-checked."
        )
    )
    raw_line: str = Field(
        description=(
            "The exact OCR line this row came from, byte for byte. Whatever "
            "normalization did or failed to do, this is the audit trail: it can be "
            "compared against the paper report directly."
        )
    )

    @classmethod
    def from_domain(cls, result: LabResult) -> LabResultItem:
        return cls(
            test_name=result.test_name,
            value=NumericValueModel.from_domain(result.value),
            unit=result.unit,
            reference_range=(
                NumericValueModel.from_domain(result.reference_range)
                if result.reference_range is not None
                else None
            ),
            flag=result.flag,
            raw_line=result.raw_line,
        )


class ExtractResponse(BaseModel):
    """Structured extraction of one lab-report photograph."""

    meta: ReportMetaModel
    results: list[LabResultItem]
    provider: str = Field(description="OCR adapter that produced these lines.")
    ocr_mean_confidence: float | None = Field(
        default=None,
        description=(
            "Mean OCR confidence across the page, where the engine reports one. Low "
            "values are a signal to review `raw_line` before trusting the parse."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal notes -- how many values could not be parsed, how many "
            "row-like lines were omitted, whether metadata was absent."
        ),
    )

    @classmethod
    def from_domain(cls, report: ExtractedReport) -> ExtractResponse:
        return cls(
            meta=ReportMetaModel.from_domain(report.meta),
            results=[LabResultItem.from_domain(item) for item in report.results],
            provider=report.provider,
            ocr_mean_confidence=report.ocr_mean_confidence,
            warnings=list(report.warnings),
        )
