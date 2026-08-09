"""Domain core: business logic, normalizers, orchestration.

LAYER CONTRACT (mechanically enforced by tests/unit/test_layer_boundaries.py)
    MAY import: stdlib, pydantic, services.*
    MUST NOT:   import fastapi or starlette (Request, Response, UploadFile,
                HTTPException, Depends), nor any provider SDK (openai,
                pytesseract, easyocr, PIL, ...).

The outside world is reached only through the Protocols in services.ports, so
the dependency arrow always points inward: api -> services <- adapters.
"""
