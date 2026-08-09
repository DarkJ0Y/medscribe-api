"""HTTP layer: routing, request/response models, input validation.

LAYER CONTRACT
    MAY import: fastapi, pydantic, api.*, services.* (services + domain types)
    MUST NOT:   hold business logic, parsing/normalization rules, or provider
                SDK calls.
"""
