"""Outbound adapters -- the ONLY layer permitted to touch provider SDKs.

LAYER CONTRACT
    MAY import: provider SDKs (openai, pytesseract, PIL, ...), services.ports,
                services.domain, config.settings
    MUST NOT:   import fastapi, or hold domain/business rules.
"""
