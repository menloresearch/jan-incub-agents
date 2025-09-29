import os
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security.utils import get_authorization_scheme_param


def verify_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    expected_key = os.getenv("MEM0_API_KEY")
    if not expected_key:
        # If no key configured, treat as open access for development.
        return ""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, token = get_authorization_scheme_param(authorization)
    if scheme.lower() != "token" or token != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token"
        )
    return token


def require_api_key(_: str = Depends(verify_api_key)) -> None:
    """Dependency that enforces API key validation."""
