from fastapi import APIRouter, Depends, Header
from typing import Optional
import time

router = APIRouter(prefix="/api/history", tags=["History"])


async def get_user_id(authorization: Optional[str] = Header(None)):
    if not authorization:
        return None
    from firebase_auth import verify_token
    decoded = await verify_token(authorization)
    return decoded.get("uid")


@router.get("")
async def get_history(user_id: Optional[str] = Depends(get_user_id)):
    if not user_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Authentication required")
    from database import get_user_predictions
    predictions = get_user_predictions(user_id)
    return {"predictions": predictions, "count": len(predictions)}
