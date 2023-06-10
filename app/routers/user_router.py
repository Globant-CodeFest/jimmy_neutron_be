from fastapi import APIRouter
from app.models.user import get_mock_user
router = APIRouter()

@router.get("/user")
def get_user():
    return get_mock_user()
