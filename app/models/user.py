from pydantic import BaseModel


class User(BaseModel):
    username: str
    wallet: list = []


def get_mock_user():
    return User(
        username="test_user",
        wallet={"BTC", "ETH", "LTC"})

# Path: app\routers\user_router.py
