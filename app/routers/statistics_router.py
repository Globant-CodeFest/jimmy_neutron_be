from fastapi import APIRouter
from app.models.coin_statictic import mock_coin_list_statistic

router = APIRouter()

@router.get("/statistics")
def get_average_statistic():
    total = 0
    forecast = mock_coin_list_statistic()
    for i in range(len(forecast)):
        total += forecast[i].increase
    return total / len(forecast) * 100


@router.get("/statistics/{symbol}")
def get_coin_statistic(symbol: str):
    forecast = mock_coin_list_statistic()
    for i in range(len(forecast)):
        if forecast[i].symbol == symbol:
            return forecast[i].increase * 100
    return

# Path: app\routers\statistics_router.py