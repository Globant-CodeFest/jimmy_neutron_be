from pydantic import BaseModel


class CoinStatistic(BaseModel):
    symbol: str
    increase: float


def mock_coin_list_statistic():
    return [
        CoinStatistic(symbol="BTC", increase=0.1),
        CoinStatistic(symbol="ETH", increase=0.2),
        CoinStatistic(symbol="ADA", increase=-0.3),
    ]
