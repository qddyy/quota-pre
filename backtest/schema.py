from datetime import datetime, timedelta
from collections import defaultdict


class futureAccount:
    base: float
    pool: dict[str:int]
    fu_overtoday_fee: float
    fu_intoday_fee: float
    timeline: str
    portfolio_value: float
    benchmark: str
    factors: list[str,]
    cash: float
    stand: str
    transactions: dict[str, list[str:dict]]
    current_date: str

    def __init__(
        self,
        current_date: str = "20220913",
        base: float = 10000000,
        pool: dict[str:dict] = {},
        fu_overtoday_fee: float = 0.000024,
        fu_intoday_fee: float = 0.00035,
    ):
        self.current_date = current_date
        self.base = base
        self.cash = base
        self.pool = pool
        self.fu_overtoday_fee = fu_overtoday_fee
        self.fu_intoday_fee = fu_intoday_fee
        self.portfolio_value = self.calculate_portfolio_value()
        self.transactions = defaultdict(list)

    def update_date(self, num: int):
        date_format = "%Y%m%d"
        old_date = datetime.strptime(self.current_date, date_format)
        new_date = (old_date + timedelta(days=num)).strftime(date_format)
        self.current_date = new_date

    def order(self, symbol: str, volumes: int, price: float):
        "按照量买卖证券,限制买空不限制卖空"
        try:
            if volumes > 0:
                if self.cash < price * (1 + self.fu_overtoday_fee):
                    return
                elif price * volumes * (1 + self.fu_overtoday_fee) > self.cash:
                    volumes = int(self.cash / price / (1 + self.fu_overtoday_fee))
            self.cash -= volumes * price
            if symbol not in self.pool:
                self.pool[symbol] = {
                    "code": symbol,
                    "volume": volumes,
                    "price": price,
                }
            else:
                self.pool[symbol]["volume"] += volumes
                self.pool[symbol]["price"] = price
            if self.pool[symbol]["volume"] == 0:
                del self.pool[symbol]
            self.cash -= abs(volumes * price) * self.fu_overtoday_fee
            self.calculate_portfolio_value()
            self.transactions[self.current_date].append(
                {
                    "code": symbol,
                    "volume": volumes,
                    "price": price,
                }
            )
        except Exception as e:
            print(e)

    def order_to(self, symbol: str, volumes: float, price: float):
        """购买股票到满仓的指定比例，volumes需要大于零"""
        try:
            if self.cash < price and volumes < 0:
                return
            if volumes <= 0:
                try:
                    total_volume = self.pool[symbol]["volume"]
                    volumes = -int(total_volume * (1 - volumes))
                except KeyError:
                    volumes = int(self.cash * volumes / price)
            else:
                volumes = int(self.cash / price * volumes)
            self.order(symbol, volumes, price)
        except Exception as e:
            print(e)

    def update_price(self, price_dic: dict[str:float]) -> None:
        for stock in self.pool.keys():
            self.pool[stock]["price"] = price_dic[stock]
            self.calculate_portfolio_value()

    def calculate_portfolio_value(self):
        """
        计算投资组合价值。计算之前需要先更新证券价格
        """
        try:
            if len(self.pool) == 0:
                return self.cash
            else:
                security_value = 0
                for symbol in self.pool.keys():
                    price = self.pool[symbol]["price"]
                    volume = self.pool[symbol]["volume"]
                    security_value += price * volume
                self.portfolio_value = self.cash + security_value
                return self.portfolio_value
        except Exception as e:
            print(e)
