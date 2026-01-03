import datetime
import pytz
from .logger import setup_logger

logger = setup_logger(__name__)

class MarketTimingManager:
    def __init__(self, market_timezone='Asia/Calcutta'):
        self.market_timezone = pytz.timezone(market_timezone)
        self.market_open_time = datetime.time(9, 15)  # 9:15 AM IST
        self.market_close_time = datetime.time(15, 30) # 3:30 PM IST
        # Hardcoded holidays for 2025 (example, should be updated annually or fetched from API)
        self.holidays = [
            datetime.date(2025, 1, 26),  # Republic Day
            datetime.date(2025, 3, 14),  # Maha Shivaratri
            datetime.date(2025, 3, 25),  # Holi
            datetime.date(2025, 4, 18),  # Good Friday
            datetime.date(2025, 5, 1),   # Maharashtra Day
            datetime.date(2025, 8, 15),  # Independence Day
            datetime.date(2025, 10, 2),  # Gandhi Jayanti
            datetime.date(2025, 10, 24), # Diwali (Laxmi Pujan)
            datetime.date(2025, 12, 25)  # Christmas
        ]

    def get_current_market_time(self):
        """Returns the current time in the market's timezone."""
        utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        market_time = utc_now.astimezone(self.market_timezone)
        return market_time

    def is_market_open(self):
        """Checks if the market is currently open."""
        now = self.get_current_market_time()
        
        if self.is_holiday(now.date()):
            logger.info(f"Market is closed: Today ({now.date()}) is a holiday.")
            return False
        
        if now.weekday() >= 5: # Saturday or Sunday
            logger.info(f"Market is closed: Today ({now.date()}) is a weekend.")
            return False

        if self.market_open_time <= now.time() <= self.market_close_time:
            logger.info(f"Market is open: Current time {now.time()} is within {self.market_open_time}-{self.market_close_time}.")
            return True
        else:
            logger.info(f"Market is closed: Current time {now.time()} is outside {self.market_open_time}-{self.market_close_time}.")
            return False

    def is_market_closed_for_day(self):
        """
        Checks if the market is closed for the day (after market close time)
        and it's not a holiday or weekend.
        This is for triggering post-market validation.
        """
        now = self.get_current_market_time()

        if self.is_holiday(now.date()) or now.weekday() >= 5:
            return False # If it's a holiday or weekend, it's not "closed for the day" in the sense of post-market activity

        # Market is closed for the day if current time is after market close
        if now.time() > self.market_close_time:
            logger.info(f"Market is closed for the day: Current time {now.time()} is after {self.market_close_time}.")
            return True
        
        logger.info(f"Market is not yet closed for the day: Current time {now.time()}.")
        return False

    def is_holiday(self, date_to_check):
        """Checks if a given date is a market holiday."""
        # In a real application, this would fetch from an API or a more robust list
        is_a_holiday = date_to_check in self.holidays
        if is_a_holiday:
            logger.info(f"Date {date_to_check} is a market holiday.")
        return is_a_holiday

    def get_next_trading_day(self, current_date):
        """Calculates the next trading day, skipping weekends and holidays."""
        next_day = current_date + datetime.timedelta(days=1)
        while next_day.weekday() >= 5 or self.is_holiday(next_day):
            next_day += datetime.timedelta(days=1)
        return next_day

if __name__ == "__main__":
    manager = MarketTimingManager()

    print(f"Current Market Time: {manager.get_current_market_time()}")
    print(f"Is Market Open? {manager.is_market_open()}")
    print(f"Is Market Closed for Day? {manager.is_market_closed_for_day()}")
    
    # Test holiday
    test_holiday = datetime.date(2025, 1, 26)
    print(f"Is {test_holiday} a holiday? {manager.is_holiday(test_holiday)}")

    # Test next trading day
    today = datetime.date.today()
    next_trading_day = manager.get_next_trading_day(today)
    print(f"Next trading day after {today}: {next_trading_day}")

    # Simulate a weekend
    saturday = datetime.date(2025, 11, 22) # A Saturday
    print(f"Is {saturday} a holiday? {manager.is_holiday(saturday)}") # Should be False unless it's also a holiday
    # To properly test is_market_open for a weekend, you'd need to mock datetime.datetime.now()
    # For now, rely on the weekday check.