from .logger import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    """
    Manages risk calculations including position sizing and R-multiple tracking.
    """
    def __init__(self, default_account_size=100000.0, default_risk_per_trade_pct=1.0):
        self.account_size = default_account_size
        self.risk_per_trade_pct = default_risk_per_trade_pct

    def calculate_position_size(self, entry_price, stop_loss_price, account_size=None, risk_pct=None):
        """
        Calculates the position size (number of shares/lots) based on risk parameters.
        
        Args:
            entry_price (float): The entry price for the trade.
            stop_loss_price (float): The stop loss price.
            account_size (float, optional): Total capital available. Defaults to init value.
            risk_pct (float, optional): Percentage of capital to risk. Defaults to init value.
            
        Returns:
            dict: Contains 'quantity', 'capital_required', 'risk_amount', 'risk_per_share'.
        """
        acc_size = account_size if account_size is not None else self.account_size
        risk = risk_pct if risk_pct is not None else self.risk_per_trade_pct

        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error(f"Invalid prices for sizing: Entry={entry_price}, SL={stop_loss_price}")
            return None

        # Calculate risk amount per share (absolute difference)
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            logger.error("Entry price and Stop Loss are identical. Cannot calculate size.")
            return None

        # Total money to risk
        total_risk_amount = acc_size * (risk / 100.0)

        # Quantity
        quantity = int(total_risk_amount / risk_per_share)
        
        # Capital required
        capital_required = quantity * entry_price

        # Check if we have enough capital (this is a basic check, doesn't account for margin)
        if capital_required > acc_size:
            logger.warning(f"Calculated position size ({quantity}) requires {capital_required:.2f}, which exceeds account size {acc_size}. Capping at max affordable.")
            quantity = int(acc_size / entry_price)
            capital_required = quantity * entry_price
            # Recalculate actual risk taking this cap into account
            total_risk_amount = quantity * risk_per_share

        result = {
            "quantity": quantity,
            "capital_required": round(capital_required, 2),
            "total_risk_amount": round(total_risk_amount, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_percentage_actual": round((total_risk_amount / acc_size) * 100, 2)
        }
        
        logger.debug(f"Risk Calculation: {result}")
        return result

if __name__ == "__main__":
    # Test
    rm = RiskManager(account_size=200000, default_risk_per_trade_pct=2)
    print(rm.calculate_position_size(entry_price=100, stop_loss_price=95))
