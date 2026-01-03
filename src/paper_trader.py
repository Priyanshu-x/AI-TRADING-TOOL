import uuid
from datetime import datetime
from .logger import setup_logger

logger = setup_logger(__name__)

class PaperTradingEngine:
    def __init__(self, db_manager, initial_capital=100000.0):
        """
        Initializes the Paper Trading Engine.
        
        Args:
            db_manager (SignalDatabaseManager): To persist orders.
            initial_capital (float): Starting virtual cash.
        """
        self.db_manager = db_manager
        self.initial_capital = initial_capital
        # In a real app, cash balance should also be persisted in DB.
        # For this prototype, we re-calculate it from trade history + initial.
        self.current_cash = self._calculate_current_cash()

    def _calculate_current_cash(self):
        """Re-calculates available cash based on realized P&L."""
        # This is a simplified logic. Real logic would sum (Entry Cost) and (Exit Proceeds).
        # Wrapper: Start with Initial.
        # Subtract cost of OPEN trades.
        # Add P&L of CLOSED trades (adjusted for cost).
        
        # Simplified for Prototype:
        # Cash = Initial + Sum(Realized PnL of Closed Trades) - Sum(Cost of Open Trades)
        
        # Fetch all orders (we need a method for this, or just logic here)
        # For now, we'll assume a session-based tracking or add a proper 'accounts' table later.
        # Let's stick to a simple in-memory session override for now, assuming persistence is for records.
        return self.initial_capital

    def place_order(self, signal, quantity):
        """
        Opens a position based on a signal.
        """
        cost = signal['entry_price'] * quantity
        if cost > self.current_cash:
            logger.warning(f"Insufficient funds for trade. Cost: {cost}, Cash: {self.current_cash}")
            return {"status": "failed", "reason": "Insufficient Funds"}

        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "symbol": signal['symbol'],
            "trade_type": signal.get('trade_type', 'STOCK'),
            "action": signal['action'], # BUY or BUY CALL/PUT
            "quantity": quantity,
            "entry_price": signal['entry_price'],
            "status": "OPEN",
            "timestamp": datetime.now().isoformat(),
            "strategy_ref": signal.get('trade_type')
        }
        
        self.db_manager.save_order(order)
        self.current_cash -= cost
        logger.info(f"Paper Trade Executed: {signal['symbol']} - {quantity} qty @ {signal['entry_price']}")
        return {"status": "success", "order_id": order_id}

    def get_portfolio_status(self, current_prices):
        """
        Returns the current portfolio status including Unrealized P&L.
        current_prices: dict {symbol: price}
        """
        open_orders = self.db_manager.get_open_orders()
        total_value = self.current_cash
        holdings_value = 0
        unrealized_pnl = 0
        
        detailed_positions = []
        
        for order in open_orders:
            symbol = order['symbol']
            qty = order['quantity']
            entry = order['entry_price']
            
            current_price = current_prices.get(symbol, entry) # Default to entry if no live price
            
            # PnL Logic
            if "BUY" in order['action']:
                pnl = (current_price - entry) * qty
            else: # SELL/SHORT (not fully supported yet)
                pnl = (entry - current_price) * qty
            
            market_val = current_price * qty
            holdings_value += market_val
            unrealized_pnl += pnl
            
            detailed_positions.append({
                "Symbol": symbol,
                "Qty": qty,
                "Entry": entry,
                "LTP": current_price,
                "PnL": pnl,
                "Value": market_val
            })
            
        total_equity = self.current_cash + holdings_value # Approximation
        
        return {
            "Cash": self.current_cash,
            "Equity": total_equity,
            "Holdings Value": holdings_value,
            "Unrealized PnL": unrealized_pnl,
            "Positions": detailed_positions
        }
        
    def check_exits(self, current_prices, target_pct=0.04, stop_loss_pct=0.02):
        """
        Checks open positions against Stop Loss and Target Price.
        Closes them if triggered.
        """
        open_orders = self.db_manager.get_open_orders()
        
        for order in open_orders:
            symbol = order['symbol']
            if symbol not in current_prices: continue
            
            ltp = current_prices[symbol]
            entry = order['entry_price']
            qty = order['quantity']
            
            # Determine Exit Conditions
            is_target = ltp >= entry * (1 + target_pct)
            is_sl = ltp <= entry * (1 - stop_loss_pct)
            
            if is_target or is_sl:
                exit_reason = "TARGET" if is_target else "STOP_LOSS"
                pnl = (ltp - entry) * qty
                
                # Close Order
                order['status'] = 'CLOSED'
                order['exit_price'] = ltp
                order['pnl'] = pnl
                order['exit_timestamp'] = datetime.now().isoformat()
                
                self.db_manager.save_order(order)
                self.current_cash += (ltp * qty) # Credit proceeds
                
                logger.info(f"Paper Trade Closed ({exit_reason}): {symbol} @ {ltp}. PnL: {pnl}")
