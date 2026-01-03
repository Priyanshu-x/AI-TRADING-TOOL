import pytest
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.paper_trader import PaperTradingEngine
from src.signal_database_manager import SignalDatabaseManager

@pytest.fixture
def mock_db():
    db = MagicMock(spec=SignalDatabaseManager)
    db.get_open_orders.return_value = []
    return db

def test_paper_trader_init(mock_db):
    engine = PaperTradingEngine(mock_db, initial_capital=50000)
    assert engine.current_cash == 50000

def test_place_order_sufficient_funds(mock_db):
    engine = PaperTradingEngine(mock_db, initial_capital=10000)
    signal = {'symbol': 'TEST', 'action': 'BUY', 'entry_price': 100, 'trade_type': 'STOCK'}
    
    result = engine.place_order(signal, quantity=10) # Cost 1000
    
    assert result['status'] == 'success'
    assert engine.current_cash == 9000
    mock_db.save_order.assert_called_once()
    assert mock_db.save_order.call_args[0][0]['status'] == 'OPEN'

def test_place_order_insufficient_funds(mock_db):
    engine = PaperTradingEngine(mock_db, initial_capital=500)
    signal = {'symbol': 'TEST', 'action': 'BUY', 'entry_price': 100, 'trade_type': 'STOCK'}
    
    result = engine.place_order(signal, quantity=10) # Cost 1000
    
    assert result['status'] == 'failed'
    assert engine.current_cash == 500
    mock_db.save_order.assert_not_called()

def test_exit_logic_target_hit(mock_db):
    engine = PaperTradingEngine(mock_db, initial_capital=0)
    # Mock an open order
    open_order = {
        'order_id': '123', 'symbol': 'TEST', 'quantity': 10, 'entry_price': 100, 
        'status': 'OPEN', 'action': 'BUY'
    }
    mock_db.get_open_orders.return_value = [open_order]
    
    # Live price hits target (100 + 4% = 104)
    live_prices = {'TEST': 105} 
    
    engine.check_exits(live_prices, target_pct=0.04)
    
    # Verify save_order called with CLOSED status
    saved_order = mock_db.save_order.call_args[0][0]
    assert saved_order['status'] == 'CLOSED'
    assert saved_order['exit_price'] == 105
    assert saved_order['pnl'] == (105 - 100) * 10
    assert engine.current_cash == 1050 # Proceeds credited
