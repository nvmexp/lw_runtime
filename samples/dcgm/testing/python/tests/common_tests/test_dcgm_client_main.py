import signal
from _test_helpers import maybemock

from common import dcgm_client_main as m

@maybemock.patch('__builtin__.exit')
def test_exit_handler(mock_exit):
    m.exit_handler(None, None)
    mock_exit.assert_called()

@maybemock.patch('signal.signal')
def test_initialize_signal_handlers(mock_signal):
    m.initialize_signal_handlers()
    assert mock_signal.mock_calls[0][1] == (signal.SIGINT, m.exit_handler)
    assert mock_signal.mock_calls[1][1] == (signal.SIGTERM, m.exit_handler)
