from functools import wraps
from test_utils import skip_test
import logger

MOCK_MSG = "mock not installed. Please run \"pip install mock\""

try:
    import mock
    MOCK_INSTALLED = True
except ImportError:
    logger.warning(MOCK_MSG)
    MOCK_INSTALLED = False

def skip_test_if_no_mock():
    '''
    Returns a decorator for functions. The decorator skips
    the test in the provided function if mock is not installed
    '''
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwds):
            if MOCK_INSTALLED:
                fn(*args, **kwds)
            else:
                skip_test(MOCK_MSG)
        return wrapper
    return decorator

# Do not use this class directly
class _MaybeMock:
    def __call__(self, *args, **kwds):
        return skip_test_if_no_mock()

    def __getattr__(self, attr):
        if (MOCK_INSTALLED):
            return getattr(mock, attr)
        return self

maybemock = _MaybeMock()
