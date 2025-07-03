from common.Struct import Struct

def test_struct():
    s = Struct(field='field')
    assert s.field == 'field' # pylint: disable=no-member

    try:
        i = s.notfield # pylint: disable=no-member
        assert False # notfield should not exist
    except:
        pass
