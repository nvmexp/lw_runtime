from socket import socket, AF_INET, SOCK_DGRAM

from common.Struct import Struct
from dcgm_fluentd import DcgmFluentd

def test_send_to_fluentd():
    # Can't create a proper closure in Python, so we create an object which acts
    # as a closure
    namespace = Struct(message=None, dest=None)

    def mysendto(_message, _dest):
        namespace.message = _message
        namespace.dest = _dest

    mysock = Struct(sendto=mysendto)

    dr = DcgmFluentd('FAKE_HOST', 101010)

    # Assert that we are sending over UDP
    assert dr.m_sock.family == AF_INET
    assert dr.m_sock.type == SOCK_DGRAM

    dr.m_sock = mysock

    dr.SendToFluentd('message')

    assert(namespace.message == 'message')
    assert(namespace.dest == ('FAKE_HOST', 101010))

def test_lwstom_json_handler():
    namespace = Struct(arg=None)

    def MySendToFluentd(json):
        namespace.arg = json # pylint: disable=no-member

    dr = DcgmFluentd('FAKE_HOST', 101010)
    dr.SendToFluentd = MySendToFluentd

    dr.LwstomJsonHandler('value')
    assert namespace.arg == 'value' # pylint: disable=no-member
