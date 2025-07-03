'''
Tests written for test_utils.py
'''
import test_utils
import dcgm_agent_internal

@test_utils.run_with_embedded_host_engine()
def test_utils_run_with_embedded_host_engine(handle):
    '''
    Sanity test for running with an embedded host engine
    '''
    assert(handle.value == dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), \
            "Expected embedded handle %s but got %s" % \
            (hex(dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), hex(handle.value))
            
@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
def test_utils_run_with_standalone_host_engine(handle):
    '''
    Sanity test for running with a standalone host engine
    '''
    assert(handle.value != dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value), \
            "Expected a handle different from the embedded one %s" % \
            hex(dcgm_agent_internal.DCGM_EMBEDDED_HANDLE.value)