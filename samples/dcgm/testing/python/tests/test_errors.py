import pydcgm
import dcgm_field_helpers
import dcgm_structs
import dcgm_structs_internal
import test_utils
import dcgm_errors

def helper_test_dcgm_error_get_priority(handle, gpuIds):
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
    assert prio == dcgm_errors.DCGM_ERROR_ISOLATE, "DBE errors should be an isolate priority, but found %d" % prio
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_LWML_API)
    assert prio == dcgm_errors.DCGM_ERROR_MONITOR, "DBE errors should be a monitor priority, but found %d" % prio

    prio = dcgm_errors.dcgmErrorGetPriorityByCode(-1)
    assert prio == dcgm_errors.DCGM_ERROR_UNKNOWN, "Non-existent error should be unknown priority, but found %d" % prio
    prio = dcgm_errors.dcgmErrorGetPriorityByCode(dcgm_errors.DCGM_FR_ERROR_SENTINEL)
    assert prio == dcgm_errors.DCGM_ERROR_UNKNOWN, "The sentinel error error should be unknown priority, but found %d" % prio

def helper_test_dcgm_error_get_msg(handle, gpuIds):
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD)
    assert msg == dcgm_errors.DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD_MSG, \
           "Expected '%s' as msg, but found '%s'" % (dcgm_errors.DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD_MSG, msg)

    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH)
    assert msg == dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH_MSG, \
           "Expected '%s' as msg, but found '%s'" % (dcgm_errors.DCGM_FR_DEVICE_COUNT_MISMATCH_MSG, msg)
    
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(dcgm_errors.DCGM_FR_ERROR_SENTINEL)
    assert not msg, "The sentinel error error should be empty, but found %s" % msg
    
    msg = dcgm_errors.dcgmErrorGetFormatMsgByCode(-1)
    assert not msg, "Non-existent error should be empty, but found %s" % msg

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_priority_embedded(handle, gpuIds):
    helper_test_dcgm_error_get_priority(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_priority_standalone(handle, gpuIds):
    helper_test_dcgm_error_get_priority(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_msg_embedded(handle, gpuIds):
    helper_test_dcgm_error_get_msg(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_error_get_msg_standalone(handle, gpuIds):
    helper_test_dcgm_error_get_msg(handle, gpuIds)
