# test the policy manager for DCGM

import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent
import dcgm_agent_internal
import dcgmvalue
import logger
import test_utils
import dcgm_fields
import time
from ctypes import *
import sys
import os
import pprint
import dcgm_internal_helpers
import dcgm_errors

def helper_dcgm_health_set_pcie(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == 0)

    newSystems = lwrrentSystems | dcgm_structs.DCGM_HEALTH_WATCH_PCIE

    groupObj.health.Set(newSystems)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == newSystems)

    #Set it back to 0 and validate it
    groupObj.health.Set(0)
    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == 0)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_health_set_pcie_embedded(handle):
    helper_dcgm_health_set_pcie(handle)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_health_set_pcie_standalone(handle):
    helper_dcgm_health_set_pcie(handle)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_health_ilwalid_group_embedded(handle):
    '''
    Validate that group operations fail if a bogus group ID is provided
    '''
    ilwalidGroupId = c_void_p(99)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = pydcgm.DcgmGroup(handleObj, groupId=ilwalidGroupId)
    
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Get()

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)

    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_NOT_CONFIGURED)):
        groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

def helper_dcgm_health_check_pcie(handle, gpuIds):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE
    groupObj.health.Set(newSystems)
    
    #response = dcgm_structs.c_dcgmHealthResponse_v1()
    #response.version = dcgm_structs.dcgmHealthResponse_version1

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
            0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    response = groupObj.health.Check()
    # we expect that there will be no data here

    # inject an error into PCI
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
            10, 100) # set the injected data into the future
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_pcie_embedded(handle, gpuIds):
    helper_dcgm_health_check_pcie(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_pcie_standalone(handle, gpuIds):
    helper_dcgm_health_check_pcie(handle, gpuIds)

def helper_test_dcgm_health_check_mem_dbe(handle, gpuIds):
    """
    Verifies that the health check will fail if there's 1 DBE and it continues to be
    reported
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)
    
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            2, -50) # set the injected data to 50 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    # Make sure we got a failure
    assert (responseV1.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV1.gpu[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    # Make sure we got a failure
    assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV2.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    # Make sure we got a failure
    assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Give it the same failure 45 seconds ago and make sure we fail again
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            2, -45)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    
    # Make sure this is regarded as a continual failure
    assert (responseV1.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV1.gpu[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    # Make sure this is regarded as a continual failure
    assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV2.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    # Make sure we got a failure
    assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)

    # Make the failure count go down to zero. This should clear the error
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            0, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    assert (responseV1.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_mem_dbe(handle, gpuIds):
    helper_test_dcgm_health_check_mem_dbe(handle, gpuIds)

def helper_verify_dcgm_health_watch_mem_result(groupObj, errorCode, verifyFail=False, gpuId=0):
    """
    Verify that memory health check result is what was expected. If verifyFail is False, verify a pass result, 
    otherwise verify a failure oclwrred.
    """
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    if not verifyFail:
        assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)
        assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS)
        return
    
    assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV2.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    
    assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

def helper_reset_page_retirements(handle, gpuId=0, reset_sbe=False):
    """
    Helper function to reset non volatile page retirements.
    """
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
        0, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)
    if reset_sbe:
        ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            0, -30) # set the injected data to 30 seconds ago
        assert (ret == dcgm_structs.DCGM_ST_OK)

def helper_test_dcgm_health_check_mem_retirements(handle, gpuIds):
    """
    Verifies that the health check will fail when the number of non-volatile page retirements 
    match the failure criteria.

    Specifically tests the criteria given in DCGM-458.
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

    # Skip the test if the GPU is already failing health checks
    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    if responseV1.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        test_utils.skip_test("Skipping because GPU %s is unhealthy and we need a healthy GPU in order to test." % gpuId)
    
    ####### Tests #######
    #### Condition 1 ####
    ### Fail if the total number of page retirements (due to DBE or SBE) meets or exceeds 60
    ## Test 1: >= 60 page retirements total should fail
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            30, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            30, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_LIMIT, verifyFail=True,
                                               gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 2: 59 page retirements total should pass
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            10, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            49, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    #### Condition 2 ####
    ### Fail if > 15 page retirement due to DBEs AND more than 1 DBE page retirement in past week
    ## Test 1: 15 page retirements due to DBEs should pass
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            15, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 2: 16 page retirements due to DBE should fail (since all 16 are inserted in current week)
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            16, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we get a failure
    helper_verify_dcgm_health_watch_mem_result(groupObj, dcgm_errors.DCGM_FR_RETIRED_PAGES_DBE_LIMIT,
                                               verifyFail=True, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 3: 16 page retirements due to SBEs should pass
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
            16, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId, reset_sbe=True)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

    ## Test 4: 16 page retirements due to DBEs (with first 15 pages inserted more than 1 week ago, 
    # and 16th page inserted in current week) should pass
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            15, -604860) # set the injected data to 7 days and 1 minute ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
            1, -30) # set the injected data to 30 seconds ago
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we pass
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)
    # Reset the field and verify clean result
    helper_reset_page_retirements(handle, gpuId=gpuId)
    helper_verify_dcgm_health_watch_mem_result(groupObj, 0, gpuId=gpuId)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_mem_retirements_standalone(handle, gpuIds):
    helper_test_dcgm_health_check_mem_retirements(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_mem_retirements_embedded(handle, gpuIds):
    helper_test_dcgm_health_check_mem_retirements(handle, gpuIds)

def helper_test_dcgm_health_check_mem(handle, gpuIds):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)

 
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
            0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    # we expect that there will be no data here

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
            100, -40)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)

    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV2.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_PENDING_PAGE_RETIREMENTS)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)

    # Clear the error
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING,
            0, -35)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    # Make sure we've set the monitor frequency to less than 35 seconds - that will make us around 
    # half or less of the 60 seconds we give the data before calling it stale.
    cmFieldInfo = dcgm_agent_internal.dcgmGetCacheManagerFieldInfo(handle, gpuId, dcgm_fields.DCGM_FI_DEV_RETIRED_PENDING)
    assert cmFieldInfo.monitorFrequencyUsec < 35000000

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_mem_standalone(handle, gpuIds):
    helper_test_dcgm_health_check_mem(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_mem_embedded(handle, gpuIds):
    helper_test_dcgm_health_check_mem(handle, gpuIds)
    
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_health_set_thermal(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == 0)

    newSystems = lwrrentSystems | dcgm_structs.DCGM_HEALTH_WATCH_THERMAL

    groupObj.health.Set(newSystems)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == newSystems)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_check_thermal(handle, gpuIds):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_THERMAL
    groupObj.health.Set(newSystems)
    
    #response = dcgm_structs.c_dcgmHealthResponse_v1()
    #response.version = dcgm_structs.dcgmHealthResponse_version1

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                       dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    # we expect that there will be no data here
    #assert (dcgm_structs.DCGM_ST_OK == result or dcgm_structs.DCGM_ST_NO_DATA == result)

    # inject an error into thermal
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuIds[0],
                                                       dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION, 1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuIds[0])
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuIds[0])
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuIds[0])
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_CLOCK_THROTTLE_THERMAL)
    
@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_health_set_power(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == 0)

    newSystems = lwrrentSystems | dcgm_structs.DCGM_HEALTH_WATCH_POWER

    groupObj.health.Set(newSystems)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == newSystems)

def helper_check_health_response_v1(gpuIds, response):
    numErrors = 0
    if response.version == 0:
        numErrors += 1
        logger.error("bad response.version x%X" % response.version)
    if response.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        numErrors += 1
        logger.error("bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
    if response.gpuCount > len(gpuIds):
        numErrors += 1
        logger.error("bad response.gpuCount %d > %d" % (response.gpuCount, len(gpuIds)))
    
    for i in range(response.gpuCount):
        if not response.gpu[i].gpuId in gpuIds:
            numErrors += 1
            logger.error("gpuId %u at index %d was not in gpuIds %s" % (response.gpu[i].gpuId, i, str(gpuIds)))
        if response.gpu[i].overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
            numErrors += 1
            logger.error("gpuId %u at index %d has overallHealth of %d. Is this gpu unhealthy?" % (response.gpu[i].gpuId, i, response.gpu[i].overallHealth))
        if response.gpu[i].incidentCount != 0:
            numErrors += 1
            logger.error("gpuId %u at index %d has incidentCount of %d. Is this gpu unhealthy?" % (response.gpu[i].gpuId, i, response.gpu[i].incidentCount))
        for j in range(dcgm_structs.DCGM_HEALTH_WATCH_COUNT_V1):
            if response.gpu[i].systems[j].health != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
                numErrors += 1
                logger.error("gpuId %u at index %d, system %d (x%X) has health of %d. Is this gpu unhealthy?" % (response.gpu[i].gpuId, i, j, response.gpu[i].systems[j].system, response.gpu[i].systems[j].health))
            if len(response.gpu[i].systems[j].errorString) > 0:
                numErrors += 1
                logger.error("gpuId %u at index %d, system %d (x%X) has errorString %s" % (response.gpu[i].gpuId, i, j, response.gpu[i].systems[j].system, response.gpu[i].systems[j].errorString))
    
    assert numErrors == 0, "Errors were encountered. See above."

def helper_check_health_response_v2(gpuIds, response):
    numErrors = 0
    if response.version == 0:
        numErrors += 1
        logger.error("bad response.version x%X" % response.version)
    if response.overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
        numErrors += 1
        logger.error("bad response.overallHealth %d. Are these GPUs really healthy?" % response.overallHealth)
    if response.entityCount > len(gpuIds):
        numErrors += 1
        logger.error("bad response.gpuCount %d > %d" % (response.entityCount, len(gpuIds)))
    
    for i in range(response.entityCount):
        if not response.entities[i].entityId in gpuIds:
            numErrors += 1
            logger.error("entityId %u at index %d was not in gpuIds %s" % (response.entities[i].entityId, i, str(gpuIds)))
        if response.gpu[i].overallHealth != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
            numErrors += 1
            logger.error("entityId %u at index %d was had overallHealth of %d. Is this gpu unhealthy?" % (response.entities[i].entityId, i, response.entities[i].overallHealth))
        if response.gpu[i].incidentCount != 0:
            numErrors += 1
            logger.error("entityId %u at index %d was had incidentCount of %d. Is this gpu unhealthy?" % (response.entities[i].entityId, i, response.entities[i].incidentCount))
        for j in range(dcgm_structs.DCGM_HEALTH_WATCH_COUNT_V2):
            if response.gpu[i].systems[j].health != dcgm_structs.DCGM_HEALTH_RESULT_PASS:
                numErrors += 1
                logger.error("entityId %u at index %d, system %d (x%X) had health of %d. Is this gpu unhealthy?" % (response.entities[i].entityId, i, j, response.entities[i].systems[j].system, response.entities[i].systems[j].health))

            if response.version == dcgm_structs.dcgmHealthResponse_version2:
                if len(response.gpu[i].systems[j].errorString) > 0:
                    numErrors += 1
                    logger.error("entityId %u at index %d, system %d (x%X) had errorString %s" % (response.entities[i].entityId, i, j, response.entities[i].systems[j].system, response.entities[i].systems[j].errorString))
            else:
                # version 3
                if response.gpu[i].systems[j].errorCount != 0:
                    numErrors += 1
                    logger.error("Expected error count to be 0, but it is %d for entityId %u at index %d, system %d (x%X)" % (response.gpu[i].systems[j].errorCount, response.entities[i].entityId, i, j, response.entities[i].systems[j].system))
                    for e in range(0, response.gpu[i].systems[j].errorCount):
                        if response.gpu[i].systems[j].errors[e].msg:
                            numErrors += 1
                            logger.error("Expected no errors, but found '%s' for entityId %u at index %d, system %d (x%X)" % (response.gpu[i].systems[j].errors[e].msg, response.entities[i].entityId, i, j, response.entities[i].systems[j].system))
    
    assert numErrors == 0, "Errors were encountered. See above."

def helper_run_dcgm_health_check_sanity(handle, gpuIds, system_to_check):
    """
    Verifies that the DCGM health checks return healthy for all GPUs on live systems.
    """

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    
    groupObj = systemObj.GetGroupWithGpuIds('testgroup', gpuIds)
    groupObj.health.Set(system_to_check)

    systemObj.UpdateAllFields(1)

    #This will throw an exception on error
    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    #Check that our response comes back clean
    helper_check_health_response_v1(gpuIds, responseV1)
    helper_check_health_response_v2(gpuIds, responseV2)
    helper_check_health_response_v2(gpuIds, responseV3)

################ Start health sanity checks
# The health sanity checks verify that that the DCGM health checks return healthy for all GPUs on live systems. 
# Note: These tests can fail if a GPU is really unhealthy. We should give detailed feedback so that this is attributed 
# to the GPU and not the test

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_pcie(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_pcie_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_mem(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_MEM)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_mem_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_MEM)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_inforom(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_INFOROM)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_inforom_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_INFOROM)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_thermal(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_thermal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_THERMAL)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_power(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_POWER)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_power_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_POWER)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwlink(handle, gpuIds):
    #We will get false failures if any lwlinks are down on the GPUs
    test_utils.skip_test_if_any_lwlinks_down(handle)
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwlink_standalone(handle, gpuIds):
    #We will get false failures if any lwlinks are down on the GPUs
    test_utils.skip_test_if_any_lwlinks_down(handle)
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwswitch_nonfatal(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwswitch_nonfatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwswitch_fatal(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL)

@test_utils.run_with_standalone_host_engine()
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_health_check_sanity_lwswitch_fatal_standalone(handle, gpuIds):
    helper_run_dcgm_health_check_sanity(handle, gpuIds, dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL)

################ End health sanity checks

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_check_power(handle, gpuIds):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)
    
    #response = dcgm_structs.c_dcgmHealthResponse_v1()
    #response.version = dcgm_structs.dcgmHealthResponse_version1

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
            0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    # we expect that there will be no data here

    # inject an error into power
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
            1000, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 1)
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)

    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuIds[0])
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_CLOCK_THROTTLE_POWER)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_check_lwlink(handle, gpuIds):
    helper_health_check_lwlink_error_counters(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_embedded_health_check_lwlink(handle, gpuIds):
    helper_health_check_lwlink_error_counters(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_standalone_health_set_lwlink(handle):
    """
    Verifies that the set/get path for the health monitor is working
    Checks for call errors are done in the bindings
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetDefaultGroup()

    groupObj.health.Set(0)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == 0)

    newSystems = lwrrentSystems | dcgm_structs.DCGM_HEALTH_WATCH_LWLINK

    groupObj.health.Set(newSystems)

    lwrrentSystems = groupObj.health.Get()
    assert (lwrrentSystems == newSystems)

def helper_health_check_lwlink_error_counters(handle, gpuIds):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    #We will get false failures if any lwlinks are down on the GPUs
    test_utils.skip_test_if_any_lwlinks_down(handle)
    
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_LWLINK
    groupObj.health.Set(newSystems)
    
    #response = dcgm_structs.c_dcgmHealthResponse_v1()
    #response.version = dcgm_structs.dcgmHealthResponse_version1

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    
    # we expect that there will be no data here
    # inject an error into LW Link
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       100, 10)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1), responseV1.gpuCount
    assert (responseV1.gpu[0].gpuId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV1.gpu[0].gpuId, gpuId)
    assert (responseV1.gpu[0].incidentCount == 1), responseV1.gpu[0].incidentCount
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV1.gpu[0].systems[0].system

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.entities[0].entityId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV2.entities[0].entityId, gpuId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV2.entities[0].systems[0].system

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_LWLINK_ERROR_THRESHOLD)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)

def helper_lwlink_check_fatal_errors(handle, gpuIds):
    test_utils.skip_test_if_any_lwlinks_down(handle)
    
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_LWLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                                       1, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1), responseV1.gpuCount
    assert (responseV1.gpu[0].gpuId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV1.gpu[0].gpuId, gpuId)
    assert (responseV1.gpu[0].incidentCount == 1), responseV1.gpu[0].incidentCount
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV1.gpu[0].systems[0].system

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.entities[0].entityId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV2.entities[0].entityId, gpuId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV2.entities[0].systems[0].system

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_LWLINK_ERROR_CRITICAL)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_lwlink_fatal(handle, gpuIds):
    helper_lwlink_check_fatal_errors(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_embedded_lwlink_fatal(handle, gpuIds):
    helper_lwlink_check_fatal_errors(handle, gpuIds)

def helper_lwlink_crc_fatal_threshold(handle, gpuIds): 
    test_utils.skip_test_if_any_lwlinks_down(handle)
    
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_LWLINK
    groupObj.health.Set(newSystems)

    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    # Trigger a failure by having more than 100 CRC errors per second
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId,
                                                       dcgm_fields.DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                                       1000000, -20)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    
    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    
    assert (responseV1.gpuCount == 1), responseV1.gpuCount
    assert (responseV1.gpu[0].gpuId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV1.gpu[0].gpuId, gpuId)
    assert (responseV1.gpu[0].incidentCount == 1), responseV1.gpu[0].incidentCount
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV1.gpu[0].systems[0].system

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.entities[0].entityId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV2.entities[0].entityId, gpuId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV2.entities[0].systems[0].system

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_FAIL)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_lwlink_crc_threshold(handle, gpuIds):
    helper_lwlink_crc_fatal_threshold(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_only_with_live_gpus()
def test_dcgm_embedded_lwlink_crc_threshold(handle, gpuIds):
    helper_lwlink_crc_fatal_threshold(handle, gpuIds)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_dcgm_standalone_health_large_groupid(handle, gpuIds):
    """
    Verifies that a health check can run on a large groupId
    This verifies the fix for bug 1868821
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()

    #Make a bunch of groups and delete them right away so our next groupId is large
    for i in range(100):
        groupObj = systemObj.GetEmptyGroup("test_group_%d" % i)
        groupObj.Delete()

    groupObj = systemObj.GetEmptyGroup("test_good_group")
    groupObj.AddGpu(gpuIds[0])

    assert groupObj.GetId() >= 100, "Expected groupId > 100. got %d" % groupObj.GetId()

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_ALL

    #Any of these will throw an exception on error. Making it past these = success
    groupObj.health.Set(newSystems)
    systemObj.UpdateAllFields(True)
    groupObj.health.Get()
    groupObj.health.Check()


def helper_health_check_lwswitch_errors(handle, switchIds, fieldId, healthSystem, healthResult, errorCode):
    """
    Verifies that a check error oclwrs when an error is injected
    Checks for call errors are done in the bindings except dcgmClientHealthCheck
    """
    #This test will fail if any LwLinks are done
    test_utils.skip_test_if_any_lwlinks_down(handle)

    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    switchId = switchIds[0]
    groupObj.AddEntity(dcgm_fields.DCGM_FE_SWITCH, switchId)
    
    newSystems = healthSystem
    groupObj.health.Set(newSystems)

    field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
    field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
    field.fieldId = fieldId
    field.status = 0
    field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
    field.ts = int((time.time()-5) * 1000000.0) 
    field.value.i64 = 0
 
    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    field.ts = int((time.time()-50) * 1000000.0) 
    field.value.i64 = 0
 
    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    
    # we expect that there will be no data here
    # inject an error into LwSwitch
    field.ts = int((time.time() - 1) * 1000000.0) # set the injected data for a second ago
    field.value.i64 = 5

    ret = dcgm_agent_internal.dcgmInjectEntityFieldValue(handle, dcgm_fields.DCGM_FE_SWITCH, 
                                                         switchId, field)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    #ResponseV1 doesn't know about LwSwitches
    assert (responseV1.gpuCount == 0), responseV1.gpuCount

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.overallHealth == healthResult)
    assert (responseV2.entities[0].entityId == switchId), "Returned switchId %d != sent switchId %d" % (responseV2.entities[0].entityId, switchId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].overallHealth == healthResult), responseV2.entities[0].overallHealth
    assert (responseV2.entities[0].systems[0].system == healthSystem), responseV2.entities[0].systems[0].system

    assert (responseV3.entityCount == 1)
    assert (responseV3.overallHealth == healthResult)
    assert (responseV3.entities[0].entityId == switchId)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].overallHealth == healthResult), responseV3.entities[0].overallHealth
    assert (responseV3.entities[0].systems[0].system == healthSystem)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == errorCode)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwswitch_fatal_errors_standalone(handle, switchIds):
    helper_health_check_lwswitch_errors(handle, switchIds, 
                                        dcgm_fields.DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS, 
                                        dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL, 
                                        dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
                                        dcgm_errors.DCGM_FR_LWSWITCH_FATAL_ERROR)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwswitch_fatal_errors_embedded(handle, switchIds):
    helper_health_check_lwswitch_errors(handle, switchIds, 
                                        dcgm_fields.DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS, 
                                        dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL, 
                                        dcgm_structs.DCGM_HEALTH_RESULT_FAIL,
                                        dcgm_errors.DCGM_FR_LWSWITCH_FATAL_ERROR)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwswitch_nonfatal_errors_standalone(handle, switchIds):
    helper_health_check_lwswitch_errors(handle, switchIds, 
                                        dcgm_fields.DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS, 
                                        dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL, 
                                        dcgm_structs.DCGM_HEALTH_RESULT_WARN,
                                        dcgm_errors.DCGM_FR_LWSWITCH_NON_FATAL_ERROR)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwswitch_nonfatal_errors_embedded(handle, switchIds):
    helper_health_check_lwswitch_errors(handle, switchIds, 
                                        dcgm_fields.DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS, 
                                        dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL, 
                                        dcgm_structs.DCGM_HEALTH_RESULT_WARN,
                                        dcgm_errors.DCGM_FR_LWSWITCH_NON_FATAL_ERROR)

def helper_health_check_lwlink_link_down_gpu(handle, gpuIds):
    """
    Verifies that a check error oclwrs when a LwLink link is set to broken
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]
    
    #Set all links of our injected GPU to Up
    for linkId in range(dcgm_structs.DCGM_LWLINK_MAX_LINKS_PER_GPU):
        dcgm_agent_internal.dcgmSetEntityLwLinkLinkState(handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmLwLinkLinkStateUp)

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_LWLINK
    groupObj.health.Set(newSystems)

    #By default, the health check should pass

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert responseV1.gpuCount == 0, "Expected no errors. Got %d GPUs with errors" % responseV1.gpuCount
    assert responseV2.entityCount == 0, "Expected no errors. Got %d entities with errors" % responseV2.entityCount

    #Set a link to Down
    linkId = 3
    dcgm_agent_internal.dcgmSetEntityLwLinkLinkState(handle, dcgm_fields.DCGM_FE_GPU, gpuId, linkId, dcgm_structs.DcgmLwLinkLinkStateDown)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV1.gpuCount == 1), responseV1.gpuCount
    assert (responseV1.gpu[0].gpuId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV1.gpu[0].gpuId, gpuId)
    assert (responseV1.gpu[0].incidentCount == 1), responseV1.gpu[0].incidentCount
    assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV1.gpu[0].systems[0].system
    assert str(linkId) in responseV1.gpu[0].systems[0].errorString, "Didn't find linkId %d in %s" % (linkId, responseV1.gpu[0].systems[0].errorString)
    logger.info("Health String: " + responseV1.gpu[0].systems[0].errorString)

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.entities[0].entityId == gpuId), "Returned gpuId %d != sent gpuId %d" % (responseV2.entities[0].entityId, gpuId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK), responseV2.entities[0].systems[0].system
    assert str(linkId) in responseV2.entities[0].systems[0].errorString, "Didn't find linkId %d in %s" % (linkId, responseV2.entities[0].systems[0].errorString)

    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWLINK)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_LWLINK_DOWN)
    assert str(linkId) in responseV3.entities[0].systems[0].errors[0].msg, "Didn't find linkId %d in %s" % (linkId, responseV3.entities[0].systems[0].errors[0].msg)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_health_check_lwlink_link_down_gpu_standalone(handle, gpuIds):
    helper_health_check_lwlink_link_down_gpu(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_lwlink_link_down_gpu_embedded(handle, gpuIds):
    helper_health_check_lwlink_link_down_gpu(handle, gpuIds)

def helper_health_check_lwlink_link_down_lwswitch(handle, switchIds):
    """
    Verifies that a check error oclwrs when a LwLink link is set to broken
    """
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    switchId = switchIds[0]
    groupObj.AddEntity(dcgm_fields.DCGM_FE_SWITCH, switchId)
    linkId = 17

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL
    groupObj.health.Set(newSystems)

    #By default, the health check should pass
    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    assert responseV2.entityCount == 0, "Expected no errors. Got %d entities with errors" % responseV2.entityCount
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    assert responseV3.entityCount == 0, "Expected no errors. Got %d entities with errors" % responseV3.entityCount

    #Set a link to Down
    dcgm_agent_internal.dcgmSetEntityLwLinkLinkState(handle, dcgm_fields.DCGM_FE_SWITCH, switchId, linkId, dcgm_structs.DcgmLwLinkLinkStateDown)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)

    assert (responseV2.entityCount == 1), responseV2.entityCount
    assert (responseV2.entities[0].entityId == switchId), "Returned switchId %d != sent switchId %d" % (responseV2.entities[0].entityId, switchId)
    assert (responseV2.entities[0].incidentCount == 1), responseV2.entities[0].incidentCount
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL), responseV2.entities[0].systems[0].system
    assert str(linkId) in responseV2.entities[0].systems[0].errorString, "Didn't find linkId %d in %s" % (linkId, responseV2.entities[0].systems[0].errorString)
    
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityId == switchId)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_SWITCH)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_LWSWITCH_FATAL)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_LWLINK_DOWN)
    assert str(linkId) in responseV3.entities[0].systems[0].errors[0].msg, "Didn't find linkId %d in %s" % (linkId, responseV3.entities[0].systems[0].errors[0].msg)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwlink_link_down_lwswitch_standalone(handle, switchIds):
    helper_health_check_lwlink_link_down_lwswitch(handle, switchIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_lwswitches()
def test_health_check_lwlink_link_down_lwswitch_embedded(handle, switchIds):
    helper_health_check_lwlink_link_down_lwswitch(handle, switchIds)

def helper_health_check_multiple_failures(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    # We are going to trigger two failures at the same time
    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_PCIE | dcgm_structs.DCGM_HEALTH_WATCH_MEM
    groupObj.health.Set(newSystems)
    
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
            0, -50)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            0, -45)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    response = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    assert (response.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_PASS), "System should be healthy to start"

    # inject a PCI error and a memory error, and make sure we report both
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
            4, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)
    ret = dcgm_internal_helpers.inject_field_value_i64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
            100, 100)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV1 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version1)
    assert (responseV1.gpuCount == 1)
    assert (responseV1.gpu[0].gpuId == gpuId)
    assert (responseV1.gpu[0].incidentCount == 2)
    if responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE:
        assert (responseV1.gpu[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    else:
        assert (responseV1.gpu[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
        assert (responseV1.gpu[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 2)
    if responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE:
        assert (responseV2.entities[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
    else:
        assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
        assert (responseV2.entities[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 2)
    if responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE:
        assert (responseV3.entities[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
        
        # The replay error is in position 0 here
        assert (responseV3.entities[0].systems[0].errorCount == 1)
        assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)
        # The memory error is in position 1 here
        assert (responseV3.entities[0].systems[1].errorCount == 1)
        assert (responseV3.entities[0].systems[1].errors[0].code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
    else:
        assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_MEM)
        assert (responseV3.entities[0].systems[1].system == dcgm_structs.DCGM_HEALTH_WATCH_PCIE)

        # The memory error is in position 0 here
        assert (responseV3.entities[0].systems[0].errorCount == 1)
        assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_VOLATILE_DBE_DETECTED)
        # The replay error is in position 1 here
        assert (responseV3.entities[0].systems[1].errorCount == 1)
        assert (responseV3.entities[0].systems[1].errors[0].code == dcgm_errors.DCGM_FR_PCI_REPLAY_RATE)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_health_check_standalone_multiple_failures(handle, gpuIds):
    helper_health_check_multiple_failures(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_embedded_multiple_failures(handle, gpuIds):
    helper_health_check_multiple_failures(handle, gpuIds)

def helper_health_check_unreadable_power_usage(handle, gpuIds):
    handleObj = pydcgm.DcgmHandle(handle=handle)
    systemObj = handleObj.GetSystem()
    groupObj = systemObj.GetEmptyGroup("test1")
    groupObj.AddGpu(gpuIds[0])
    gpuIds = groupObj.GetGpuIds() #Limit gpuIds to GPUs in our group
    gpuId = gpuIds[0]

    newSystems = dcgm_structs.DCGM_HEALTH_WATCH_POWER
    groupObj.health.Set(newSystems)
    
    ret = dcgm_internal_helpers.inject_field_value_fp64(handle, gpuId, dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
            dcgmvalue.DCGM_FP64_BLANK, 50)
    assert (ret == dcgm_structs.DCGM_ST_OK)

    responseV2 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version2)
    assert (responseV2.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV2.entityCount == 1)
    assert (responseV2.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV2.entities[0].entityId == gpuId)
    assert (responseV2.entities[0].incidentCount == 1)
    assert (responseV2.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV2.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)

    responseV3 = groupObj.health.Check(dcgm_structs.dcgmHealthResponse_version3)
    assert (responseV3.overallHealth == dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV3.entityCount == 1)
    assert (responseV3.entities[0].entityGroupId == dcgm_fields.DCGM_FE_GPU)
    assert (responseV3.entities[0].entityId == gpuId)
    assert (responseV3.entities[0].incidentCount == 1)
    assert (responseV3.entities[0].systems[0].system == dcgm_structs.DCGM_HEALTH_WATCH_POWER)
    assert (responseV3.entities[0].systems[0].health == dcgm_structs.DCGM_HEALTH_RESULT_WARN)
    assert (responseV3.entities[0].systems[0].errorCount == 1)
    assert (responseV3.entities[0].systems[0].errors[0].code == dcgm_errors.DCGM_FR_POWER_UNREADABLE)

@test_utils.run_with_standalone_host_engine(120)
@test_utils.run_with_initialized_client()
@test_utils.run_with_injection_gpus()
def test_health_check_standalone_unreadable_power_usage(handle, gpuIds):
    helper_health_check_unreadable_power_usage(handle, gpuIds)

@test_utils.run_with_embedded_host_engine()
@test_utils.run_with_injection_gpus()
def test_health_check_embedded_unreadable_power_usage(handle, gpuIds):
    helper_health_check_unreadable_power_usage(handle, gpuIds)
