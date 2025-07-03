from DcgmReader import *
import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
from dcgm_structs import dcgmExceptionClass
import test_utils
import time

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_reader_default(handle):
    dr = DcgmReader()
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldNameDict()

    for gpuId in latest:
        # latest data might be less than the list, because blank values aren't included
        assert len(latest[gpuId]) <= len(defaultFieldIds)

        # Make sure we get strings
        for key in latest[gpuId]:
            assert isinstance(key, basestring)

    sample = dr.GetLatestGpuValuesAsFieldIdDict()

    for gpuId in sample:
        assert len(sample[gpuId]) <= len(defaultFieldIds)
        
        # Make sure we get valid integer field ids
        for fieldId in sample[gpuId]:
            assert isinstance(fieldId, int)
            assert dcgm_fields.DcgmFieldGetById(fieldId) != None

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
def test_dcgm_reader_specific_fields(handle):
    specificFields = [dcgm_fields.DCGM_FI_DEV_POWER_USAGE, dcgm_fields.DCGM_FI_DEV_XID_ERRORS]
    dr = DcgmReader(fieldIds=specificFields)
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldNameDict()

    for gpuId in latest:
        assert len(latest[gpuId]) <= len(specificFields)

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_reading_specific_data(handle, gpuIds):
    """ 
    Verifies that we can inject specific data and get that same data back
    """

    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()

    specificFieldIds = [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION]
    fieldValues = [1,
                   1000,
                   9000]
    
    for i in range(0, len(specificFieldIds)):
        field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
        field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
        field.fieldId = specificFieldIds[i]
        field.status = 0
        field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
        field.ts = int((time.time()+10) * 1000000.0) # set the injected data into the future
        field.value.i64 = fieldValues[i]
        ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuIds[0], field)
        assert (ret == dcgm_structs.DCGM_ST_OK)
    
    dr = DcgmReader(fieldIds=specificFieldIds)
    dr.SetHandle(handle)
    latest = dr.GetLatestGpuValuesAsFieldIdDict()

    assert len(latest[gpuIds[0]]) == len(specificFieldIds)

    for i in range(0, len(specificFieldIds)):
        assert latest[gpuIds[0]][specificFieldIds[i]] == fieldValues[i]
        

