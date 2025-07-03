import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
from dcgm_structs import dcgmExceptionClass
import test_utils
import time
import os
import sys

# Set up the environment for the DcgmCollectd class before importing
os.elwiron['DCGM_TESTING_FRAMEWORK'] = 'True'
if 'LD_LIBRARY_PATH' in os.elwiron:
    os.elwiron['DCGMLIBPATH'] = os.elwiron['LD_LIBRARY_PATH']

stubspath  = os.path.dirname(os.path.realpath(__file__)) + '/stubs/'
if stubspath not in sys.path:
     sys.path.insert(0, stubspath)

import collectd_tester_globals
import dcgm_collectd_plugin

@test_utils.run_with_standalone_host_engine(20)
@test_utils.run_with_initialized_client()
@test_utils.run_only_with_live_gpus()
def test_collectd_integration(handle, gpuIds):
    """ 
    Verifies that we can inject specific data and get that same data back
    """
    dcgmHandle = pydcgm.DcgmHandle(handle)
    dcgmSystem = dcgmHandle.GetSystem()

    specificFieldIds = [dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,
                dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,
                dcgm_fields.DCGM_FI_DEV_POWER_VIOLATION,
                dcgm_fields.DCGM_FI_DEV_THERMAL_VIOLATION]
    fieldValues = [1,
                   5,
                   1000,
                   9000]
    
    for gpuId in gpuIds:    
        for i in range(0, len(specificFieldIds)):
            field = dcgm_structs_internal.c_dcgmInjectFieldValue_v1()
            field.version = dcgm_structs_internal.dcgmInjectFieldValue_version1
            field.fieldId = specificFieldIds[i]
            field.status = 0
            field.fieldType = ord(dcgm_fields.DCGM_FT_INT64)
            field.ts = int((time.time()+10) * 1000000.0) # set the injected data into the future
            field.value.i64 = fieldValues[i]
            ret = dcgm_agent_internal.dcgmInjectFieldValue(handle, gpuId, field)
            assert (ret == dcgm_structs.DCGM_ST_OK)

    gvars = collectd_tester_globals.gvars

    assert 'init' in gvars
    gvars['init']()

    assert 'read' in gvars
    gvars['read']()

    assert 'out' in gvars
    outDict = gvars['out']

    assert 'shutdown' in gvars
#    gvars['shutdown']()

    for gpuId in gpuIds:
        assert str(gpuId) in outDict

        gpuDict = outDict[str(gpuId)]

        for i in range(0, len(specificFieldIds)):
            fieldTag = dcgmSystem.fields.GetFieldById(specificFieldIds[i]).tag

            assert fieldTag in gpuDict
            assert gpuDict[fieldTag] == fieldValues[i]

