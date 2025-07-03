
import pydcgm
import dcgm_structs
import test_utils
import dcgm_agent

@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_get_statuses(handle):
    '''
    Do a basic sanity check of the DCGM module statuses returned
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    ms = dcgmSystem.modules.GetStatuses()

    assert ms.numStatuses == dcgm_structs.DcgmModuleIdCount, "%d != %d" % (ms.numStatuses, dcgm_structs.DcgmModuleIdCount)
    assert ms.statuses[0].id == dcgm_structs.DcgmModuleIdCore, "%d != %d" % (ms.statuses[0].id, dcgm_structs.DcgmModuleIdCore)
    assert ms.statuses[0].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[0].status, dcgm_structs.DcgmModuleStatusLoaded)

    for i in range(1, ms.numStatuses):
        #.id == index
        assert ms.statuses[i].id == i, "%d != %d" % (ms.statuses[i].id, i)
        #Assert all non-core modules aren't loaded
        assert ms.statuses[i].status == dcgm_structs.DcgmModuleStatusNotLoaded, "%d != %d" % (ms.statuses[i].status, dcgm_structs.DcgmModuleStatusNotLoaded)

@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_in_use_introspection(handle):
    '''
    Make sure that the introspection module cannot be blacklisted after it's loaded
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect

    #Lazy load the introspection module
    dcgmSystem.introspect.state.toggle(dcgm_structs.DCGM_INTROSPECT_STATE.ENABLED)

    #Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    assert ms.statuses[moduleId].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[moduleId].status, dcgm_structs.DcgmModuleStatusLoaded)
    
    #Make sure we can't blacklist the module after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Blacklist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_blacklist_introspection(handle):
    '''
    Make sure that the introspection module can be blacklisted
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    moduleId = dcgm_structs.DcgmModuleIdIntrospect
    
    dcgmSystem.modules.Blacklist(moduleId)

    #Try to lazy load the blacklisted introspection module
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        dcgmSystem.introspect.state.toggle(dcgm_structs.DCGM_INTROSPECT_STATE.ENABLED)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_in_use_health(handle):
    '''
    Make sure that the health module cannot be blacklisted after it's loaded
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth

    #Lazy load the health module
    dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)

    #Make sure the module was loaded
    ms = dcgmSystem.modules.GetStatuses()
    assert ms.statuses[moduleId].status == dcgm_structs.DcgmModuleStatusLoaded, "%d != %d" % (ms.statuses[moduleId].status, dcgm_structs.DcgmModuleStatusLoaded)
    
    #Make sure we can't blacklist the module after it's loaded
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_IN_USE)):
        dcgmSystem.modules.Blacklist(moduleId)


@test_utils.run_with_embedded_host_engine()
def test_dcgm_modules_blacklist_health(handle):
    '''
    Make sure that the health module can be blacklisted
    '''
    dcgmHandle = pydcgm.DcgmHandle(handle=handle)
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmGroup = dcgmSystem.GetDefaultGroup()
    moduleId = dcgm_structs.DcgmModuleIdHealth
    
    dcgmSystem.modules.Blacklist(moduleId)

    #Try to lazy load the blacklisted introspection module
    with test_utils.assert_raises(dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_MODULE_NOT_LOADED)):
        dcgmGroup.health.Set(dcgm_structs.DCGM_HEALTH_WATCH_ALL)

