-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCISYNC_BACKEND_TEGRA
-- Unit(s) Under Test: lwscisync_backend_tegra
-- 
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscisync_backend_tegra

-- Subprogram: LwSciSyncCoreRmAlloc

-- Test Case: TC_001.LwSciSyncCoreRmAlloc.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmAlloc.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmAlloc.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmAlloc() Normal operation.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmBackEnd.
 * LwRmHost1xGetDefaultOpenAttrs() returns the default initialized attributes to open a new Host1x handle.
 * LwRmHost1xOpen() returns LwSuccess and valid host1x handle when device is successfully opened.}
 *
 * @testinput{backEnd is used as backend RM handle.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate backend RM handle.
 * LwRmHost1xGetDefaultOpenAttrs() is called to initialize attributes to open a new Host1x handle
 * LwRmHost1xOpen() is called to open the Host1x handle.
 * Returns LwSciError_Success when backend RM handle is allocated successfully.}
 *
 * @testcase{18853749}
 *
 * @verify{18844830}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs.return.war_t186_b_instance:true
TEST.VALUE:uut_prototype_stubs.LwRmHost1xOpen.return:LwError_Success
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xOpen.attrs.war_t186_b_instance:true
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs
  uut_prototype_stubs.LwRmHost1xOpen
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp.host1xp[0]
<<uut_prototype_stubs.LwRmHost1xOpen.host1xp>>[0] = ( 345 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp
{{ <<uut_prototype_stubs.LwRmHost1xOpen.host1xp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd.ParamLwSciSyncCoreRmBackEnd[0].host1x
<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>>[0].host1x = ( 10 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd
{{ <<lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd>>[0]->host1x == ( 345 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmAlloc.DeviceFailedToOpenHost1xHandle
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmAlloc.DeviceFailedToOpenHost1xHandle
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmAlloc.DeviceFailedToOpenHost1xHandle}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmAlloc() device failed to open Host1x handle.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmBackEnd.
 * LwRmHost1xGetDefaultOpenAttrs() returns the default initialized attributes to open a new Host1x handle.
 * LwRmHost1xOpen() returns not supported when device is failed to open Host1x handle.}
 *
 * @testinput{backEnd is used as backend RM handle.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate backend RM handle.
 * LwRmHost1xGetDefaultOpenAttrs() is called to initialize attributes to open a new Host1x handle
 * LwRmHost1xOpen() is called to open the Host1x handle.
 * LwSciCommonFree() is called to de-allocate backend RM handle.
 * Returns LwSciError_ResourceError when device is failed to open Host1x handle.}
 *
 * @testcase{18853752}
 *
 * @verify{18844830}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs.return.war_t186_b_instance:false
TEST.VALUE:uut_prototype_stubs.LwRmHost1xOpen.return:LwError_NotSupported
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xOpen.attrs.war_t186_b_instance:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs
  uut_prototype_stubs.LwRmHost1xOpen
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp.host1xp[0]
<<uut_prototype_stubs.LwRmHost1xOpen.host1xp>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp
{{ <<uut_prototype_stubs.LwRmHost1xOpen.host1xp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ (<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>> )) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmAlloc.EnoughMemoryUnavailableToAllocate
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmAlloc.EnoughMemoryUnavailableToAllocate
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmAlloc.EnoughMemoryUnavailableToAllocate}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmAlloc() enough memory is not available to allocate.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmBackEnd.
 * LwRmHost1xGetDefaultOpenAttrs() returns the default initialized attributes to open a new Host1x handle.
 * LwRmHost1xOpen() returns insufficient memory when enough memory is not available to allocate.}
 *
 * @testinput{backEnd is used as backend RM handle.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate backend RM handle.
 * LwRmHost1xGetDefaultOpenAttrs() is called to initialize attributes to open a new Host1x handle
 * LwRmHost1xOpen() is called to open the Host1x handle.
 * LwSciCommonFree() is called to de-allocate backend RM handle.
 * Returns LwSciError_ResourceError when enough memory is not available to allocate.}
 *
 * @testcase{18853755}
 *
 * @verify{18844830}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs.return.war_t186_b_instance:true
TEST.VALUE:uut_prototype_stubs.LwRmHost1xOpen.return:LwError_InsufficientMemory
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xOpen.attrs.war_t186_b_instance:true
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xGetDefaultOpenAttrs
  uut_prototype_stubs.LwRmHost1xOpen
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp.host1xp[0]
<<uut_prototype_stubs.LwRmHost1xOpen.host1xp>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xOpen.host1xp
{{ <<uut_prototype_stubs.LwRmHost1xOpen.host1xp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ (<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd>> ))}}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreRmAlloc.BackendRMHandleNotAllocated
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreRmAlloc.BackendRMHandleNotAllocated
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreRmAlloc.BackendRMHandleNotAllocated}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmAlloc() backend RM handle not allocated}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL (backend RM handle not allocated).}
 *
 * @testinput{backEnd is used as backend RM handle.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate backend RM handle.
 * Returns LwSciError_InsufficientMemory when backend RM handle not allocated}
 *
 * @testcase{18853758}
 *
 * @verify{18844830}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd
{{ *<<lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreRmAlloc.BackEndIsNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreRmAlloc.BackEndIsNULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreRmAlloc.BackEndIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmAlloc() backEnd is NULL}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{backEnd is NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmAlloc() Panics.}
 *
 * @testcase{18853761}
 *
 * @verify{18844830}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmAlloc.backEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreRmFree

-- Test Case: TC_001.LwSciSyncCoreRmFree.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmFree.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmFree.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmFree() Normal operation}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{host1x handle is not NULL.}
 *
 * @testinput{backEnd is pointer to core RM backend structure.}
 *
 * @testbehavior{LwRmHost1xClose() is called to close the Host1x handle.
 * LwSciCommonFree() is called to de-allocate backend RM handle.}
 *
 * @testcase{18853764}
 *
 * @verify{18844833}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
  uut_prototype_stubs.LwRmHost1xClose
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xClose.host1x
{{ <<uut_prototype_stubs.LwRmHost1xClose.host1x>> == ( 10 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd>>) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd.backEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd>>[0].host1x = ( 10 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmFree.Host1xHandleIsNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmFree.Host1xHandleIsNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmFree.Host1xHandleIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmFree() host1x handle is NULL}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{host1x handle is driven as NULL.}
 *
 * @testinput{backEnd is pointer to core RM backend structure.}
 *
 * @testbehavior{LwSciCommonFree() is called to de-allocate backend RM handle.}
 *
 * @testcase{18853767}
 *
 * @verify{18844833}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd>>) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd.backEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd>>[0].host1x = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmFree.BackEndNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmFree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmFree.BackEndNULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmFree.BackEndNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmFree() backEnd is NULL}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None.}
 *
 * @testinput{backEnd is NULL.}
 *
 * @testbehavior{Nothing to do when backend RM handle is already de-allocate.}
 *
 * @testcase{18853770}
 *
 * @verify{18844833}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmFree.backEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmFree
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreRmGetHost1xHandle

-- Test Case: TC_001.LwSciSyncCoreRmGetHost1xHandle.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmGetHost1xHandle
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmGetHost1xHandle.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmGetHost1xHandle.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmGetHost1xHandle() Normal operation}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{host1x handle is not NULL.}
 *
 * @testinput{backEnd is pointer to core RM backend structure.}
 *
 * @testbehavior{Returns host1x handle in the backend}
 *
 * @testcase{18853773}
 *
 * @verify{18844845}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmGetHost1xHandle
  lwscisync_backend_tegra.c.LwSciSyncCoreRmGetHost1xHandle
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd.backEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd>>[0].host1x = ( 10 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.return
{{ <<lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.return>> == ( 10 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmGetHost1xHandle.Host1xHandleNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmGetHost1xHandle
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmGetHost1xHandle.Host1xHandleNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmGetHost1xHandle.Host1xHandleNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmGetHost1xHandle() host1x handle is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{host1x handle is NULL.}
 *
 * @testinput{backEnd is pointer to core RM backend structure.}
 *
 * @testbehavior{Returns host1x handle in the backend}
 *
 * @testcase{18853776}
 *
 * @verify{18844845}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmGetHost1xHandle
  lwscisync_backend_tegra.c.LwSciSyncCoreRmGetHost1xHandle
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd.backEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd>>[0].host1x = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.return
{{ <<lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.return>> == ( NULL) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmGetHost1xHandle.PanicsBackEndNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmGetHost1xHandle
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmGetHost1xHandle.PanicsBackEndNULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmGetHost1xHandle.PanicsBackEndNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmGetHost1xHandle() backEnd is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{backEnd is NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmGetHost1xHandle() Panics.}
 *
 * @testcase{18853779}
 *
 * @verify{18844845}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmGetHost1xHandle.backEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreRmWaitCtxBackEndAlloc

-- Test Case: TC_001.LwSciSyncCoreRmWaitCtxBackEndAlloc.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmWaitCtxBackEndAlloc.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmWaitCtxBackEndAlloc.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() Normal operation.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{Host1x Handle is not NULL.
 * LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmWaitContextBackEnd.
 * LwRmHost1xWaiterAllocate() returns LwSuccess, when waiter object is allocated sucessfully.}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is not NULL.
 * waitContextBackEnd is not NULL.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate LwSciSyncCoreRmWaitContextBackEnd.
 * Retrieve LwRmHost1xHandle held in LwSciSyncCoreRmBackEnd.
 * LwRmHost1xWaiterAllocate() is called to allocate waiter object.
 * Returns LwSciError_Success when LwSciSyncCoreRmWaitContextBackEnd is allocated successfully.}
 *
 * @testcase{18853782}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.return:LwError_Success
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xWaiterAllocate
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp.waiterp[0]
<<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>>[0] = ( 54321 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x>> == ( 123 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmWaitContextBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd.rmBackEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>>[0].host1x = ( 123 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd
{{<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd>>[0]->waiterHandle == (54321)}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmWaitCtxBackEndAlloc.Host1xHandleNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmWaitCtxBackEndAlloc.Host1xHandleNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmWaitCtxBackEndAlloc.Host1xHandleNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() Host1x Handle is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{Host1x Handle is NULL.
 * LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmWaitContextBackEnd.
 * LwRmHost1xWaiterAllocate() returns BadParameter, when Host1x Handle is NULL.}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is not NULL.
 * waitContextBackEnd is not NULL.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate LwSciSyncCoreRmWaitContextBackEnd.
 * Retrieve LwRmHost1xHandle held in LwSciSyncCoreRmBackEnd.
 * LwRmHost1xWaiterAllocate() is called to allocate waiter object.
 * LwRmHost1xWaiterFree() is called to free an allocated waiter.
 * LwSciCommonFree() is called to deallocate LwSciSyncCoreRmWaitContextBackEnd
 * Returns LwSciError_ResourceError when Host1x Handle is NULL.}
 *
 * @testcase{18853785}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.return:LwError_BadParameter
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xWaiterAllocate
  uut_prototype_stubs.LwRmHost1xWaiterFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp.waiterp[0]
<<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>>[0] = ( 54321 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterFree.waiter
{{ <<uut_prototype_stubs.LwRmHost1xWaiterFree.waiter>> == ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>>->waiterHandle ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd.rmBackEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>>[0].host1x = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmWaitCtxBackEndAlloc.FailedToOpenHost1xHandle
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmWaitCtxBackEndAlloc.FailedToOpenHost1xHandle
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmWaitCtxBackEndAlloc.FailedToOpenHost1xHandle}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() failed to open Host1x handle.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{Host1x Handle is NULL.
 * LwSciCommonCalloc() returns pointer to the allocated LwSciSyncCoreRmWaitContextBackEnd.
 * LwRmHost1xWaiterAllocate() returns NotSupported when failed to open Host1x handle.}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is not NULL.
 * waitContextBackEnd is not NULL.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate LwSciSyncCoreRmWaitContextBackEnd.
 * Retrieve LwRmHost1xHandle held in LwSciSyncCoreRmBackEnd.
 * LwRmHost1xWaiterAllocate() is called to allocate waiter object.
 * LwSciCommonFree() is called to deallocate LwSciSyncCoreRmWaitContextBackEnd
 * Returns LwSciError_ResourceError when failed to open Host1x handle.}
 *
 * @testcase{18853788}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.return:LwError_NotSupported
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwRmHost1xWaiterAllocate
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp.waiterp[0]
<<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.waiterp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xWaiterAllocate.host1x>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmWaitContextBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_backend_tegra.<<GLOBAL>>.ParamLwSciSyncCoreRmWaitContextBackEnd>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd.rmBackEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>>[0].host1x = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEndNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEndNULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEndNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() rmBackEnd is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is NULL.
 * waitContextBackEnd is not NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() Panics.}
 *
 * @testcase{18853791}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<null>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaitContextBackEndIsNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaitContextBackEndIsNULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaitContextBackEndIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() waitContextBackEnd is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is not NULL.
 * waitContextBackEnd is NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() Panics.}
 *
 * @testcase{18853794}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd.rmBackEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>>[0].host1x = ( 123 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaiterObjectNotAllocated
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaiterObjectNotAllocated
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreRmWaitCtxBackEndAlloc.WaiterObjectNotAllocated}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndAlloc() waiter object is not allocated.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{Host1x Handle is not NULL.
 * LwSciCommonCalloc() waitContextBackEnd is not allocated.}
 *
 * @testinput{rmBackEnd (backend Rm handle) for allocation is not NULL.
 * waitContextBackEnd is not NULL}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate LwSciSyncCoreRmWaitContextBackEnd.
 * Returns LwSciError_InsufficientMemory when LwSciSyncCoreRmWaitContextBackEnd is not allocated.}
 *
 * @testcase{18853797}
 *
 * @verify{18844836}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd:<<malloc 1>>
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<malloc 1>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd:<<null>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCoreRmWaitContextBackEndRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd.rmBackEnd[0].host1x
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>>[0].host1x = ( 123 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreRmWaitCtxBackEndFree

-- Test Case: TC_001.LwSciSyncCoreRmWaitCtxBackEndFree.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmWaitCtxBackEndFree.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmWaitCtxBackEndFree.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndFree() normal operation.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{waiterHandle of the given waitContextBackEnd is not NULL}
 *
 * @testinput{waitContextBackEnd is not NULL}
 *
 * @testbehavior{LwRmHost1xWaiterFree() is called to free an allocated waiter.
 * LwSciCommonFree() is called to deallocate LwSciSyncCoreRmWaitContextBackEnd}
 *
 * @testcase{18853800}
 *
 * @verify{18844839}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
  uut_prototype_stubs.LwRmHost1xWaiterFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xWaiterFree.waiter
{{ <<uut_prototype_stubs.LwRmHost1xWaiterFree.waiter>> == ( 1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd.waitContextBackEnd[0].waiterHandle
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>>[0].waiterHandle = ( 1234 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmWaitCtxBackEndFree.waiterHandleIsNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmWaitCtxBackEndFree.waiterHandleIsNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmWaitCtxBackEndFree.waiterHandleIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndFree() waiterHandle is NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{waiterHandle of the given waitContextBackEnd is NULL}
 *
 * @testinput{waitContextBackEnd is not NULL}
 *
 * @testbehavior{LwSciCommonFree() is called to deallocate LwSciSyncCoreRmWaitContextBackEnd}
 *
 * @testcase{18853803}
 *
 * @verify{18844839}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd.waitContextBackEnd[0].waiterHandle
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>>[0].waiterHandle = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmWaitCtxBackEndFree.WaitContextBackendNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndFree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmWaitCtxBackEndFree.WaitContextBackendNULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmWaitCtxBackEndFree.WaitContextBackendNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndFree() waitContextBackEnd is NULL.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{waitContextBackEnd is NULL}
 *
 * @testbehavior{Nothing to do when LwSciSyncCoreRmWaitContextBackEnd is already deallocated.}
 *
 * @testcase{18853806}
 *
 * @verify{18844839}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndFree
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreRmWaitCtxBackEndValidate

-- Test Case: TC_001.LwSciSyncCoreRmWaitCtxBackEndValidate.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndValidate
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmWaitCtxBackEndValidate.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmWaitCtxBackEndValidate.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndValidate() Normal operation.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{waiterHandle of the given waitContextBackEnd is not NULL}
 *
 * @testinput{waitContextBackEnd is not null}
 *
 * @testbehavior{returns LwSciError_Success when waiterHandle of the given waitContextBackEnd is not NULL}
 *
 * @testcase{18853809}
 *
 * @verify{18844842}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd:<<malloc 1>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndValidate
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndValidate
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd.waitContextBackEnd[0].waiterHandle
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd>>[0].waiterHandle = ( 123 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmWaitCtxBackEndValidate.waiterHandleIsNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndValidate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmWaitCtxBackEndValidate.waiterHandleIsNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmWaitCtxBackEndValidate.waiterHandleIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndValidate() waiterHandle is NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{waiterHandle of the given waitContextBackEnd is NULL}
 *
 * @testinput{waitContextBackEnd is not NULL.}
 *
 * @testbehavior{returns LwSciError_BadParameter when waiterHandle of the given waitContextBackEnd is NULL}
 *
 * @testcase{18853812}
 *
 * @verify{18844842}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd:<<malloc 1>>
TEST.EXPECTED:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndValidate
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndValidate
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd.waitContextBackEnd[0].waiterHandle
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd>>[0].waiterHandle = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreRmWaitCtxBackEndValidate.WaitContextBackendNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxBackEndValidate
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreRmWaitCtxBackEndValidate.WaitContextBackendNULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreRmWaitCtxBackEndValidate.WaitContextBackendNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxBackEndValidate() waitContextBackEnd is NULL.}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{waitContextBackEnd is NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmWaitCtxBackEndValidate() Panics.}
 *
 * @testcase{18853815}
 *
 * @verify{18844842}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxBackEndValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreRmWaitCtxGetWaiterHandle

-- Test Case: TC_001.LwSciSyncCoreRmWaitCtxGetWaiterHandle.NormalOperation
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxGetWaiterHandle
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreRmWaitCtxGetWaiterHandle.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreRmWaitCtxGetWaiterHandle.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxGetWaiterHandle() Normal operation.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{waitContextBackEnd is not null}
 *
 * @testbehavior{returns LwRmHost1xWaiterHandle from the backend.}
 *
 * @testcase{18853818}
 *
 * @verify{18844848}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxGetWaiterHandle
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd.waitContextBackEnd[0].waiterHandle
<<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>>[0].waiterHandle = ( 123 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
{{ <<lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> == ( 123 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreRmWaitCtxGetWaiterHandle.WaitContextBackendNULL
TEST.UNIT:lwscisync_backend_tegra
TEST.SUBPROGRAM:LwSciSyncCoreRmWaitCtxGetWaiterHandle
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreRmWaitCtxGetWaiterHandle.WaitContextBackendNULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreRmWaitCtxGetWaiterHandle.WaitContextBackendNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreRmWaitCtxGetWaiterHandle()  waitContextBackEnd is null}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{waitContextBackEnd is null}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreRmWaitCtxGetWaiterHandle() Panics.}
 *
 * @testcase{18853821}
 *
 * @verify{18844848}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_backend_tegra.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd:<<null>>
TEST.FLOW
  lwscisync_backend_tegra.c.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END
