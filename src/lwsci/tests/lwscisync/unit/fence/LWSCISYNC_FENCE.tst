-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
--
-- Environment    : LWSCISYNC_FENCE
-- Unit(s) Under Test: lwscisync_fence
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscisync_fence

-- Subprogram: LwSciSyncFenceClear

-- Test Case: TC_001.LwSciSyncFenceClear.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceClear
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceClear.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceClear.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceClear() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to '0x0' (coreFence->syncObj).
 * - 'syncFence->payload[1..5]' is set to '0x0'. Fence is cleared.}
 *
 * @testcase{18851691}
 *
 * @verify{18844485}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0..5]:0x0
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceClear
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceClear
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceClear.syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceClear
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceClear.syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceClear.syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceClear() when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x0' or NULL.
 * - 'syncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' contains '0x0' (coreFence->syncObj).
 * - 'syncFence->payload[1..5]' still contains '0x5555555555555555'. Fence is not cleared.}
 *
 * @testcase{18851694}
 *
 * @verify{18844485}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[1..5]:0x5555555555555555
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceClear
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceClear
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceClear.syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceClear
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceClear.syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceClear.syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceClear() when LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid LwSciSyncObj).
 * - 'syncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceClear() panics.}
 *
 * @testcase{18851697}
 *
 * @verify{18844485}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[1..5]:0x5555555555555555
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceClear
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceClear.syncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceClear
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceClear.syncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceClear.syncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceClear() when the memory pointed to by 'syncFence' pointer is already cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x0'.
 * - 'syncFence->payload[1..5]' is set to '0x0'.}
 *
 * @testbehavior{- 'syncFence->payload[0]' contains '0x0' (coreFence->syncObj).
 * - 'syncFence->payload[1..5]' still contains '0x0'. Fence is already cleared.}
 *
 * @testcase{18851700}
 *
 * @verify{18844485}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceClear.syncFence[0].payload[0..5]:0x0
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceClear
  lwscisync_fence.c.LwSciSyncFenceClear
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncFenceClear.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceClear
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceClear.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceClear.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceClear() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to NULL.}
 *
 * @testbehavior{- 'syncFence' pointer still points to NULL. No operation.}
 *
 * @testcase{18851703}
 *
 * @verify{18844485}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceClear.syncFence:<<null>>
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceClear
  lwscisync_fence.c.LwSciSyncFenceClear
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncFenceDup

-- Test Case: TC_001.LwSciSyncFenceDup.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceDup.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceDup.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjFreeObjAndRef() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - 'dstSyncFence->payload[0]' is set to '0x9999999999999999' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' is set to '0x5555555555555555'. Destination fence is duplicated.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851706}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceDup.srcSyncFence_syncObj_refCount_is_max
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceDup.srcSyncFence_syncObj_refCount_is_max
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceDup.srcSyncFence_syncObj_refCount_is_max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'srcSyncFence' LwSciSyncObj reference count is maximum.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjFreeObjAndRef() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_IlwalidState'.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'dstSyncFence->payload[0]' is set to '0x0' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' is set to '0x0'. Destination fence is cleared, but not duplicated.
 * - returns 'LwSciError_IlwalidState'.}
 *
 * @testcase{18851709}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_IlwalidState
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceDup.srcSyncFence_syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceDup.srcSyncFence_syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceDup.srcSyncFence_syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'srcSyncFence' when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjFreeObjAndRef() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x0' or NULL.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'dstSyncFence->payload[0]' is set to '0x0' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' is set to '0x0'. Destination fence is cleared, but not duplicated.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851712}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success,LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceDup.srcSyncFence_syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceDup.srcSyncFence_syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceDup.srcSyncFence_syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'srcSyncFence' LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x0'.
 * - 'dstSyncFence->payload[1..5]' is set to '0x0'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjFreeObjAndRef() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid src LwSciSyncObj).
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceDup() panics.}
 *
 * @testcase{18851715}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncFenceDup.srcSyncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceDup.srcSyncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceDup.srcSyncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when the memory pointed to by 'srcSyncFence' pointer is already cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjFreeObjAndRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x0'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x0'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - 'dstSyncFence->payload[0]' is set to '0x0' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' is set to '0x0'. Destination fence is cleared and empty source fence is duplicated.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851718}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0..5]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncFenceDup.dstyncFence_syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_006.LwSciSyncFenceDup.dstyncFence_syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncFenceDup.dstyncFence_syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'dstSyncFence' LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid dst LwSciSyncObj).
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncFenceDup() panics.}
 *
 * @testcase{18851721}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncFenceDup.srcSyncFence_and_dstSyncFence_refs_are_same
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_007.LwSciSyncFenceDup.srcSyncFence_and_dstSyncFence_refs_are_same
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncFenceDup.srcSyncFence_and_dstSyncFence_refs_are_same}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'srcSyncFence' 'dstSyncFence' pointer references point to the same LwSciSyncFence object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to the same memory as 'srcSyncFence'.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- 'dstSyncFence->payload[0]' contains '0x9999999999999999' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' contains '0x5555555555555555'. No operation on destination fence. Neither cleared nor duplicated.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851724}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence
<<lwscisync_fence.LwSciSyncFenceDup.dstSyncFence>> = ( <<lwscisync_fence.LwSciSyncFenceDup.srcSyncFence>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncFenceDup.dstSyncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_008.LwSciSyncFenceDup.dstSyncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncFenceDup.dstSyncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'dstSyncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'dstSyncFence' pointer points to NULL.}
 *
 * @testinput{- 'srcSyncFence' pointer points to a valid memory.
 * - 'srcSyncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'srcSyncFence->payload[1..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- 'dstSyncFence' pointer still points to NULL. No operation on destination fence.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851727}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence[0].payload[1..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncFenceDup.srcSyncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceDup
TEST.NEW
TEST.NAME:TC_009.LwSciSyncFenceDup.srcSyncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncFenceDup.srcSyncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceDup() when 'srcSyncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'dstSyncFence' pointer points to a valid memory.
 * - 'dstSyncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'dstSyncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.}
 *
 * @testinput{- 'srcSyncFence' pointer points to NULL.}
 *
 * @testbehavior{- 'dstSyncFence->payload[0]' contains '0x8888888888888888' (dstFence->syncObj).
 * - 'dstSyncFence->payload[1..5]' contains '0xAAAAAAAAAAAAAAAA'. No operation on destination fence. Neither cleared nor duplicated.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851730}
 *
 * @verify{18844488}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.srcSyncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.dstSyncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceDup.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceDup
  lwscisync_fence.c.LwSciSyncFenceDup
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncFenceExtractFence

-- Test Case: TC_001.LwSciSyncFenceExtractFence.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceExtractFence.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceExtractFence.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - Memory pointed to by 'id' pointer is set to '0x1234ABCD'.
 * - Memory pointed to by 'value' pointer is set to '0x1A2B3C4D'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851733}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[3..4]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x1234ABCD
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x1A2B3C4D
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceExtractFence.syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceExtractFence.syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceExtractFence.syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x0' or NULL.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - Memory pointed to by 'id' pointer still contains the value '0x55'. No operation.
 * - Memory pointed to by 'value' pointer still contains the value '0x55'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851736}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[3..4]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceExtractFence.syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceExtractFence.syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceExtractFence.syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid LwSciSyncObj).
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceDup() panics.}
 *
 * @testcase{18851739}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[3..4]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceExtractFence.value_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceExtractFence.value_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceExtractFence.value_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when 'value' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to NULL.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- Memory pointed to by 'id' pointer still contains the value '0x55'. No operation.
 * - 'value' pointer still points to NULL. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851742}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[3..4]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncFenceExtractFence.id_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceExtractFence.id_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceExtractFence.id_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when 'id' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'id' pointer points to NULL.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- 'id' pointer still points to NULL. No operation.
 * - Memory pointed to by 'value' pointer still contains the value '0x55'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851745}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[3..4]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciSyncFenceExtractFence.syncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_006.LwSciSyncFenceExtractFence.syncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncFenceExtractFence.syncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when the memory pointed to by 'syncFence' pointer is already cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[1..5]' is set to '0x0'.}
 *
 * @testbehavior{- Memory pointed to by 'id' pointer still contains the value '0x55'. No operation.
 * - Memory pointed to by 'value' pointer still contains the value '0x55'. No operation.
 * - returns 'LwSciError_ClearedFence'.}
 *
 * @testcase{18851748}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence[0].payload[0..4]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_ClearedFence
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciSyncFenceExtractFence.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceExtractFence
TEST.NEW
TEST.NAME:TC_007.LwSciSyncFenceExtractFence.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncFenceExtractFence.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceExtractFence() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'id' pointer points to valid memory and the value at this memory is set to '0x55'.
 * - 'value' pointer points to valid memory and the value at this memory is set to '0x55'.}
 *
 * @testinput{- 'syncFence' pointer points to NULL.}
 *
 * @testbehavior{- Memory pointed to by 'id' pointer still contains the value '0x55'. No operation.
 * - Memory pointed to by 'value' pointer still contains the value '0x55'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851751}
 *
 * @verify{18844503}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.VALUE:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.id[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.value[0]:0x55
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceExtractFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceExtractFence
  lwscisync_fence.c.LwSciSyncFenceExtractFence
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncFenceGetSyncObj

-- Test Case: TC_001.LwSciSyncFenceGetSyncObj.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceGetSyncObj.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceGetSyncObj.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncObj' pointer points to a valid memory, and the value at that memory is set to the address '0x8888888888888888'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - Memory pointed to by 'syncObj' pointer is set to the address '0x9999999999999999'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851754}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
<<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
{{ <<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] == ( 0x9999999999999999 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceGetSyncObj.syncFence_syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceGetSyncObj.syncFence_syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceGetSyncObj.syncFence_syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() when 'syncFence' LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncObj' pointer points to a valid memory, and the value at that memory is set to the address '0x8888888888888888'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x0' or NULL.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - Memory pointed to by 'syncObj' pointer still contains the address '0x8888888888888888'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851757}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[0]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
<<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
{{ <<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceGetSyncObj.syncFence_syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceGetSyncObj.syncFence_syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceGetSyncObj.syncFence_syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() when 'syncFence' LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncObj' pointer points to a valid memory, and the value at that memory is set to the address '0x8888888888888888'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid dst LwSciSyncObj).
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceUpdateFence() panics.}
 *
 * @testcase{18851760}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<malloc 1>>
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
<<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceGetSyncObj.syncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceGetSyncObj.syncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceGetSyncObj.syncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() when the memory pointed to by 'syncFence' pointer is already cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncObj' pointer points to a valid memory, and the value at that memory is set to the address '0x8888888888888888'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[1..5]' is set to '0x0'.}
 *
 * @testbehavior{- Memory pointed to by 'syncObj' pointer still contains the address '0x8888888888888888'. No operation.
 * - returns 'LwSciError_ClearedFence'.}
 *
 * @testcase{18851763}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[0..5]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_ClearedFence
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
<<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
{{ <<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncFenceGetSyncObj.syncObj_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceGetSyncObj.syncObj_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceGetSyncObj.syncObj_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() when 'syncObj' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncObj' pointer points to NULL.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- 'syncObj' pointer still points to NULL. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851766}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciSyncFenceGetSyncObj.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceGetSyncObj
TEST.NEW
TEST.NAME:TC_006.LwSciSyncFenceGetSyncObj.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncFenceGetSyncObj.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceGetSyncObj() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncObj' pointer points to a valid memory, and the value at that memory is set to the address '0x8888888888888888'.}
 *
 * @testinput{- 'syncFence' pointer points to NULL.}
 *
 * @testbehavior{- Memory pointed to by 'syncObj' pointer still contains the address '0x8888888888888888'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851769}
 *
 * @verify{18844506}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceGetSyncObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
  lwscisync_fence.c.LwSciSyncFenceGetSyncObj
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
<<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj.syncObj[0]
{{ <<lwscisync_fence.LwSciSyncFenceGetSyncObj.syncObj>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncFenceUpdateFence

-- Test Case: TC_001.LwSciSyncFenceUpdateFence.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceUpdateFence.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceUpdateFence.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '0x1234ABCD'.
 * - 'value' is set to '0x1A2B3C4D'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '0x1234ABCD' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851772}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:0x1234ABCD
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:0x1A2B3C4D
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceUpdateFence.syncObj_refCount_is_max
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceUpdateFence.syncObj_refCount_is_max
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceUpdateFence.syncObj_refCount_is_max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when LwSciSyncObj reference count is maximum.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_IlwalidState'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '0x1234ABCD'.
 * - 'value' is set to '0x1A2B3C4D'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0..5]' is set to '0x0'. Fence is cleared.
 * - returns 'LwSciError_IlwalidState'.}
 *
 * @testcase{18851775}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_IlwalidState
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4]:EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[5]:EXPECTED_BASE=16
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceUpdateFence.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceUpdateFence.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceUpdateFence.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to NULL.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '0x1234ABCD'.
 * - 'value' is set to '0x1A2B3C4D'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence' pointer still points to NULL. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851778}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceUpdateFence.syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceUpdateFence.syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceUpdateFence.syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x0' or NULL.
 * - 'id' is set to '0x1234ABCD'.
 * - 'value' is set to '0x1A2B3C4D'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' contains '0x8888888888888888' (dstFence->syncObj).
 * - 'syncFence->payload[1..5]' contains '0xAAAAAAAAAAAAAAAA'. No operation on fence.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851781}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x0 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncFenceUpdateFence.syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceUpdateFence.syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceUpdateFence.syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid dst LwSciSyncObj).
 * - 'id' is set to '0x1234ABCD'.
 * - 'value' is set to '0x1A2B3C4D'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceUpdateFence() panics.}
 *
 * @testcase{18851784}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncFenceUpdateFence.id_equals_valid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_006.LwSciSyncFenceUpdateFence.id_equals_valid_LW
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncFenceUpdateFence.id_equals_valid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'id' is set to valid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '2147483647' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '2147483648' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851787}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:2147483647
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:2147483648
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncFenceUpdateFence.id_equals_ilwalid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_007.LwSciSyncFenceUpdateFence.id_equals_ilwalid_LW
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncFenceUpdateFence.id_equals_ilwalid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'id' is set to invalid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '5000000000'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' is still set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is still set to '0xAAAAAAAAAAAAAAAA'.
 * - returns 'LwSciError_Overflow'.}
 *
 * @testcase{18851790}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:5000000000
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncFenceUpdateFence.id_equals_LBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_008.LwSciSyncFenceUpdateFence.id_equals_LBV
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncFenceUpdateFence.id_equals_LBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'id' is set to lower boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '0'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '0' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '2147483648' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851793}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:2147483648
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncFenceUpdateFence.id_equals_UBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_009.LwSciSyncFenceUpdateFence.id_equals_UBV
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncFenceUpdateFence.id_equals_UBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'id' is set to upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '4294967294'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '4294967294' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '2147483648' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851796}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:4294967294
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:4294967294
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:2147483648
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncFenceUpdateFence.id_equals_above_UBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_010.LwSciSyncFenceUpdateFence.id_equals_above_UBV
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncFenceUpdateFence.id_equals_above_UBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'id' is set just above upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '4294967295'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' is still set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is still set to '0xAAAAAAAAAAAAAAAA'.
 * - returns 'LwSciError_Overflow'.}
 *
 * @testcase{18851799}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:4294967295
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncFenceUpdateFence.value_equals_valid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_011.LwSciSyncFenceUpdateFence.value_equals_valid_LW
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncFenceUpdateFence.value_equals_valid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'value' is set to valid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '2147483648'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '2147483647' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '2147483648' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851802}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:2147483648
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:2147483647
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:2147483648
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncFenceUpdateFence.value_equals_ilwalid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_012.LwSciSyncFenceUpdateFence.value_equals_ilwalid_LW
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncFenceUpdateFence.value_equals_ilwalid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'value' is set to invalid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '5000000000'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' is still set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is still set to '0xAAAAAAAAAAAAAAAA'.
 * - returns 'LwSciError_Overflow'.}
 *
 * @testcase{18851805}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:5000000000
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncFenceUpdateFence.value_equals_LBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_013.LwSciSyncFenceUpdateFence.value_equals_LBV
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncFenceUpdateFence.value_equals_LBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'value' is set to lower boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '0'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '2147483647' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '0' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851808}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:2147483647
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncFenceUpdateFence.value_equals_UBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_014.LwSciSyncFenceUpdateFence.value_equals_UBV
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncFenceUpdateFence.value_equals_UBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'value' is set to upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '4294967295'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '2147483647' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '4294967295' (coreFence->value).
 * - 'syncFence->payload[3]' is set to '0xFFFFFFFF' (coreFence->timestampSlot).
 * - 'syncFence->payload[4..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851811}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:4294967295
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..2]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1]:2147483647
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[2]:4294967295
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[3]:0xFFFFFFFF
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[4..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncFenceUpdateFence.value_equals_above_UBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceUpdateFence
TEST.NEW
TEST.NAME:TC_015.LwSciSyncFenceUpdateFence.value_equals_above_UBV
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncFenceUpdateFence.value_equals_above_UBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceUpdateFence() when 'value' is set just above upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is set to '0xAAAAAAAAAAAAAAAA'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'id' is set to '2147483647'.
 * - 'value' is set to '4294967296'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0]' is still set to the address '0x8888888888888888'.
 * - 'syncFence->payload[1..5]' is still set to '0xAAAAAAAAAAAAAAAA'.
 * - returns 'LwSciError_Overflow'.}
 *
 * @testcase{18851814}
 *
 * @verify{18844500}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.id:2147483647
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.value:4294967296
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.VALUE:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[0]:0x8888888888888888
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.syncFence[0].payload[1..5]:0xAAAAAAAAAAAAAAAA
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceUpdateFence.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncFenceUpdateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj
<<lwscisync_fence.LwSciSyncFenceUpdateFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncFenceWait

-- Test Case: TC_001.LwSciSyncFenceWait.normal_operation
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_001.LwSciSyncFenceWait.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncFenceWait.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() for normal operation.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '-1'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851817}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:-1
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x1234ABCD
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.ATTRIBUTES:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:EXPECTED_BASE=16
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncFenceWait.wait_not_complete_in_given_timeout
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_002.LwSciSyncFenceWait.wait_not_complete_in_given_timeout
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncFenceWait.wait_not_complete_in_given_timeout}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when wait did not complete in the given timeout.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Timeout'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Timeout'.}
 *
 * @testcase{18851820}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Timeout
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Timeout
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x1234ABCD
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncFenceWait.wait_operation_failed
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_003.LwSciSyncFenceWait.wait_operation_failed
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncFenceWait.wait_operation_failed}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when wait operation failed.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_ResourceError'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_ResourceError'.}
 *
 * @testcase{18851823}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x1234ABCD
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncFenceWait.fence_id_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_004.LwSciSyncFenceWait.fence_id_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncFenceWait.fence_id_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when fence id is out of valid range.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Overflow'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x100000000'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Overflow'.}
 *
 * @testcase{18851826}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x100000000
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Overflow
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x100000000
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncFenceWait.syncObj_underlying_primitive_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_005.LwSciSyncFenceWait.syncObj_underlying_primitive_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncFenceWait.syncObj_underlying_primitive_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the underlying primitive of syncObj is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x0' or NULL into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x100000000'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851829}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x100000000
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x100000000
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x0 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncFenceWait.get_primitive_internal_failure
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_006.LwSciSyncFenceWait.get_primitive_internal_failure
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncFenceWait.get_primitive_internal_failure}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when there is an internal failure in LwSciSyncCoreObjGetPrimitive().}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncCoreObjGetPrimitive() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x100000000'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851832}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x100000000
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
LwSciCommonPanic();
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncFenceWait.syncObj_attrList_is_not_cpu_waiter
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_007.LwSciSyncFenceWait.syncObj_attrList_is_not_cpu_waiter
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncFenceWait.syncObj_attrList_is_not_cpu_waiter}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the LwSciSyncCoreAttrList has needCpuAccess set to false.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'false' into memory pointed to by 'isCpuWaiter' pointer.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{
 * - LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851835}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncFenceWait.syncObj_attrList_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_008.LwSciSyncFenceWait.syncObj_attrList_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncFenceWait.syncObj_attrList_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the LwSciSyncCoreAttrList is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid attrList) into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851838}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncFenceWait.contextModule_and_fenceModule_are_not_the_same
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_009.LwSciSyncFenceWait.contextModule_and_fenceModule_are_not_the_same
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncFenceWait.contextModule_and_fenceModule_are_not_the_same}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the modules associated with LwSciSyncCpuWaitContext and LwSciSyncObj are not the same.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'false' into memory pointed to by 'isDup' pointer.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851841}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncFenceWait.memory_insufficient_to_duplicate_module
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_010.LwSciSyncFenceWait.memory_insufficient_to_duplicate_module
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncFenceWait.memory_insufficient_to_duplicate_module}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the memory is not sufficient to duplicate modules.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - returns 'LwSciError_InsufficientMemory'.}
 *
 * @testcase{18851844}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncFenceWait.contextModule_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_011.LwSciSyncFenceWait.contextModule_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncFenceWait.contextModule_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the module associated with LwSciSyncCpuWaitContext is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0xFFFFFFFFFFFFFFFF'(addresss of invalid module) as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851847}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncFenceWait.fenceModule_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_012.LwSciSyncFenceWait.fenceModule_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncFenceWait.fenceModule_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the module associated with LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid module) into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851850}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncFenceWait.fenceModule_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_013.LwSciSyncFenceWait.fenceModule_is_null
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncFenceWait.fenceModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the module reference associated with LwSciSyncObj is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x0' or NULL into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851853}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x0 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncFenceWait.contextModule_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_014.LwSciSyncFenceWait.contextModule_is_null
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncFenceWait.contextModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the module reference associated with LwSciSyncCpuWaitContext is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851856}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncFenceWait.context_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_015.LwSciSyncFenceWait.context_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncFenceWait.context_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when LwSciSyncCpuWaitContext is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid context).
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncFenceWait() panics.}
 *
 * @testcase{18851859}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncFenceWait.context_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_016.LwSciSyncFenceWait.context_is_null
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncFenceWait.context_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'context' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer points to NULL.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851862}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncFenceWait.syncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_019.LwSciSyncFenceWait.syncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncFenceWait.syncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the memory pointed to by 'syncFence' pointer is cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to the address '0x0'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- returns 'LwSciError_Success'.}
 *
 * @testcase{18851871}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0..5]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_020.LwSciSyncFenceWait.timeout_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_020.LwSciSyncFenceWait.timeout_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_020.LwSciSyncFenceWait.timeout_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when the given timeout is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '-2'.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851874}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:-2
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_021.LwSciSyncFenceWait.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_021.LwSciSyncFenceWait.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_021.LwSciSyncFenceWait.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to NULL.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '0x123'.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851877}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciSyncFenceWait.timeoutUs_equals_ilwalid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_022.LwSciSyncFenceWait.timeoutUs_equals_ilwalid_LW
TEST.NOTES:
/**
 * @testname{TC_022.LwSciSyncFenceWait.timeoutUs_equals_ilwalid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'timeoutUs' is set to invalid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '-10'.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851880}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:-10
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncFenceWait.timeoutUs_equals_valid_LW
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_023.LwSciSyncFenceWait.timeoutUs_equals_valid_LW
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncFenceWait.timeoutUs_equals_valid_LW}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'timeoutUs' is set to valid nominal value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '4294967295'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851883}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:4294967295
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x1234ABCD
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:4294967295
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncFenceWait.timeoutUs_equals_below_LBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_024.LwSciSyncFenceWait.timeoutUs_equals_below_LBV
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncFenceWait.timeoutUs_equals_below_LBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'timeoutUs' is set just below upper boundary value.}
 *
 * @casederiv{Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '-2'.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851886}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:-2
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_025.LwSciSyncFenceWait.timeoutUs_equals_LBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_025.LwSciSyncFenceWait.timeoutUs_equals_LBV
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncFenceWait.timeoutUs_equals_LBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'timeoutUs' is set to lower boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to '-1'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851889}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:-1
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.id:0x1234ABCD
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0] = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>>[0] = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
<<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>>[0] = ( 0x3333333333333333 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ *<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> == ( 0x4444444444444444 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter
{{ *<<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextValidate.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetModule.context>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.primitive>> == ( 0x3333333333333333 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext
{{ <<uut_prototype_stubs.LwSciSyncCoreWaitOnPrimitive.waitContext>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetModule.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetModule.module
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjGetModule.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup
{{ *<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncFenceWait.timeoutUs_equals_UBV
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncFenceWait
TEST.NEW
TEST.NAME:TC_026.LwSciSyncFenceWait.timeoutUs_equals_UBV
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncFenceWait.timeoutUs_equals_UBV}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncFenceWait() when 'timeoutUs' is set to upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreCpuWaitContextGetModule() returns '0x7777777777777777' as the address of LwSciSyncModule.
 * - LwSciSyncCoreObjGetModule() writes the address '0x6666666666666666' into the memory pointed to by 'module' pointer.
 * - LwSciSyncCoreModuleIsDup() writes 'true' into memory pointed to by 'isDup' pointer.
 * - LwSciSyncObjGetAttrList() returns 'LwSciError_Success'.
 * - LwSciSyncObjGetAttrList() writes the address '0x4444444444444444' into the memory pointed to by 'syncAttrList' pointer.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() writes 'true' into memory pointed to by 'isCpuWaiter' pointer.
 * - LwSciSyncCoreObjGetPrimitive() writes the address '0x3333333333333333' into the memory pointed to by 'primitive' pointer.
 * - LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'context' pointer is set to the address '0x8888888888888888'.
 * - 'timeoutUs' is set to 'UINT64_MAX'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreCpuWaitContextValidate() is called to validate LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreCpuWaitContextGetModule() is called to retrieve LwSciSyncModule associated with the LwSciSyncCpuWaitContext.
 * - LwSciSyncCoreObjGetModule() is called to retrieve the LwSciSyncModule associated with the given LwSciSyncObj using LwSciSyncCoreAttrListGetModule.
 * - LwSciSyncCoreModuleIsDup() is called to check if the given LwSciSyncModules are referring to the same module resource.
 * - LwSciSyncObjGetAttrList() is called to retrieve the reconciled LwSciSyncAttrList associated with an input LwSciSyncObj.
 * - LwSciSyncCoreAttrListTypeIsCpuWaiter() is called to check if slot 0 of the LwSciSyncCoreAttrList contained in LwSciSyncCoreAttrListObj referenced by the input LwSciSyncAttrList, has needCpuAccess set to true and actualPerm has LwSciSyncAccessPerm_WaitOnly.
 * - LwSciSyncCoreObjGetPrimitive() is called to retrieve the underlying LwSciSyncCorePrimitive that the given LwSciSyncObj is associated with.
 * - LwSciSyncCoreWaitOnPrimitive() is called to wait on the input syncpoint id and threshold value.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851892}
 *
 * @verify{18844491}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[1]:0x1234ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[2]:0x1A2B3C4D
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.timeoutUs:<<MAX>>
TEST.VALUE:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncFenceWait.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncFenceWait
  lwscisync_fence.c.LwSciSyncFenceWait
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncFenceWait.context
<<lwscisync_fence.LwSciSyncFenceWait.context>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncIpcExportFence

-- Test Case: TC_001.LwSciSyncIpcExportFence.normal_operation_lwrrent_ipcEndpoint_as_origin
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncIpcExportFence.normal_operation_lwrrent_ipcEndpoint_as_origin
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncIpcExportFence.normal_operation_lwrrent_ipcEndpoint_as_origin}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() for normal operation and export LwSciSyncFence with current ipcEndPoint as origin.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjGetId() writes the value '0x1A' into 'objId->moduleCntr'.
 * - LwSciSyncCoreObjGetId() writes the value '0x0' into 'objId->ipcEndpoint'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjGetId() is called to get the LwSciSyncCoreObjId identifying the given LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - 'desc->payload[0]' is set to '0x1A' (objId->moduleCntr).
 * - 'desc->payload[1]' is set to '0x123' (ipcEndpoint).
 * - 'desc->payload[2]' is set to '0x1234567890ABCD' (coreFence->id).
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089' (coreFence->value).
 * - 'desc->payload[4..6] is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851895}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].moduleCntr:0x1A
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].ipcEndpoint:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0]:0x1A
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[1]:0x123
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[4..6]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0x123
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:16,(2)8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].moduleCntr:0x0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].ipcEndpoint:0x0
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_fence.LwSciSyncIpcExportFence.desc>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcExportFence.desc>> + 16 ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcExportFence.desc>> + 24) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<uut_prototype_stubs.LwSciSyncCoreObjGetId.objId>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcExportFence.syncFence>>->payload[1]) ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcExportFence.syncFence>>->payload[2]) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetId.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetId.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncIpcExportFence.normal_operation_forward_origin_ipcEndpoint
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_002.LwSciSyncIpcExportFence.normal_operation_forward_origin_ipcEndpoint
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncIpcExportFence.normal_operation_forward_origin_ipcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() for normal operation and forward origin ipcEndPoint of LwSciSyncFence export.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjGetId() writes the value '0x1A' into 'objId->moduleCntr'.
 * - LwSciSyncCoreObjGetId() writes the value '0x07' into 'objId->ipcEndpoint'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjGetId() is called to get the LwSciSyncCoreObjId identifying the given LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - 'desc->payload[0]' is set to '0x1A' (objId->moduleCntr).
 * - 'desc->payload[1]' is set to '0x7' (objId->ipcEndpoint).
 * - 'desc->payload[2]' is set to '0x1234567890ABCD' (coreFence->id).
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089' (coreFence->value).
 * - 'desc->payload[4..6] is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851898}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3]:0x55555555123456EF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[4..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].moduleCntr:0x1A
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].ipcEndpoint:0x7
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0]:0x1A
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[1]:0x7
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[4..6]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0x123
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:16,(2)8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].moduleCntr:0x0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreObjGetId.objId[0].ipcEndpoint:0x0
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_fence.LwSciSyncIpcExportFence.desc>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcExportFence.desc>> + 16 ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcExportFence.desc>> + 24) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<uut_prototype_stubs.LwSciSyncCoreObjGetId.objId>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcExportFence.syncFence>>->payload[1]) ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcExportFence.syncFence>>->payload[2]) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjGetId.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjGetId.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncIpcExportFence.syncFence_is_already_cleared
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncIpcExportFence.syncFence_is_already_cleared
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncIpcExportFence.syncFence_is_already_cleared}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when the memory pointed to by 'syncFence' pointer is already cleared.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to the address '0x0'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint
 * - 'desc->payload[0..6]' is set to '0x0'. 'desc' memory is cleared.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851901}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0..5]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncIpcExportFence.syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncIpcExportFence.syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncIpcExportFence.syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x0' or NULL.
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'desc->payload[0..6]' still contains '0x5555555555555555'. No operation on 'desc' memory.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851904}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3]:0x55555555123456EF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[4..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncIpcExportFence.syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_005.LwSciSyncIpcExportFence.syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncIpcExportFence.syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid LwSciSyncObj).
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncIpcExportFence() panics.}
 *
 * @testcase{18851907}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3]:0x55555555123456EF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[4..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0x123
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncIpcExportFence.ipcEndpoint_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_006.LwSciSyncIpcExportFence.ipcEndpoint_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncIpcExportFence.ipcEndpoint_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when LwSciIpcEndpoint is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.
 * - LwSciIpcGetEndpointInfo() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0xFFFFFFFFFFFFFFFF'(value of invalid ipcEndpoint).}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint.
 * - 'desc->payload[0..6]' still contains '0x5555555555555555'. No operation on 'desc' memory.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851910}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3]:0x55555555123456EF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[4..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0xFFFFFFFFFFFFFFFF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0xFFFFFFFFFFFFFFFF
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncIpcExportFence.desc_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_007.LwSciSyncIpcExportFence.desc_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncIpcExportFence.desc_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when 'desc' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'desc' pointer points to NULL.}
 *
 * @testinput{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0]' is set to the address '0x9999999999999999'.
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD'.
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'syncFence->payload[3..5]' is set to '0x5555555555555555'.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- 'desc' pointer points to NULL. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851913}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[0]:0x9999999999999999
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[3]:0x55555555123456EF
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence[0].payload[4..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciSyncIpcExportFence.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcExportFence
TEST.NEW
TEST.NAME:TC_008.LwSciSyncIpcExportFence.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncIpcExportFence.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportFence() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'desc' pointer points to a valid memory.
 * - 'desc->payload[0..6]' is set to '0x5555555555555555'.}
 *
 * @testinput{- 'syncFence' pointer points to NULL.
 * - 'ipcEndPoint' is set to value '0x123'.}
 *
 * @testbehavior{- LwSciSyncCoreValidateIpcEndpoint() is called to validate. the input LwSciIpcEndpoint.
 * - 'desc->payload[0..6]' still contains '0x5555555555555555'. No operation on 'desc' memory.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851916}
 *
 * @verify{18844494}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.ipcEndpoint:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.desc[0].payload[0..6]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcExportFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcExportFence
  lwscisync_fence.c.LwSciSyncIpcExportFence
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncIpcImportFence

-- Test Case: TC_001.LwSciSyncIpcImportFence.normal_operation_non-empty_fence_desc
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncIpcImportFence.normal_operation_non-empty_fence_desc
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncIpcImportFence.normal_operation_non-empty_fence_desc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() for normal operation when fence descriptor is non-empty.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciSyncCoreObjMatchId() writes 'true' into the memory pointed to by 'isEqual' pointer.
 * - LwSciSyncObjRef() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciSyncCoreObjMatchId() is called to compare the given LwSciSyncCoreObjId against the LwSciSyncCoreObjId of the given LwSciSyncObj to check whether they refer to the same underlying Synchronization Object.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - 'syncFence->payload[0]' is set to '0x9999999999999999' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089' (coreFence->value).
 * - 'syncFence->payload[3..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851919}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.isEqual[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0]:0x9999999999999999
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[1]:0x1234567890ABCD
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[2]:0x1A2B3C4D5E6F7089
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[3..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:16,(2)8
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciSyncObjRef
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[1]) ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[2]) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_fence.LwSciSyncIpcImportFence.desc>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 16 ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 24) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->moduleCntr == ( 0x1A ) }}
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->ipcEndpoint == ( 0x123 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.isEqual
{{ *<<uut_prototype_stubs.LwSciSyncCoreObjMatchId.isEqual>> == ( false ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncIpcImportFence.normal_operation_empty_fence_desc
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_002.LwSciSyncIpcImportFence.normal_operation_empty_fence_desc
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncIpcImportFence.normal_operation_empty_fence_desc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() for normal operation when fence descriptor is empty.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[1..6]' is set to '0x0'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciSyncObjFreeObjAndRef() is called to free the underlying object and reference to the LwSciSyncObj.
 * - 'syncFence->payload[0..5]' is set to '0x0'.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18851922}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0..6]:0x0
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x0
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncIpcImportFence.syncObj_refCount_is_max
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncIpcImportFence.syncObj_refCount_is_max
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncIpcImportFence.syncObj_refCount_is_max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when LwSciSyncObj reference count is maximum.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciSyncCoreObjMatchId() writes 'true' into the memory pointed to by 'isEqual' pointer.
 * - LwSciSyncObjRef() returns 'LwSciError_IlwalidState'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciSyncCoreObjMatchId() is called to compare the given LwSciSyncCoreObjId against the LwSciSyncCoreObjId of the given LwSciSyncObj to check whether they refer to the same underlying Synchronization Object.
 * - LwSciSyncObjRef() is called to increment the reference count on the input LwSciSyncObj.
 * - 'syncFence->payload[0]' is set to '0x0' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089' (coreFence->value).
 * - 'syncFence->payload[3..5]' is set to '0x0'.
 * - returns 'LwSciError_IlwalidState'.}
 *
 * @testcase{18851925}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjRef.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.isEqual[0]:true
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_IlwalidState
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:16
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciSyncObjRef
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjRef.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[1]) ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[2]) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_fence.LwSciSyncIpcImportFence.desc>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 16 ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 24) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x9999999999999999 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->moduleCntr == ( 0x1A ) }}
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->ipcEndpoint == ( 0x123 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncIpcImportFence.non-empty_fence_desc_not_associated_to_syncObj
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncIpcImportFence.non-empty_fence_desc_not_associated_to_syncObj
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncIpcImportFence.non-empty_fence_desc_not_associated_to_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when the non-empty fence descriptor is not associated to syncObj.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_Success'.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.
 * - LwSciSyncCoreObjMatchId() writes 'false' into the memory pointed to by 'isEqual' pointer.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x8888888888888888'.
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer.
 * - LwSciSyncCoreObjMatchId() is called to compare the given LwSciSyncCoreObjId against the LwSciSyncCoreObjId of the given LwSciSyncObj to check whether they refer to the same underlying Synchronization Object.
 * - 'syncFence->payload[0]' is set to '0x0' (coreFence->syncObj).
 * - 'syncFence->payload[1]' is set to '0x1234567890ABCD' (coreFence->id).
 * - 'syncFence->payload[2]' is set to '0x1A2B3C4D5E6F7089' (coreFence->value).
 * - 'syncFence->payload[3..5]' is set to '0x0'.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851928}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:(2)LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.isEqual[0]:false
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:16
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreObjMatchId
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[1]) ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)&(<<lwscisync_fence.LwSciSyncIpcImportFence.syncFence>>->payload[2]) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_fence.LwSciSyncIpcImportFence.desc>> ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 16 ) }}
else if(cntr == 3) {{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( (uint8_t *)<<lwscisync_fence.LwSciSyncIpcImportFence.desc>> + 24) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
static int cntr = 0;
cntr++;

if(cntr == 1) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x8888888888888888 ) }}
else if(cntr == 2) {{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.syncObj>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->moduleCntr == ( 0x1A ) }}
{{ <<uut_prototype_stubs.LwSciSyncCoreObjMatchId.objId>>->ipcEndpoint == ( 0x123 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncIpcImportFence.syncObj_ref_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_005.LwSciSyncIpcImportFence.syncObj_ref_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncIpcImportFence.syncObj_ref_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when LwSciSyncObj reference is NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to the address '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x0' or NULL.
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - 'syncFence->payload[0..5]' still contains '0x5555555555555555'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851931}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0x0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x0 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncIpcImportFence.syncObj_is_ilwalid
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_006.LwSciSyncIpcImportFence.syncObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncIpcImportFence.syncObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when LwSciSyncObj is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to '0x5555555555555555'.
 * - LwSciSyncCoreObjValidate() panics.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0xFFFFFFFFFFFFFFFF'(addresss of invalid LwSciSyncObj).
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- LwSciSyncCoreObjValidate() is called to validate an LwSciSyncObj.
 * - LwSciCommonPanic() is called to terminate exelwtion of the program.
 * - LwSciSyncIpcImportFence() panics.}
 *
 * @testcase{18851934}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncIpcImportFence.syncFence_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_007.LwSciSyncIpcImportFence.syncFence_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncIpcImportFence.syncFence_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when 'syncFence' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'syncFence' pointer points to NULL.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'(addresss of invalid LwSciSyncObj).
 * - 'desc' pointer points to a valid memory.
 * - 'desc->payload[0]' is set to '0x1A'.
 * - 'desc->payload[1]' is set to '0x123'.
 * - 'desc->payload[2]' is set to '0x1234567890ABCD'.
 * - 'desc->payload[3]' is set to '0x1A2B3C4D5E6F7089'.
 * - 'desc->payload[4..6]' is set to '0x5555555555555555'.}
 *
 * @testbehavior{- 'syncFence' pointer still points to NULL. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851937}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[0]:0x1A
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[1]:0x123
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[2]:0x1234567890ABCD
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[3]:0x1A2B3C4D5E6F7089
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc[0].payload[4..6]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<null>>
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncIpcImportFence.desc_is_null
TEST.UNIT:lwscisync_fence
TEST.SUBPROGRAM:LwSciSyncIpcImportFence
TEST.NEW
TEST.NAME:TC_008.LwSciSyncIpcImportFence.desc_is_null
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncIpcImportFence.desc_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportFence() when 'desc' pointer points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'syncFence' pointer points to a valid memory.
 * - 'syncFence->payload[0..5]' is set to the address '0x5555555555555555'.}
 *
 * @testinput{- 'syncObj' pointer points to the address '0x9999999999999999'.
 * - 'desc' pointer points to NULL.}
 *
 * @testbehavior{- 'syncFence->payload[0..5]' still contains '0x5555555555555555'. No operation.
 * - returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18851940}
 *
 * @verify{18844497}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.desc:<<null>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.VALUE:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.syncFence[0].payload[0..5]:0x5555555555555555
TEST.EXPECTED:lwscisync_fence.LwSciSyncIpcImportFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_fence.c.LwSciSyncIpcImportFence
  lwscisync_fence.c.LwSciSyncIpcImportFence
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_fence.LwSciSyncIpcImportFence.syncObj
<<lwscisync_fence.LwSciSyncIpcImportFence.syncObj>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

