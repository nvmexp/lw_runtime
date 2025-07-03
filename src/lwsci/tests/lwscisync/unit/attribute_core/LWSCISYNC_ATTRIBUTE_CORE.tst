-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_ATTRIBUTE_CORE
-- Unit(s) Under Test: lwscisync_attribute_core
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:REMOVED_CL_PREFIX
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscisync_attribute_core

-- Subprogram: LwSciSyncAttrListAppendUnreconciled

-- Test Case: LwSciSyncAttrListAppendUnreconciled_attr_isnt_unreconsciled
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:LwSciSyncAttrListAppendUnreconciled_attr_isnt_unreconsciled
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListAppendUnreconciled_attr_isnt_Unreconciled}
 *
 * @verifyFunction{Argument checking failed because expression
 * ((*objAttrList)->state != LwSciSyncCoreAttrListState_Unreconciled
 * evaluates to FALSE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 0.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852546}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_001.LwSciSyncAttrListAppendUnreconciled_LwSciCalloc_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListAppendUnreconciled_LwSciCalloc_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListAppendUnreconciled_LwSciCalloc_return_FAIL}
 *
 * @verifyFunction{LwSciSyncCoreAttrListCreateMultiSlot() failed because it can't allocate
 * memory for new module.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciCommonCalloc, return NULL.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18852516}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjLock_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjLock_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjLock_return_FAIL}
 *
 * @verifyFunction{CopySlots failed because LwSciCommonObjLock() returned
 * status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciCommonObjLock(), return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852519}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjUnlock_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjUnlock_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListAppendUnreconciled_LwSciCommonObjUnlock_return_FAIL}
 *
 * @verifyFunction{CopySlots failed because LwSciCommonObjLock() returned
 * status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciCommonObjLock(), return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852522}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListAppendUnreconciled_LwSciCommonSort_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListAppendUnreconciled_LwSciCommonSort_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListAppendUnreconciled_LwSciCommonSort_return_FAIL}
 *
 * @verifyFunction{LwSciCommonSort() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonSort() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852525}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreCopyIpcTable_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreCopyIpcTable_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreCopyIpcTable_return_FAIL}
 *
 * @verifyFunction{Two-level deep CopySlots fail scenario:
 *  LwSciSyncCoreAttrListCopy() fails because LwSciSyncCoreCopyIpcTable() fails.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciSyncCoreCopyIpcTable(), return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852528}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FAIL}
 *
 * @verifyFunction{UnpackAttrList fails because LwSciSyncCoreModuleIsDup()
 * returns error code other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciSyncCoreModuleIsDup(), return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852531}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FALSE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_007.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FALSE
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncAttrListAppendUnreconciled_LwSciSyncCoreModuleIsDup_return_FALSE}
 *
 * @verifyFunction{UnpackAttrList fails because LwSciSyncCoreModuleIsDup() returns FALSE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciSyncCoreModuleIsDup(), bind false to its isDup output parameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852534}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListAppendUnreconciled_Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListAppendUnreconciled_Success
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListAppendUnreconciled_Success}
 *
 * @verifyFunction{Normal exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852537}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListAppendUnreconciled_attrListArray_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListAppendUnreconciled_attrListArray_is_NULL
TEST.BASIS_PATH:1 of 5
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListAppendUnreconciled_attrListArray_is_NULL}
 *
 * @verifyFunction{inputUnreconciledAttrListArray is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputUnreconciledAttrListArray set to NULL.
 *  - inputUnreconciledAttrListCount set to 0.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852540}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListAppendUnreconciled_attrListCount_equals_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListAppendUnreconciled_attrListCount_equals_to_0
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListAppendUnreconciled_attrListCount_equals_to_0}
 *
 * @verifyFunction{Argument checking failed because input array has length 0 (but array itself isn't NULL).}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not required.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 0.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852543}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncAttrListAppendUnreconciled_input_list_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_012.LwSciSyncAttrListAppendUnreconciled_input_list_is_null
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncAttrListAppendUnreconciled_input_list_is_null}
 *
 * @verifyFunction{inputUnreconciledAttrListArray is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 0.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852549}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncAttrListAppendUnreconciled_input_list_size_is_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_013.LwSciSyncAttrListAppendUnreconciled_input_list_size_is_0
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncAttrListAppendUnreconciled_input_list_size_is_0}
 *
 * @verifyFunction{inputUnreconciledAttrListCount is equal to 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 0.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852552}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:0
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncAttrListAppendUnreconciled_numCoreAttrList_is_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListAppendUnreconciled
TEST.NEW
TEST.NAME:TC_014.LwSciSyncAttrListAppendUnreconciled_numCoreAttrList_is_0
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncAttrListAppendUnreconciled_numCoreAttrList_is_0}
 *
 * @verifyFunction{LwSciSyncCoreAttrListCreateMultiSlot() failed because slotCount is 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set coreAttrList to NULL and numCoreAttrList to 0, 
 * set valid header.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852555}
 *
 * @verify{18844221}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].coreAttrList:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].numCoreAttrList:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListAppendUnreconciled
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[1][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncAttrListAppendUnreconciled.inputUnreconciledAttrListArray>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListClone

-- Test Case: TC_001.LwSciSyncAttrListClone_BadAlloc
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListClone_BadAlloc
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListClone_BadAlloc}
 *
 * @verifyFunction{LwSciCommonCalloc return NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for out pointer.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18852558}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListClone_BadAlloc2
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListClone_BadAlloc2
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListClone_BadAlloc2}
 *
 * @verifyFunction{LwSciCommonAllocObjWithRef return value other that LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonAllocObjWithRef() return LwSciError_InsufficientMemory.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for out pointer.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18852561}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListClone_LwSciCommonObjLock_return_ERROR
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListClone_LwSciCommonObjLock_return_ERROR
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListClone_LwSciCommonObjLock_return_ERROR}
 *
 * @verifyFunction{LwSciCommonObjLock() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852564}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] ->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0];
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListClone_LwSciCommonObjUnlock_return_ERROR
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListClone_LwSciCommonObjUnlock_return_ERROR
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListClone_LwSciCommonObjUnlock_return_ERROR}
 *
 * @verifyFunction{LwSciCommonObjUnlock() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjUnlock() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciCommonPanic.}
 *
 * @testcase{18852567}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] ->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0];
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListClone_LwSciSyncCoreAttrListCreateMultiSlot_parameter_check_failed
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListClone_LwSciSyncCoreAttrListCreateMultiSlot_parameter_check_failed
TEST.BASIS_PATH:3 of 11 (partial)
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListClone_LwSciSyncCoreAttrListCreateMultiSlot_parameter_check_failed}
 *
 * @verifyFunction{LwSciSyncCoreAttrListCreateMultiSlot() failed because its valueCount parameter is equal to 0.
 * This value is derived from numCoreAttrList field of coreAttrList input structure.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set numCoreAttrList to 0, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852570}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL}
 *
 * @verifyFunction{LwSciSyncCoreAttrListCopy() return status code other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreAttrListCopy() return LwSciError_BadParameter. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852573}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] ->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0];
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL2
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_007.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL2
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncAttrListClone_SciSyncCoreAttrListCopy_FAIL2}
 *
 * @verifyFunction{LwSciSyncCoreAttrListCopy() returned status code other than LwSciError_Success. 
 *
 * Reason: LwSciSyncCoreCopyIpcTable() returned status code other than LwSciError_Success. 
 *
 * After that, LwSciCommonObjUnlock() returned error code other than
 * LwSciError_Success, therefore cleanup failed.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreCopyIpcTable() return LwSciError_BadParameter. 
 *
 * LwSciCommonObjUnlock return LwSciError_BadParameter. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852576}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] ->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0];
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListClone_Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListClone_Success
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListClone_Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852579}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] ->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0];
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (  &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.ipcTable
{{ <<uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.ipcTable>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].coreAttrList[0].ipcTable ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleDup.module>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListClone_Success.keyState_locked
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListClone_Success.keyState_locked
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListClone_Success.keyState_locked}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * One of keystates is equal to LwSciSyncCoreAttrKeyState_SetLocked.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 * Set one on keyState fields to "locked" state.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852582}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
free(<<uut_prototype_stubs.LwSciCommonFree.ptr>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> == ( LwSciError_Success ); 
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (  &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.ipcTable
{{ <<uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.ipcTable>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].coreAttrList[0].ipcTable ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleDup.module>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].numCoreAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].numCoreAttrList = ( 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListClone_newAttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListClone_newAttrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListClone_newAttrList_is_NULL}
 *
 * @verifyFunction{newAttrList parameter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- origAttrList valid instance.
 *  - newAttrList NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter}
 *
 * @testcase{18852585}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList
<<lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncAttrListClone_origAttrList_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListClone
TEST.NEW
TEST.NAME:TC_011.LwSciSyncAttrListClone_origAttrList_is_null
TEST.BASIS_PATH:1 of 11
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListClone_origAttrList_is_null}
 *
 * @verifyFunction{origAttrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- origAttrList set to NULL.
 *  - newAttrList allocate space for pointer to output pointer.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852588}
 *
 * @verify{18844224}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.origAttrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListClone.newAttrList[0]:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListClone.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
  lwscisync_attribute_core.c.LwSciSyncAttrListClone
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncAttrListCreate

-- Test Case: TC_001.LwSciSyncAttrListCreate.attrList_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListCreate
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListCreate.attrList_is_null
TEST.BASIS_PATH:1 of 2
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListCreate.attrList_is_null}
 *
 * @verifyFunction{AttrList is null. Function fails on parameter check.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- module is null.
 *  - attrList set to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852591}
 *
 * @verify{18844206}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListCreate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListCreate.bad_alloc
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListCreate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListCreate.bad_alloc
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListCreate.bad_alloc}
 *
 * @verifyFunction{AttrList is null. Function fails on parameter check.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc return NULL}
 *
 * @testinput{- module is null.
 *  - attrList set to valid instance.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18852594}
 *
 * @verify{18844206}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListCreate.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListCreate.duplicate_fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListCreate
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListCreate.duplicate_fail
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListCreate.duplicate_fail}
 *
 * @verifyFunction{LwSciSyncCoreModuleDup() returned value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return NULL.
 *
 * LwSciSyncCoreModuleDup return LwSciError_BadParameter.
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- module is null.
 *  - attrList set to valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852597}
 *
 * @verify{18844206}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListCreate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListCreate.module_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListCreate
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListCreate.module_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListCreate.module_ilwalid}
 *
 * @verifyFunction{LwSciSyncCoreModuleValidate() returns status other than LwSciError_Success}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() return LwSciError_BadParameter.}
 *
 * @testinput{- module is null.
 *  - attrList set to valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852600}
 *
 * @verify{18844206}
 */
TEST.END_NOTES:
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListCreate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListCreate.success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListCreate
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListCreate.success
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListCreate.success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc map to calloc from CRT.
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- module is null.
 *  - attrList set to valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852603}
 *
 * @verify{18844206}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListCreate.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  lwscisync_attribute_core.c.LwSciSyncAttrListCreate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == (1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == (sizeof(LwSciSyncCoreAttrList)) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == (sizeof(LwSciSyncCoreAttrListObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncAttrListRec)) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( <<lwscisync_attribute_core.LwSciSyncAttrListCreate.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleDup.module>> == (<<lwscisync_attribute_core.LwSciSyncAttrListCreate.module>>) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListCreate.module
<<lwscisync_attribute_core.LwSciSyncAttrListCreate.module>> = (malloc(10));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListCreate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListFree

-- Test Case: TC_001.LwSciSyncAttrListFree_ptr_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListFree_ptr_is_null
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListFree_ptr_is_null}
 *
 * @verifyFunction{argument is null.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to NULL.}
 *
 * @testbehavior{No-op behavior.}
 *
 * @testcase{18852606}
 *
 * @verify{18844209}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListFree.attrList:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListFree
  lwscisync_attribute_core.c.LwSciSyncAttrListFree
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListFree_success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListFree_success
TEST.BASIS_PATH:2 of 3
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListFree_success}
 *
 * @verifyFunction{Main exelwtion scenario.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to valid instance.}
 *
 * @testbehavior{Function should forward request to LwSciCommonFreeObjAndRef().}
 *
 * @testcase{18852609}
 *
 * @verify{18844209}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_attribute_core.c.LwSciSyncAttrListFree
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListFree.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListFree.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListGetAttr

-- Test Case: TC_001.LwSciSyncAttrListGetAttr_Key_Ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttr
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListGetAttr_Key_Ilwalid
TEST.BASIS_PATH:4 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListGetAttr_Key_Ilwalid}
 *
 * @verifyFunction{invalid key.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key set arbitrary value outside of LwSciSyncAttrKey_LowerBound .. LwSciSyncAttrKey_UpperBound.
 *  - value allocate memory for a pointer.
 *  - len allocate memory for size output parameter.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852612}
 *
 * @verify{18844230}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.len:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.key
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttr.key>> = ( LwSciSyncAttrKey_LowerBound - 1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListGetAttr_Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttr
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListGetAttr_Success
TEST.BASIS_PATH:3 of 4
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListGetAttr_Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to LwSciSyncAttrKey_NeedCpuAccess.
 *  - value allocate memory for a pointer.
 *  - len allocate memory for size output parameter.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852615}
 *
 * @verify{18844230}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.len:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListGetAttr_len_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttr
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListGetAttr_len_is_NULL
TEST.BASIS_PATH:2 of 4
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListGetAttr_len_is_NULL}
 *
 * @verifyFunction{len is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to valid value.
 *  - value set to NULL.
 *  - len allocate memory for size output parameter.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852618}
 *
 * @verify{18844230}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.len:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList>> = (<<lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListGetAttr_value_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttr
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListGetAttr_value_is_NULL
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListGetAttr_value_is_NULL}
 *
 * @verifyFunction{value is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to valid value.
 *  - value allocate memory for a pointer
 *  - len set to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852621}
 *
 * @verify{18844230}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.value:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.len:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttr
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttr.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListGetAttrs

-- Test Case: TC_001.LwSciSyncAtrListGetAttrs_Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttrs
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAtrListGetAttrs_Success
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAtrListGetAttrs_Success}
 *
 * @verifyFunction{Main exelwtion scenario. No error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852624}
 *
 * @verify{18844215}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray[0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray[0].len:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray.pairArray[0].value
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray>>[0].value = ( malloc(1) );

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAtrListGetAttrs_attrKey_is_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttrs
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAtrListGetAttrs_attrKey_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAtrListGetAttrs_attrKey_is_ilwalid}
 *
 * @verifyFunction{AttrListSlotGetAttrs returns status other than LwSciError_Success because
 * index of one of the keys lies outside LwSciSyncAttrKey_LowerBound .. LwSciSyncAttrKey_UpperBound
 * range.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to valid instance, set attrKey to LwSciSyncAttrKey_UpperBound + 1.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852627}
 *
 * @verify{18844215}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray[0].len:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray.pairArray[0].attrKey
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray>>[0].attrKey = ( LwSciSyncAttrKey_UpperBound + 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray.pairArray[0].value
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray>>[0].value = ( malloc(1) );

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAtrListGetAttrs_numCoreAttrList_equals_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttrs
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAtrListGetAttrs_numCoreAttrList_equals_to_0
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAtrListGetAttrs_numCoreAttrList_equals_to_0}
 *
 * @verifyFunction{attrList is null. This function fails on argument verification stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set valid header. Set coreAttrList to NULL
 * and numCoreAttrList to 0.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to valid instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852630}
 *
 * @verify{18844215}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAtrListGetAttrs_pairArray_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttrs
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAtrListGetAttrs_pairArray_is_null
TEST.BASIS_PATH:1 of 2
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAtrListGetAttrs_pairArray_is_null}
 *
 * @verifyFunction{pairArray is null. This function fails on argument verification stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set valid header. Set coreAttrList to NULL
 * and numCoreAttrList to 0.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to NULL.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852633}
 *
 * @verify{18844215}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairCount:<<MIN>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAtrListGetAttrs_pairCount_equals_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetAttrs
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAtrListGetAttrs_pairCount_equals_to_0
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAtrListGetAttrs_pairCount_equals_to_0}
 *
 * @verifyFunction{PairCount is 0. This function fails on argument verifiaction stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set valid header. Set coreAttrList to NULL
 * and numCoreAttrList to 0.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to valid instance.
 *  - pairCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852636}
 *
 * @verify{18844215}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.pairCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListGetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListGetInternalAttrs

-- Test Case: TC_001.LwSciSyncAttrListGetInternalAttrs_Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetInternalAttrs
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListGetInternalAttrs_Success
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListGetInternalAttrs_Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize internalAttrKeyValuePair instance, assign valid values to attrKey, value and len fields. }
 *
 * @testinput{- attrList valid instance.
 *  - pairArray reference to internalAttrKeyValuePair global variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852639}
 *
 * @verify{18844236}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].attrKey:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].len:16
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair.internalAttrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0][0].value = ( malloc(16) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListGetInternalAttrs_attrKey_is_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetInternalAttrs
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListGetInternalAttrs_attrKey_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListGetInternalAttrs_attrKey_is_ilwalid}
 *
 * @verifyFunction{attrKey has value outside of 
 * LwSciSyncInternalAttrKey_LowerBound .. LwSciSyncInternalAttrKey_UpperBound range.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize internalAttrKeyValuePair instance, assign plausible value and len fields, put invalid value to attrKey.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852642}
 *
 * @verify{18844236}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair.internalAttrKeyValuePair[0][0].attrKey
<<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0][0].attrKey = (  LwSciSyncInternalAttrKey_LowerBound - 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair.internalAttrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListGetInternalAttrs_numCoreAttrList_equals_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetInternalAttrs
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListGetInternalAttrs_numCoreAttrList_equals_to_0
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListGetInternalAttrs_numCoreAttrList_equals_to_0}
 *
 * @verifyFunction{numCoreAttrList is equal to 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set numCoreAttrList to 0.
 *
 * Initialize internalAttrKeyValuePair instance, assign plausible value and len fields, put invalid value to attrKey.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852645}
 *
 * @verify{18844236}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListGetInternalAttrs_pairArray_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetInternalAttrs
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListGetInternalAttrs_pairArray_is_null
TEST.BASIS_PATH:1 of 2
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListGetInternalAttrs_pairArray_is_null}
 *
 * @verifyFunction{pairArray is NULL, function fails on argument validation stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray set to NULL.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852648}
 *
 * @verify{18844236}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListGetInternalAttrs_pairCount_equals_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetInternalAttrs
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListGetInternalAttrs_pairCount_equals_to_0
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListGetInternalAttrs_pairCount_equals_to_0}
 *
 * @verifyFunction{pairCount is 0, function fails on argument validation stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray set to valid array anstance containing single element.
 *  - pairCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852651}
 *
 * @verify{18844236}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.pairCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListGetInternalAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListGetSingleInternalAttr

-- Test Case: TC_001.LwSciSyncAttrListGetSingleInternalAttr.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSingleInternalAttr
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListGetSingleInternalAttr.Success
TEST.BASIS_PATH:4 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListGetSingleInternalAttr.Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key value in LwSciSyncInternalAttrKey_LowerBound ..  LwSciSyncInternalAttrKey_UpperBound range.
 *  - value allocate memory for output pointer.
 *  - len allocate memory for output size.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852654}
 *
 * @verify{18844239}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListGetSingleInternalAttr.cant_get_internal_attrs.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSingleInternalAttr
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListGetSingleInternalAttr.cant_get_internal_attrs.Fail
TEST.BASIS_PATH:3 of 4
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListGetSingleInternalAttr.cant_get_internal_attrs.Fail}
 *
 * @verifyFunction{key is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to arbitrary invalid valuevalue.
 *  - value allocate memory for output pointer.
 *  - len allocate memory for output size.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852657}
 *
 * @verify{18844239}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:<<testcase>>
size_t* pLen = malloc(sizeof(size_t));
*pLen = 1;
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len>> = (pLen);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.key
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.key>> = ( LwSciSyncInternalAttrKey_LowerBound - 1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListGetSingleInternalAttr.p_to_len_is_null.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSingleInternalAttr
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListGetSingleInternalAttr.p_to_len_is_null.Fail
TEST.BASIS_PATH:2 of 4
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListGetSingleInternalAttr.p_to_len_is_null.Fail}
 *
 * @verifyFunction{output parameter len is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to arbitrary invalid valuevalue.
 *  - value allocate memory for output pointer.
 *  - len set to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852660}
 *
 * @verify{18844239}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.key:<<MIN>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.value:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListGetSingleInternalAttr.p_to_value_is_null.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSingleInternalAttr
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListGetSingleInternalAttr.p_to_value_is_null.Fail
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListGetSingleInternalAttr.p_to_value_is_null.Fail}
 *
 * @verifyFunction{output parameter len is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - key set to arbitrary invalid valuevalue.
 *  - value set to null.
 *  - len allocate memory for output pointer.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852663}
 *
 * @verify{18844239}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.key:<<MIN>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.value:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.len:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSingleInternalAttr
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSingleInternalAttr.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListGetSlotCount

-- Test Case: TC_001.LwSciSyncAttrListGetSlotCount.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSlotCount
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListGetSlotCount.Success
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListGetSlotCount.Success}
 *
 * @verifyFunction{Main exelwtion scenario. No error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set valid header.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.}
 *
 * @testbehavior{Function should return 0.}
 *
 * @testcase{18852666}
 *
 * @verify{18844218}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListGetSlotCount.return:0
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSlotCount
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSlotCount
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSlotCount.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSlotCount.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListGetSlotCount.cant_validate_attr_list.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListGetSlotCount
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListGetSlotCount.cant_validate_attr_list.Fail
TEST.BASIS_PATH:2 of 3
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListGetSlotCount.cant_validate_attr_list.Fail}
 *
 * @verifyFunction{coreAttrListObj has invalid header.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header to arbitrary value.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.}
 *
 * @testbehavior{Function cause abnormal program termination by ilwoking LwSciCommonPanic call.}
 *
 * @testcase{18852669}
 *
 * @verify{18844218}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].header:13
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSlotCount
  lwscisync_attribute_core.c.LwSciSyncAttrListGetSlotCount
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListGetSlotCount.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListGetSlotCount.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListSetAttrs

-- Test Case: LwSciSyncAttrListSetAttrs_kvpair_is_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:LwSciSyncAttrListSetAttrs_kvpair_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListSetAttrs_kvpair_is_ilwalid}
 *
 * @verifyFunction{attrKeyValuePair has attrKey outside of LwSciSyncAttrKey_LowerBound .. LwSciSyncAttrKey_UpperBound range.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance, set attrKey  = -1. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{22060798}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:-1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList)  }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_001.LwSciSyncAttrListSetAttrs_IsAttributeWritable_is_FALSE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListSetAttrs_IsAttributeWritable_is_FALSE
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListSetAttrs_IsAttributeWritable_is_FALSE}
 *
 * @verifyFunction{IsAttributeWritable returned false because keyState isn't equal to LwSciSyncCoreAttrKeyState_SetUnlocked.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjUnlock() return LwSciError_BadParameter. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList.
 * Set keyState to  LwSciSyncCoreAttrKeyState_SetLocked.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to global attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852672}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListSetAttrs_IsValueLenSane_is_false
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListSetAttrs_IsValueLenSane_is_false
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListSetAttrs_IsValueLenSane_is_false}
 *
 * @verifyFunction{attrKeyValuePair contains staructure with invalid len.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance, set len to illegal arbitrary value. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852675}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList)  }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(13) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListSetAttrs_attrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListSetAttrs_attrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListSetAttrs_attrList_is_NULL}
 *
 * @verifyFunction{attrList argument is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize attrListKeyValuePair global instance. }
 *
 * @testinput{- attrList set to NULL.
 *  - pairArray set to global attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852678}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListSetAttrs_attrList_isnt_writable
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListSetAttrs_attrList_isnt_writable
TEST.BASIS_PATH:2 of 9 (partial)
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListSetAttrs_attrList_isnt_writable}
 *
 * @verifyFunction{Sample attribute list has writable = false, thus IsAttrListWritable() returns false and function fails.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to global sttrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852681}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0]  );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListSetAttrs_cant_lock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListSetAttrs_cant_lock
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListSetAttrs_cant_lock}
 *
 * @verifyFunction{LwSciCommonObjLock() returned status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock() return LwSciError_BadParameter. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to global attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should termanate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852684}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = (  <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListSetAttrs_cant_unlock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListSetAttrs_cant_unlock
TEST.BASIS_PATH:4 of 9 (partial)
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListSetAttrs_cant_unlock}
 *
 * @verifyFunction{LwSciCommonObjUnlock() returned status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjUnlock() return LwSciError_BadParameter. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to global attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should termanate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852687}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncAttrListSetAttrs_kvpair_value_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_007.LwSciSyncAttrListSetAttrs_kvpair_value_is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncAttrListSetAttrs_kvpair_value_is_NULL}
 *
 * @verifyFunction{attrKeyValuePair contains staructure with value being equal to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance, set value to null. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852690}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( 
   (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> ==  <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>) 
              ? *<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>
              : NULL );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList)  }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListSetAttrs_pairArray_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListSetAttrs_pairArray_is_NULL
TEST.BASIS_PATH:1 of 9
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListSetAttrs_pairArray_is_NULL}
 *
 * @verifyFunction{When pairArray is NULL argument verification routine should return error.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList.
 * Set keyState to  LwSciSyncCoreAttrKeyState_SetLocked.}
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to NULL.
 *  - pairCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852693}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListSetAttrs_pairCount_is_equal_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListSetAttrs_pairCount_is_equal_to_0
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListSetAttrs_pairCount_is_equal_to_0}
 *
 * @verifyFunction{When pairCount is 0 argument verification routine should return error.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList.
 * Set keyState to  LwSciSyncCoreAttrKeyState_SetLocked.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to global attrKeyValuePair instance.
 *  - pairCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852696}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListSetAttrs_success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetAttrs
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListSetAttrs_success
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListSetAttrs_success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return coreAttrListObj global instance. 
 *
 * Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList,
 * set writable property to false.
 *
 * Initialize attrListKeyValuePair global instance. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to attrKeyValuePair instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852699}
 *
 * @verify{18844212}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList)  }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListSetInternalAttrs

-- Test Case: TC_001.LwSciSyncAttrListSetInternalAttrs_ValidateKeyValuePair_return_FALSE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListSetInternalAttrs_ValidateKeyValuePair_return_FALSE
TEST.BASIS_PATH:3 of 9 (partial)
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListSetInternalAttrs_ValidateKeyValuePair_return_FALSE}
 *
 * @verifyFunction{coreAttrListObj is malformed.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set coreAttrList fields and header to arbitrary invalid value.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852702}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListSetInternalAttrs_attrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListSetInternalAttrs_attrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListSetInternalAttrs_attrList_is_NULL}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList set to NULL.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852705}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].attrKey:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListSetInternalAttrs_attrList_isnt_writable
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListSetInternalAttrs_attrList_isnt_writable
TEST.BASIS_PATH:2 of 9 (partial)
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListSetInternalAttrs_attrList_isnt_writable}
 *
 * @verifyFunction{Sample attribute list has writable = false, thus IsAttrListWritable() returns false and function fails.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set writable to false.}
 *
 * @testinput{- attrList set to NULL.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852708}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].attrKey:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListSetInternalAttrs_cant_lock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListSetInternalAttrs_cant_lock
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListSetInternalAttrs_cant_lock}
 *
 * @verifyFunction{LwSciCommonObjLock() returned status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set writable to false.}
 *
 * @testinput{- attrList set to NULL.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852711}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].attrKey:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].len:4
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair.internalAttrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListSetInternalAttrs_cant_unlock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListSetInternalAttrs_cant_unlock
TEST.BASIS_PATH:4 of 9 (partial)
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListSetInternalAttrs_cant_unlock}
 *
 * @verifyFunction{LwSciCommonObjUnlock() return status other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjUnlock() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set writable to false.}
 *
 * @testinput{- attrList set to NULL.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852714}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListSetInternalAttrs_pairArray_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListSetInternalAttrs_pairArray_is_NULL
TEST.BASIS_PATH:1 of 9
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListSetInternalAttrs_pairArray_is_NULL}
 *
 * @verifyFunction{When pairArray is NULL argument verification routine should return error.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjUnlock() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set writable to false.}
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to NULL.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852717}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncAttrListSetInternalAttrs_pairCount_is_equal_to_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_007.LwSciSyncAttrListSetInternalAttrs_pairCount_is_equal_to_0
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncAttrListSetInternalAttrs_pairCount_is_equal_to_0}
 *
 * @verifyFunction{When pairCount is 0 argument verification routine should return error.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set writable to false.}
 *
 * @testinput{- attrList set to valid instance.
 *  - pairArray set to array containing single valid instance.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852720}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListSetInternalAttrs_success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSetInternalAttrs
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListSetInternalAttrs_success
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListSetInternalAttrs_success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852723}
 *
 * @verify{18844233}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].writable:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].attrKey:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair[0][0].len:4
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSetInternalAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair.internalAttrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSetInternalAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.internalAttrKeyValuePair>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListSlotGetAttrs

-- Test Case: TC_001.LwSciSyncAttrListSlotGetAttrs.attr_list_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListSlotGetAttrs.attr_list_is_null
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListSlotGetAttrs.attr_list_is_null}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to NULL.
 *  - slotIndex set to 0.
 *  - pairArray reference to attrKeyValuePair global variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852726}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListSlotGetAttrs.cant_lock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListSlotGetAttrs.cant_lock
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListSlotGetAttrs.cant_lock}
 *
 * @verifyFunction{LwSciCommonObjLock return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList set to valid instance.
 *  - slotIndex set to 0.
 *  - pairArray reference to attrKeyValuePair global variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852729}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListSlotGetAttrs.cant_unlock
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListSlotGetAttrs.cant_unlock
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListSlotGetAttrs.cant_unlock}
 *
 * @verifyFunction{LwSciCommonObjUnlock return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList set to valid instance.
 *  - slotIndex set to 0.
 *  - pairArray reference to attrKeyValuePair global variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852732}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListSlotGetAttrs.key_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListSlotGetAttrs.key_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListSlotGetAttrs.key_ilwalid}
 *
 * @verifyFunction{PublicAttrKeyIsValid() return FALSE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonObjLock return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize attrKeyValuePair, set attrKey to ilwalidValue.}
 *
 * @testinput{- attrList set to valid instance.
 *  - slotIndex set to 0.
 *  - pairArray reference to attrKeyValuePair global variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852735}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].attrKey
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].attrKey = ( LwSciSyncAttrKey_LowerBound - 1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListSlotGetAttrs.pair_array_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListSlotGetAttrs.pair_array_is_null
TEST.BASIS_PATH:1 of 1
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListSlotGetAttrs.pair_array_is_null}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to valid instance.
 *  - slotIndex set to 0.
 *  - pairArray set to NULL.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852738}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListSlotGetAttrs.success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncAttrListSlotGetAttrs
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListSlotGetAttrs.success
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListSlotGetAttrs.success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - slotIndex set to 0.
 *  - pairArray reference attrKeyValuePair local variable.
 *  - pairCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852741}
 *
 * @verify{18844227}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.valSize[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].attrKey:LwSciSyncAttrKey_NeedCpuAccess
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0][0].len:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.slotIndex:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncAttrListSlotGetAttrs
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair.attrKeyValuePair[0][0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair>>[0][0].value = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray
<<lwscisync_attribute_core.LwSciSyncAttrListSlotGetAttrs.pairArray>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrKeyValuePair[0]>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListAppendUnreconciledWithLocks

-- Test Case: TC_001.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.LwSciSyncCoreCopyIpcTable_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.LwSciSyncCoreCopyIpcTable_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.LwSciSyncCoreCopyIpcTable_return_FAIL}
 *
 * @verifyFunction{LwSciSyncCoreCopyIpcTable return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create stub for LwSciSyncCoreCopyIpcTable returning LwSciError_InsufficientMemory. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Unreconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return InsufficientMemory.}
 *
 * @testcase{18852744}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyIpcTable.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_false
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_false
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_false}
 *
 * @verifyFunction{Success scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Unreconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to false.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852747}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_true
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_true
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.Success_acquireLocks_is_true}
 *
 * @verifyFunction{Success scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Unreconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852750}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.any_of_the_input_LwSciSyncAttrLists_are_not_unreconciled
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.any_of_the_input_LwSciSyncAttrLists_are_not_unreconciled
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.any_of_the_input_LwSciSyncAttrLists_are_not_unreconciled}
 *
 * @verifyFunction{One of the input attr lists is not unreconciled.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852753}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray_is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray_is_NULL}
 *
 * @verifyFunction{inputUnreconciledAttrListArray is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputUnreconciledAttrListArray set to NULL.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852756}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount_is_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount_is_0
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount_is_0}
 *
 * @verifyFunction{inputUnreconciledAttrListCount is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 0
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852759}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.module_isnt_dup
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.module_isnt_dup
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.module_isnt_dup}
 *
 * @verifyFunction{Elements of attribute list aren't bound to a single module.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize array of two global coreAttrListObj instance,
 * set module fields to arbitrary non-equal values,
 * set state field to LwSciSyncCoreAttrListState_Unreconciled, 
 * set valid header and coreAttrList fields.
 *
 * Create LwSciSyncCoreModuleIsDup() stub returning false if its input module parameters aren't equal.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 2.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParemeter.}
 *
 * @testcase{18852762}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][1].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][1].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][1].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup.isDup[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup>>[0] = ( <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module = ( malloc(10));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][1].module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][1].module = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[1].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] + 1);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList_is_NULL}
 *
 * @verifyFunction{newUnreconciledAttrList parameter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Unreconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- inputUnreconciledAttrListArray valid instance.
 *  - inputUnreconciledAttrListCount set to 1.
 *  - acquireLocks set to true.
 *  - newUnreconciledAttrList set to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852765}
 *
 * @verify{18844302}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>>);
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]>> = malloc(<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>>);
(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->objPtr = <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]>>;
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListCreateMultiSlot

-- Test Case: TC_001.LwSciSyncCoreAttrListCreateMultiSlot.Fail.LwSyncCoreModuleDup.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListCreateMultiSlot
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListCreateMultiSlot.Fail.LwSyncCoreModuleDup.Fail
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListCreateMultiSlot.Fail.LwSyncCoreModuleDup.Fail}
 *
 * @verifyFunction{LwSciSyncCoreModuleDup() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreModuleDup() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- module set to arbitrary pointer value.
 *  - valueCount set to 1.
 *  - attr list valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852768}
 *
 * @verify{18844278}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr.refPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:<<testcase>>
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListCreateMultiSlot.Fail.attrList_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListCreateMultiSlot
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListCreateMultiSlot.Fail.attrList_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListCreateMultiSlot.Fail.attrList_is_null}
 *
 * @verifyFunction{attrList set to null.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- module set to arbitrary pointer value.
 *  - valueCount set to 1.
 *  - attrList set to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParemeter.}
 *
 * @testcase{18852771}
 *
 * @verify{18844278}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:<<testcase>>
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListCreateMultiSlot.Fail.cant_validate_attr_list
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListCreateMultiSlot
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListCreateMultiSlot.Fail.cant_validate_attr_list
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListCreateMultiSlot.Fail.cant_validate_attr_list}
 *
 * @verifyFunction{LwSciSyncCoreModuleValidate() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() return LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- module set to arbitrary pointer value.
 *  - valueCount set to 1.
 *  - attr list valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852774}
 *
 * @verify{18844278}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:<<testcase>>
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListCreateMultiSlot.Fail.valueCount_is_zero
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListCreateMultiSlot
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListCreateMultiSlot.Fail.valueCount_is_zero
TEST.BASIS_PATH:5 of 10 (partial)
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListCreateMultiSlot.Fail.valueCount_is_zero}
 *
 * @verifyFunction{valueCount is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- module set to arbitrary pointer value.
 *  - valueCount set to 0.
 *  - attr list valid instance.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852777}
 *
 * @verify{18844278}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:0
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
TEST.END_FLOW
TEST.VALUE_USER_CODE:<<testcase>>
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.module
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.module>> = (malloc(10));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListCreateMultiSlot.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListCreateMultiSlot
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListCreateMultiSlot.Success
TEST.BASIS_PATH:10 of 10 (partial)
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListCreateMultiSlot.Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- module set to arbitrary pointer value.
 *  - valueCount set to 1.
 *  - attr list valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852780}
 *
 * @verify{18844278}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListCreateMultiSlot
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:<<testcase>>
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.module
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.module>> = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListCreateMultiSlot.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListDup

-- Test Case: TC_001.LwSciSyncAttrListDup.Fail.attrList_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListDup
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListDup.Fail.attrList_is_null
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListDup.Fail.attrList_is_null}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - dupAttrList allocate space for output pointer.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciComminPanic().}
 *
 * @testcase{18852783}
 *
 * @verify{18844248}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListDup.Fail.cant_duplicate_ref
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListDup
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListDup.Fail.cant_duplicate_ref
TEST.BASIS_PATH:3 of 4
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListDup.Fail.cant_duplicate_ref}
 *
 * @verifyFunction{LwSciCommonDuplicateRef() return value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonDuplicateRef() return value other than LwSciError_BadParameter.
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList valid instance.
 *  - dupAttrList allocate space for output pointer.}
 *
 * @testbehavior{Function should return LwSciError_ResourceError.}
 *
 * @testcase{18852786}
 *
 * @verify{18844248}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_ResourceError
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.return:LwSciError_ResourceError
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListDup.Fail.dupAttrList_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListDup
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListDup.Fail.dupAttrList_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListDup.Fail.dupAttrList_is_null}
 *
 * @verifyFunction{dupAttrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList valid instance.
 *  - dupAttrList is NULL.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852789}
 *
 * @verify{18844248}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListDup.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListDup
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListDup.Success
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListDup.Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList valid instance.
 *  - dupAttrList allocate space for output pointer.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852792}
 *
 * @verify{18844248}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListDup.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListGetModule

-- Test Case: TC_001.LwSciSyncCoreAttrListGetModule_AttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListGetModule
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListGetModule_AttrList_is_NULL
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListGetModule_AttrList_is_NULL}
 *
 * @verifyFunction{attrList set to null.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList set to NULL.
 *  - module valid instance.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852795}
 *
 * @verify{18844245}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListGetModule
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListGetModule_module_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListGetModule
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListGetModule_module_is_NULL
TEST.BASIS_PATH:2 of 4
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListGetModule_module_is_NULL}
 *
 * @verifyFunction{module arg is null.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList valid instance.
 *  - module set to NULL.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852798}
 *
 * @verify{18844245}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListGetModule_success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListGetModule
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListGetModule_success
TEST.BASIS_PATH:4 of 4 (partial)
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListGetModule_success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields, set module to arbitrary pointer.}
 *
 * @testinput{- attrList valid instance.
 *  - module allocate space for output pointer.}
 *
 * @testbehavior{Function should return LwSciError_Success, module output parameter should be equal to
 * module from objAttrList structure.}
 *
 * @testcase{18852801}
 *
 * @verify{18844245}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListGetModule
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module.module[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module>>[0] = (NULL);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListGetModule.module[0]>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].module ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListSetActualPerm

-- Test Case: TC_001.LwSciSyncCoreAttrListSetActualPerm_AttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListSetActualPerm
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListSetActualPerm_AttrList_is_NULL
TEST.BASIS_PATH:1 of 2
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListSetActualPerm_AttrList_is_NULL}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.
 *  - actualPerm set to valid value.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852810}
 *
 * @verify{18844251}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListSetActualPerm.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListSetActualPerm
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListSetActualPerm_success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListSetActualPerm
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListSetActualPerm_success
TEST.BASIS_PATH:2 of 2 (partial)
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListSetActualPerm_success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize coreAttrListObj global instance, set header, coreAttrList and numCoreAttrList. 
 *
 * Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. }
 *
 * @testinput{- attrList valid instance.
 *  - actualPerm set to valid value.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852813}
 *
 * @verify{18844251}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListSetActualPerm
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListSetActualPerm
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListSetActualPerm.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListSetActualPerm.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].coreAttrList[0].attrs.actualPerm == LwSciSyncAccessPerm_SignalOnly }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListTypeIsCpuSignaler

-- Test Case: TC_001.LwSciSyncCoreAttrListTypeIsCpuSignaler.AttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListTypeIsCpuSignaler.AttrList_is_NULL
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListTypeIsCpuSignaler.AttrList_is_NULL}
 *
 * @verifyFunction{AttrList is NULL. Should fail at argument validation stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to NULL.
 *  - isCpuSignaller allocate space for out boolean.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852816}
 *
 * @verify{18844254}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListTypeIsCpuSignaler_isCpuSignaller_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListTypeIsCpuSignaler_isCpuSignaller_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListTypeIsCpuSignaler_isCpuSignaller_is_NULL}
 *
 * @verifyFunction{isCpuSignaller is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuSignaller set to NULL.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852819}
 *
 * @verify{18844254}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler>>[0] == true }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_1
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_1
TEST.BASIS_PATH:3 of 3 (partial)
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_1}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncCoreAttrListType_CpuSignaller attribute}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuSignaller allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852822}
 *
 * @verify{18844254}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler>>[0] == false }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_2
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_2
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_FALSE_2}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncCoreAttrListType_CpuSignaller attribute}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuSignaller allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852825}
 *
 * @verify{18844254}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler>>[0] == false }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_TRUE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_TRUE
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListTypeIsCpuSignaler_success_TRUE}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncCoreAttrListType_CpuSignaller attribute.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuSignaller allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852828}
 *
 * @verify{18844254}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuSignaler
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler>>[0] == true }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListTypeIsCpuWaiter

-- Test Case: TC_001.LwSciSyncCoreAttrListTypeIsCpuWaiter_AttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListTypeIsCpuWaiter_AttrList_is_NULL
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListTypeIsCpuWaiter_AttrList_is_NULL}
 *
 * @verifyFunction{AttrList is NULL. Should fail at argument validation stage.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to NULL.
 *  - isCpuWaite allocate space for out boolean.}
 *
 * @testbehavior{Function should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18852831}
 *
 * @verify{18844257}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListTypeIsCpuWaiter_Success_TRUE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListTypeIsCpuWaiter_Success_TRUE
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListTypeIsCpuWaiter_Success_TRUE}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncAccessPerm_WaitOnly attribute.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuWaiter allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852834}
 *
 * @verify{18844257}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>>[0] == true }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListTypeIsCpuWaiter_isCpuWaiter_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListTypeIsCpuWaiter_isCpuWaiter_is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListTypeIsCpuWaiter_isCpuWaiter_is_NULL}
 *
 * @verifyFunction{isCpuWaiter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuWaiter set to NULL.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852837}
 *
 * @verify{18844257}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>>[0] == true }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_1
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_1
TEST.BASIS_PATH:3 of 3 (partial)
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_1}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncAccessPerm_WaitOnly attribute.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuSignaller allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852840}
 *
 * @verify{18844257}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_Auto
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>>[0] == false }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_2
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_2
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListTypeIsCpuWaiter_success_FALSE_2}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.
 * Yields true since coreAttrList does contain entry with 
 * LwSciSyncAccessPerm_WaitOnly attribute.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields,
 * set actrualPerm to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testinput{- attrList valid instance.
 *  - isCpuWaiter allocate space for out boolean.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852843}
 *
 * @verify{18844257}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListTypeIsCpuWaiter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );


TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_attribute_core.LwSciSyncCoreAttrListTypeIsCpuWaiter.isCpuWaiter>>[0] == false }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListValidate

-- Test Case: TC_001.LwSciSyncCoreAttrListValidate.InputParamIsNull.Fail
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidate
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListValidate.InputParamIsNull.Fail
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListValidate.InputParamIsNull.Fail}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852846}
 *
 * @verify{18844281}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.attrList:<<null>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListValidate
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListValidate
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListValidate.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListValidate.Success
TEST.BASIS_PATH:4 of 4 (partial)
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListValidate.Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852849}
 *
 * @verify{18844281}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListValidate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>->refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListValidate.header_is_ilwalid
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidate
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListValidate.header_is_ilwalid
TEST.BASIS_PATH:3 of 4 (partial)
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListValidate.header_is_ilwalid}
 *
 * @verifyFunction{input coreAttrListObj instance has invalid header.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set arbitrary invalid value to its header field.}
 *
 * @testinput{- attrList valid instance.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852852}
 *
 * @verify{18844281}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.attrList
<<lwscisync_attribute_core.LwSciSyncCoreAttrListValidate.attrList>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListsLock

-- Test Case: TC_001.LwSciSyncCoreAttrListsLock.InsufficientMemory
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListsLock.InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListsLock.InsufficientMemory}
 *
 * @verifyFunction{inputAttrListArr[] contains null element.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() return NULL.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize global attrList instance, reference coreAttrListObj in its objPtr field.}
 *
 * @testinput{- inputAttrListArr create an array with single element, reference global attrList instance
 * in first element.
 *  - attrListCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18852855}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListsLock.LwSciCommonLock_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListsLock.LwSciCommonLock_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListsLock.LwSciCommonLock_return_FAIL}
 *
 * @verifyFunction{LwSciCommonLock() return status other that LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonLock() return LwSciError_BadParameter.
 *
 * LwSciCommonCalloc() map to calloc() function from CRT.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize global attrList instance, reference coreAttrListObj in its objPtr field.}
 *
 * @testinput{- inputAttrListArr create an array with single element, reference global attrList instance
 * in first element.
 *  - attrListCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852858}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:1
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListsLock.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListsLock.Success
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListsLock.Success}
 *
 * @verifyFunction{Normal exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() map to calloc() function from CRT.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize five global coreAttrListObj instances, set valid header and coreAttrList fields
 * in each instance.
 *
 * Initialize five global attrList instances, reference coreAttrListObj in its objPtr fields.}
 *
 * @testinput{- inputAttrListArr create an array with five elements, reference global attrList instance
 * in each element.
 *  - attrListCount set to 5.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852861}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:5
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[1][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[2][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[3][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[4][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4][0].header = ( ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4] & (~0xFFFFULL)) | 0xCDULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[1].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[2].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[3].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[4].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[1]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[1] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[2]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[2] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[3]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[3] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[4]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[4] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListsLock.Success_attrListCount_is_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListsLock.Success_attrListCount_is_0
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListsLock.Success_attrListCount_is_0}
 *
 * @verifyFunction{inputAttrListArr[] contains null element.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() map to calloc() function from CRT.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize global attrList instance, reference coreAttrListObj in its objPtr field.}
 *
 * @testinput{- inputAttrListArr create an array with single element, reference global attrList instance
 * in first element.
 *  - attrListCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852864}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListsLock.attrList_contains_null_element
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListsLock.attrList_contains_null_element
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListsLock.attrList_contains_null_element}
 *
 * @verifyFunction{inputAttrListArr[] contains null element.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize global attrList instance, reference coreAttrListObj in its objPtr field.}
 *
 * @testinput{- inputAttrListArr create two elements, reference global attrList instance in first element
 * andset second element to null.
 *  - attrListCount set to 2.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852867}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr[1]:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:2
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsLock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreAttrListsLock.inputAttrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsLock
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreAttrListsLock.inputAttrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreAttrListsLock.inputAttrList_is_NULL}
 *
 * @verifyFunction{inputAttrListArr[] is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputAttrListArr set to NULL.
 *  - attrListCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852870}
 *
 * @verify{18844296}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.inputAttrListArr:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.attrListCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsLock.return:LwSciError_BadParameter
TEST.END

-- Subprogram: LwSciSyncCoreAttrListsUnlock

-- Test Case: TC_001.LwSciSyncCoreAttrListsUnlock.LwSciCommonObjUnlock_return_FAIL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsUnlock
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListsUnlock.LwSciCommonObjUnlock_return_FAIL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListsUnlock.LwSciCommonObjUnlock_return_FAIL}
 *
 * @verifyFunction{LwSciCommonUnlock() return status other that LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonUnlock() return LwSciError_BadParameter.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize five global coreAttrListObj instances, set valid header and coreAttrList fields
 * in each instance.
 *
 * Initialize five global attrList instances, reference coreAttrListObj in its objPtr fields.}
 *
 * @testinput{- inputAttrListArr create an array with five elements, reference global attrList instance
 * in each element.
 *  - attrListCount set to 5.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852873}
 *
 * @verify{18844299}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.attrListCount:5
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[1][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[2][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[3][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[4][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[1].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[2].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[3].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[4].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[1]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[1] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[2]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[2] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[3]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[3] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[4]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[4] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListsUnlock.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsUnlock
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListsUnlock.Success
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListsUnlock.Success}
 *
 * @verifyFunction{Normal exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize five global coreAttrListObj instances, set valid header and coreAttrList fields
 * in each instance.
 *
 * Initialize five global attrList instances, reference coreAttrListObj in its objPtr fields.}
 *
 * @testinput{- inputAttrListArr create an array with five elements, reference global attrList instance
 * in each element.
 *  - attrListCount set to 5.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852876}
 *
 * @verify{18844299}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.attrListCount:5
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[1][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[2][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[3][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[4][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[1].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[2].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[3].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[4].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[1]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[1] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[2]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[2] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[3]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[3] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[4]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[4] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListsUnlock.attrListCount_is_0
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsUnlock
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListsUnlock.attrListCount_is_0
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListsUnlock.attrListCount_is_0}
 *
 * @verifyFunction{inputAttrListArr[] contains null element.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() map to calloc() function from CRT.
 *
 * LwSciCommonGetObjFromRef() return ref->objPtr as objPtr output parameter.
 *
 * Initialize global coreAttrListObj instance, set valid header and coreAttrList fields.
 *
 * Initialize global attrList instance, reference coreAttrListObj in its objPtr field.}
 *
 * @testinput{- inputAttrListArr create an array with single element, reference global attrList instance
 * in first element.
 *  - attrListCount set to 0.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852879}
 *
 * @verify{18844299}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[1][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[2][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[3][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[4][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr:<<malloc 5>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.attrListCount:0
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[1][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[2][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[3][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[4][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[1].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[1]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[2].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[2]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[3].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[3]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList.attrList[4].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4].refAttrList.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[4]);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[1]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[1] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[1]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[2]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[2] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[3]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[3]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[3] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[2]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[4]
<<lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[4] = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListsUnlock.inputAttrListCount_is_null
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreAttrListsUnlock
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListsUnlock.inputAttrListCount_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListsUnlock.inputAttrListCount_is_null}
 *
 * @verifyFunction{inputAttrListArr[] is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputAttrListArr set to NULL.
 *  - attrListCount set to 1.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18852882}
 *
 * @verify{18844299}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.inputAttrListArr:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.attrListCount:1
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreAttrListsUnlock.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_core.c.LwSciSyncCoreAttrListsUnlock
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreGetSignalerUseExternalPrimitive

-- Test Case: TC_001.LwSciSyncCoreGetSignalerUseExternalPrimitive.Success
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreGetSignalerUseExternalPrimitive.Success
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreGetSignalerUseExternalPrimitive.Success}
 *
 * @verifyFunction{Main exelwtion scenario, no error branches taken.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Initialize LwSciCommonGetObjFromRef stub, reference global coreAttrListObj instance as its 
 * objPtr output parameter. 
 *
 * Initialize global coreAttrListObj instance, set state field to LwSciSyncCoreAttrListState_Reconciled, 
 * set valid header and coreAttrList fields.}
 *
 * @testinput{- attrList valid instance.
 *  - signalerUserExternalPrimitive set to valid instance.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852885}
 *
 * @verify{18844287}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0]:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj[0][0].numCoreAttrList:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_core.c.LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] );

TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj.coreAttrListObj[0][0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0][0].header = (  ((uint64_t)<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreAttrListObj>>[0] & (~0xFFFFULL)) | 0xCDULL  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
<<lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.attrList>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList_is_NULL}
 *
 * @verifyFunction{attrList is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList set to NULL.
 *  - signalerUserExternalPrimitive allocate memory for a single boolean.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852888}
 *
 * @verify{18844287}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive:<<malloc 1>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUserExternalPrimitive_is_NULL
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUserExternalPrimitive_is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUserExternalPrimitive_is_NULL}
 *
 * @verifyFunction{signalerUserExternalPrimitive is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrList dummy instance.
 *  - signalerUserExternalPrimitive set to null.}
 *
 * @testbehavior{Function should call LwSciCommonPanic() to abort exelwtion.}
 *
 * @testcase{18852891}
 *
 * @verify{18844287}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive:<<null>>
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreGetSignalerUseExternalPrimitive
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreValidateAttrListArray

-- Test Case: TC_001.LwSciSyncCoreValidateAttrListArray.arg_is_NULL_and_allowEmpty_is_TRUE
TEST.UNIT:lwscisync_attribute_core
TEST.SUBPROGRAM:LwSciSyncCoreValidateAttrListArray
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreValidateAttrListArray.arg_is_NULL_and_allowEmpty_is_TRUE
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreValidateAttrListArray.arg_is_NULL_and_allowEmpty_is_TRUE}
 *
 * @verifyFunction{AttrListArray is NULL, allowEmpty is TRUE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- attrListArray set to NULL.
 *  - attrListCount set to 0.
 *  - allowEmpty set to TRUE.}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18852894}
 *
 * @verify{18844293}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreValidateAttrListArray.attrListArray:<<null>>
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreValidateAttrListArray.attrListCount:0
TEST.VALUE:lwscisync_attribute_core.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:lwscisync_attribute_core.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.FLOW
  lwscisync_attribute_core.c.LwSciSyncCoreValidateAttrListArray
  lwscisync_attribute_core.c.LwSciSyncCoreValidateAttrListArray
TEST.END_FLOW
TEST.END
