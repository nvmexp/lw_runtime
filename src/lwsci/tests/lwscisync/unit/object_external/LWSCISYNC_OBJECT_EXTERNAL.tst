-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_OBJECT_EXTERNAL
-- Unit(s) Under Test: lwscisync_object_external
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

-- Subprogram: LwSciSyncAttrListAndObjFreeDesc

-- Test Case: TC_001.LwSciSyncAttrListAndObjFreeDesc.Successful_FreesDescriptor
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListAndObjFreeDesc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListAndObjFreeDesc.Successful_FreesDescriptor
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListAndObjFreeDesc.Successful_FreesDescriptor}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListAndObjFreeDesc() when frees the memory for input attrListAndObjDescBuf.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListAndObjFreeDesc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- attrListAndObjDescBuf is set to desc to be freed.}
 *
 * @testbehavior{- LwSciSyncAttrListAndObjFreeDesc() returns without any error.
 * - Input attrListAndObjDescBuf is freed using LwSciCommon functionality.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852900}
 *
 * @verify{18844728}
 */
TEST.END_NOTES:
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListAndObjFreeDesc
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncAttrListAndObjFreeDesc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf
<<lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf>> = ( &VECTORCAST_INT1  );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListAndObjFreeDesc.NULL_attrListAndObjDescBuf
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListAndObjFreeDesc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListAndObjFreeDesc.NULL_attrListAndObjDescBuf
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListAndObjFreeDesc.NULL_attrListAndObjDescBuf}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListAndObjFreeDesc() for NULL attrListAndObjDescBuf.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListAndObjFreeDesc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- attrListAndObjDescBuf is set to NULL.}
 *
 * @testbehavior{- LwSciSyncAttrListAndObjFreeDesc() returns without any error.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852903}
 *
 * @verify{18844728}
 */
TEST.END_NOTES:
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListAndObjFreeDesc
  lwscisync_object_external.c.LwSciSyncAttrListAndObjFreeDesc
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf
<<lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf
{{ <<lwscisync_object_external.LwSciSyncAttrListAndObjFreeDesc.attrListAndObjDescBuf>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncAttrListReconcileAndObjAlloc

-- Test Case: TC_001.LwSciSyncAttrListReconcileAndObjAlloc.Successful_ReconcileAndAllocObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListReconcileAndObjAlloc.Successful_ReconcileAndAllocObj
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListReconcileAndObjAlloc.Successful_ReconcileAndAllocObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() for successful reconciliation and LwSciSyncObj allocation from input unreconciled LwSciSyncAttrLists.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to array of valid unreconciled LwSciSyncAttrList(s) with attributes values not resulting in actual backend primitive allocation.
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_Success.
 * - syncObj points to new LwSciSyncObj bound to input reconciledList, no actual backend primitive is allocated and other members initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852906}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:1
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciSyncCoreInitPrimitive
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value.value[0]
<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>>[0] = ( &<<lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( &<<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncInternalAttrValPrimitiveType_LowerBound) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncObjRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( *<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj  ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.header
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList != ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive == ( <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>>[0] ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() when failed to allocate memory for LwSciSyncCorePrimitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreInitPrimitive().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852909}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[1]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:1
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.needsAllocation:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value.value[0]
<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>>[0] = ( &<<lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncInternalAttrValPrimitiveType_LowerBound) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<lwscisync_object_external.<<GLOBAL>>.PrimitiveType_var>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncObjRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( *<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.header
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList != ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive == ( <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>>[0] ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_SetRefOnReconciledList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_SetRefOnReconciledList
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_SetRefOnReconciledList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() when failed to set reference on input reconciledList.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreAttrListDup().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852915}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncObjRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.header
{{ <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.header ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocLwSciSyncObjRec
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocLwSciSyncObjRec
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_AllocLwSciSyncObjRec}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() fails due to lack of system resource.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonAllocObjWithRef().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_InsufficientResource.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852918}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_InsufficientResource
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_InsufficientResource
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncObjRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListReconcileAndObjAlloc.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListReconcileAndObjAlloc.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListReconcileAndObjAlloc.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to NULL.
 * - inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852927}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_Reconcile
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_Reconcile
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListReconcileAndObjAlloc.Fail_To_Reconcile}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() failed to reconcile the input unreconciled LwSciSyncAttrLists.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListReconcile().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_ReconciliationFailed.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852930}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 2>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
*<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListReconcileAndObjAlloc.NULL_inputArray
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListReconcileAndObjAlloc.NULL_inputArray
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListReconcileAndObjAlloc.NULL_inputArray}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() for NULL inputArray.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListReconcile().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputArray is set to NULL.
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to non-zero value.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852933}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:2
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList[0]:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:2
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncAttrListReconcileAndObjAlloc.NULL_newConflictList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_011.LwSciSyncAttrListReconcileAndObjAlloc.NULL_newConflictList
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListReconcileAndObjAlloc.NULL_newConflictList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() for NULL newConflictList.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListReconcile().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- newConflictList is set to pointer of LwSciSyncAttrList.
 * - inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - inputCount is set to number of elements in inputArray.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_UnsupportedConfig.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852936}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_UnsupportedConfig
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_UnsupportedConfig
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:1
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList.newReconciledList[0].newReconciledList[0][0].refAttrList.objPtr
<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>>[0][0].refAttrList.objPtr = ( &VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( *<<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncAttrListReconcileAndObjAlloc.Ilwalid_inputCount
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncAttrListReconcileAndObjAlloc
TEST.NEW
TEST.NAME:TC_012.LwSciSyncAttrListReconcileAndObjAlloc.Ilwalid_inputCount
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncAttrListReconcileAndObjAlloc.Ilwalid_inputCount}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncAttrListReconcileAndObjAlloc() for inputCount is 0.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListReconcileAndObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListReconcile().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- inputCount is set to 0.
 * - inputArray is set to array of valid unreconciled LwSciSyncAttrList(s).
 * - syncObj is set to pointer of LwSciSyncObj.
 * - newConflictList is set to pointer of LwSciSyncAttrList.}
 *
 * @testbehavior{- LwSciSyncAttrListReconcileAndObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{18844722}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputCount:0
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputCount:0
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListReconcile
  lwscisync_object_external.c.LwSciSyncAttrListReconcileAndObjAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.inputArray>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newReconciledList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList
{{ <<uut_prototype_stubs.LwSciSyncAttrListReconcile.newConflictList>> == ( <<lwscisync_object_external.LwSciSyncAttrListReconcileAndObjAlloc.newConflictList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncIpcExportAttrListAndObj

-- Test Case: TC_001.LwSciSyncIpcExportAttrListAndObj.Successful_ExportAttributeListAndObject
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_001.LwSciSyncIpcExportAttrListAndObj.Successful_ExportAttributeListAndObject
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncIpcExportAttrListAndObj.Successful_ExportAttributeListAndObject}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() when attribute list and object are exported sucessfully}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_Success.
 * - attrListAndObjDesc points to buffer containing valid key-values in correct sequence.
 * - attrListAndObjDescSize points to size of exported buffer.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852939}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFreeDesc
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
*<<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor1>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj  );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 33 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) + 1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 33 ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor)+1 ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 33 ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == (  &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 6 ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 1 ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncObjIpcExportDescriptor)  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>>[0].refObj  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt==5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
else if(cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize.attrListAndObjDescSize[0]
{{ <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize>>[0] == ( sizeof(LwSciSyncObjIpcExportDescriptor) + 1  ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncIpcExportAttrListAndObj.Fail_To_AllocAttrListAndObjDescMemory
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_002.LwSciSyncIpcExportAttrListAndObj.Fail_To_AllocAttrListAndObjDescMemory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncIpcExportAttrListAndObj.Fail_To_AllocAttrListAndObjDescMemory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() when memory allocation is failed for attrListAndObjDesc.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonCalloc()
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852942}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:6
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncAttrListFreeDesc
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
*<<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj  );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 33 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) + 1  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 33 ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 33 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>>[0].refObj  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt==5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
else if(cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ExportPermLessThanRequested
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_003.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ExportPermLessThanRequested
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ExportPermLessThanRequested}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() where input permissions is less than requested by peer ipcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - permissions set to lower value than requested by peer LwSciIpcEndpoint.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852945}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFreeDesc
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj  );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
*<<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>>[0].refObj  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListFreeDesc.descBuf>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncIpcExportAttrListAndObj.Fail_ToExportReconciledAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_004.LwSciSyncIpcExportAttrListAndObj.Fail_ToExportReconciledAttrList
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncIpcExportAttrListAndObj.Fail_ToExportReconciledAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() when exporting reconciled LwSciSyncAttrList fails due to system lacks resource other than memory.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIpcExportReconciled()
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852948}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
*<<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncIpcExportAttrListAndObj.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_005.LwSciSyncIpcExportAttrListAndObj.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncIpcExportAttrListAndObj.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() for invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852951}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncIpcExportAttrListAndObj.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_006.LwSciSyncIpcExportAttrListAndObj.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncIpcExportAttrListAndObj.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj set to NULL.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852954}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj:<<null>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDescSize
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_007.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDescSize
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDescSize}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() for NULL attrListAndObjDesc.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- attrListAndObjDescSize is set to NULL.
 * - syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852957}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_008.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDesc
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncIpcExportAttrListAndObj.NULL_attrListAndObjDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() for NULL attrListAndObjDesc.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- attrListAndObjDesc set to NULL.
 * - syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{LwSciSyncIpcExportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852960}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ipcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcExportAttrListAndObj
TEST.NEW
TEST.NAME:TC_009.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ipcEndpoint
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncIpcExportAttrListAndObj.Ilwalid_ipcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcExportAttrListAndObj() for invalid ipcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcExportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIpcExportReconciled()
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - ipcEndpoint set to invalid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to pointer of size_t.}
 *
 * @testbehavior{- LwSciSyncIpcExportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{18844725}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDesc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.attrListAndObjDescSize[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled
  lwscisync_object_external.c.LwSciSyncIpcExportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
*<<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.reconciledAttrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descBuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcExportReconciled.descLen>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncIpcExportAttrListAndObj.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncIpcImportAttrListAndObj

-- Test Case: TC_001.LwSciSyncIpcImportAttrListAndObj.Successful_ImportAttributeListAndObject
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_001.LwSciSyncIpcImportAttrListAndObj.Successful_ImportAttributeListAndObject
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncIpcImportAttrListAndObj.Successful_ImportAttributeListAndObject}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() when attribute list and object are imported sucessfully}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_Success.
 * - LwSciSyncObj is associated with LwSciSyncAttrList and other members are initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852963}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.Param64value:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.len:8
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor1>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> = ( memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
else if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm);
}
if(3==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_ModuleCnt );
}
if(4==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_IpcEndpoint );
}
if(5==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_CorePrimitive );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(3==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(4==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(5==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
else if(3==count7)
{
*(uint64_t *)(*(uint64_t**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 3 );
}
else if(4==count7)
{
*(LwSciIpcEndpoint *)(*(LwSciIpcEndpoint**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 4 );
}
else if(5==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT2 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(3==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(4==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(5==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor1>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == (  &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor1>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
else
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncCoreObjKey) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
else
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor1>>  ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( &VECTORCAST_INT1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;

if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
else if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data>> == ( &VECTORCAST_INT2 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc
<<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj
{{ *<<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocMemForLocalDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_002.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocMemForLocalDesc
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocMemForLocalDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() - failed to allocate memory for local desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonCalloc()
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncIpcImportAttrListAndObj.NULL_attrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_003.LwSciSyncIpcImportAttrListAndObj.NULL_attrList
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncIpcImportAttrListAndObj.NULL_attrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() for NULL attrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIpcImportReconciled().}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to NULL.
 * - attrListCount is set to non-zero value.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852969}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:1
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncIpcImportAttrListAndObj.Ilwalid_ipcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_004.LwSciSyncIpcImportAttrListAndObj.Ilwalid_ipcEndpoint
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncIpcImportAttrListAndObj.Ilwalid_ipcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() when ipcEndpoint is not a valid LwSciIpcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciIpcGetEndpointInfo().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to invalid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852972}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:1
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncIpcImportAttrListAndObj.Ilwalid_minPermissions
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_005.LwSciSyncIpcImportAttrListAndObj.Ilwalid_minPermissions
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncIpcImportAttrListAndObj.Ilwalid_minPermissions}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() when permissions is invalid.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_SignalOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852975}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:1
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncIpcImportAttrListAndObj.Ilwalid_descriptorSize
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_007.LwSciSyncIpcImportAttrListAndObj.Ilwalid_descriptorSize
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncIpcImportAttrListAndObj.Ilwalid_descriptorSize}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() when input desc size is invalid.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to invalid value.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852981}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciSyncIpcImportAttrListAndObj.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_008.LwSciSyncIpcImportAttrListAndObj.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncIpcImportAttrListAndObj.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- syncObj set to NULL.
 * - module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 *
 * @testbehavior{LwSciSyncIpcImportAttrListAndObj() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852984}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncIpcImportAttrListAndObj.NULL_attrListAndObjDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_009.LwSciSyncIpcImportAttrListAndObj.NULL_attrListAndObjDesc
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncIpcImportAttrListAndObj.NULL_attrListAndObjDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() for NULL attrListAndObjDesc.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testsetup{None.}
 *
 * @testinput{- attrListAndObjDesc is set to NULL.
 * - module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{Returns LwSciError_BadParameter when LwSciSync object based on the supplied binary descriptor is not created.}
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852987}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.END

-- Test Case: TC_010.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocSystemResource
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncIpcImportAttrListAndObj
TEST.NEW
TEST.NAME:TC_010.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocSystemResource
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncIpcImportAttrListAndObj.Fail_To_AllocSystemResource}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncIpcImportAttrListAndObj() when system lacks resource other than memory.}
 *
 * @testpurpose{Unit testing of LwSciSyncIpcImportAttrListAndObj().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIpcImportReconciled()
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- module is set to valid LwSciSyncModule.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - attrListAndObjDesc is set to pointer to void*.
 * - attrListAndObjDescSize is set to size of attrListAndObjDesc.
 * - attrList is set to array to valid unreconciled LwSciSyncAttrLists.
 * - attrListCount is set to number of entries in attrList array.
 * - minPermissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs is set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncIpcImportAttrListAndObj() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{18844731}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.module:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDescSize:1025
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList[0]:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListCount:6
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.minPermissions:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.return:LwSciError_ResourceError
TEST.EXPECTED:lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.ipcEndpoint:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descLen:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListCount:6
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
  uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_object_external.c.LwSciSyncIpcImportAttrListAndObj
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.descBuf>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrListAndObjDesc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.inputUnreconciledAttrListArray>> == ( <<lwscisync_object_external.LwSciSyncIpcImportAttrListAndObj.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIpcImportReconciled.importedReconciledAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjAlloc

-- Test Case: TC_001.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithExternalPrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithExternalPrimitive
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithExternalPrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() for successful allocation and initializing of LwSciSyncObj with external primitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList with attributes values not resulting in actual backend primitive allocation.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_Success.
 * - syncObj points to new LwSciSyncObj bound to input reconciledList, no actual backend primitive is allocated and other members initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852990}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:12
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciSyncCoreInitPrimitive
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> = ( 1234 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncObjAlloc.Fail_To_AllocMemForLwSciSyncCorePrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when failed to allocate memory for LwSciSyncCorePrimitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreInitPrimitive().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_InsufficientMemory .
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852993}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:12
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> = ( 1234 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjAlloc.Fail_To_GetCntrValueFromModule
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjAlloc.Fail_To_GetCntrValueFromModule
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjAlloc.Fail_To_GetCntrValueFromModule}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when failed to get counter value from module.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreModuleCntrGetNextValue().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18852996}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Overflow
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:0
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncObjAlloc.Fail_To_SetRefOnReconciledList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncObjAlloc.Fail_To_SetRefOnReconciledList
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncObjAlloc.Fail_To_SetRefOnReconciledList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when failed to set reference on input reconciledList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreAttrListDup().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853002}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncObjAlloc.Fail_To_AllocLwSciSyncObjRec
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_009.LwSciSyncObjAlloc.Fail_To_AllocLwSciSyncObjRec
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncObjAlloc.Fail_To_AllocLwSciSyncObjRec}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when LwSciCommonAllocObjWithRef() fails due to lack of system resource.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
  * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonAllocObjWithRef().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853014}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_ResourceError
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_ResourceError
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncObjAlloc.Fail_To_AllocWithUnreconciledAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_010.LwSciSyncObjAlloc.Fail_To_AllocWithUnreconciledAttrList
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncObjAlloc.Fail_To_AllocWithUnreconciledAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when input LwSciSyncAttrList is unreconciled.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to unreconciled LwSciSyncAttrList.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853017}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncObjAlloc.NULL_reconciledList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_011.LwSciSyncObjAlloc.NULL_reconciledList
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncObjAlloc.NULL_reconciledList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() for NULL reconciledList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIsReconciled().}
 *
 * @testinput{- reconciledList is set to NULL.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853020}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncObjAlloc.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_012.LwSciSyncObjAlloc.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncObjAlloc.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList.
 * - syncObj is set to NULL.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853023}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.END

-- Test Case: TC_013.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithoutExternalPrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjAlloc
TEST.NEW
TEST.NAME:TC_013.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithoutExternalPrimitive
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncObjAlloc.Successful_AllocAndInitLwSciSyncObjWithoutExternalPrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjAlloc() when allocates and initializes LwSciSyncObj with out external primitive attribute.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjAlloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- reconciledList is set to reconciled LwSciSyncAttrList with attributes values resulting in actual backend primitive allocation.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjAlloc() returns LwSciError_Success.
 * - syncObj points to new LwSciSyncObj bound to input reconciledList, actual backend primitive is done and other members initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853026}
 *
 * @verify{18844701}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.reconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:12
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.needsAllocation:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjAlloc
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreAttrListDup
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciSyncCoreInitPrimitive
  lwscisync_object_external.c.LwSciSyncObjAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> = ( 1234 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreInitPrimitive.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreObj) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciRef) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListDup.dupAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjAlloc.reconciledList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleCntrGetNextValue.module>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0][0].refModule ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjAlloc.syncObj.syncObj[0]
{{ <<lwscisync_object_external.LwSciSyncObjAlloc.syncObj>>[0] == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjDup

-- Test Case: TC_001.LwSciSyncObjDup.Successful_DuplicatedLwSciSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjDup.Successful_DuplicatedLwSciSyncObj
TEST.BASIS_PATH:4 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjDup.Successful_DuplicatedLwSciSyncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() Successfully creates a new LwSciSyncObj holding a reference to the original LwSciSyncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjDup() returns LwSciError_Success.
 * - LwSciCommon functionality is called to create new reference on input syncObj.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853029}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef[0][0].magicNumber:10
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef[0][0].size:20
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef[0][0].refCount:30
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef[0][0].objPtr:VECTORCAST_INT1
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.dupObj[0][0].refObj.magicNumber:10
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.dupObj[0][0].refObj.size:20
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.dupObj[0][0].refObj.refCount:30
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscisync_object_external.LwSciSyncObjDup.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjDup.Fail_To_DuplicateRefDueToInsuffientAvailableRefs
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjDup.Fail_To_DuplicateRefDueToInsuffientAvailableRefs
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncObjDup.Fail_To_DuplicateRefDueToInsuffientAvailableRefs}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for failure to create reference when number of references
 *  to the input LwSciSyncObj is INT32_MAX and the newly duplicated LwSciSyncObj tries to take one more reference.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonDuplicateRef().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj with INT32_MAX refCount.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjDup() returns LwSciError_IlwalidState.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853032}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.refCount:<<MAX>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_IlwalidState
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscisync_object_external.LwSciSyncObjDup.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjDup.dupObj.dupObj[0][0].refObj.objPtr
{{ <<lwscisync_object_external.LwSciSyncObjDup.dupObj>>[0][0].refObj.objPtr == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjDup.Fail_To_DuplicateRefDueToResourceError
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjDup.Fail_To_DuplicateRefDueToResourceError
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjDup.Fail_To_DuplicateRefDueToResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for failure to create reference due to lack of system resource other than memory.}
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonDuplicateRef().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjDup() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853035}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_ResourceError
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscisync_object_external.LwSciSyncObjDup.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjDup.Fail_To_DuplicateRefDueToMemAllocFailure
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjDup.Fail_To_DuplicateRefDueToMemAllocFailure
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncObjDup.Fail_To_DuplicateRefDueToMemAllocFailure}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for failure to create reference due to insufficient memory.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonDuplicateRef().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjDup() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853038}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscisync_object_external.LwSciSyncObjDup.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncObjDup.NULL_dupObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_005.LwSciSyncObjDup.NULL_dupObj
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncObjDup.NULL_dupObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for NULL dupObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * dupObj is set to NULL.}
 *
 * @testbehavior{- LwSciSyncObjDup() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853041}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncObjDup.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_006.LwSciSyncObjDup.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncObjDup.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreObjValidate().
 * - All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to NULL.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{LwSciSyncCoreObjValidate() is called to check for the validity of 'LwSciSync' object associated with 'syncObj' pointer.
 * LwSciCommonDuplicateRef() is called to create a new reference to underlying object.
 * Returns LwSciError_Success when valid LwSciSyncObj is duplicated sucessfully.}
  * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853044}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjDup.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_object_external.c.LwSciSyncObjDup
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncObjDup.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjDup
TEST.NEW
TEST.NAME:TC_007.LwSciSyncObjDup.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncObjDup.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjDup() for invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjDup().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncCoreObjValidate().}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * dupObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjDup() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853047}
 *
 * @verify{18844704}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjDup.dupObj:<<malloc 1>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjDup
  uut_prototype_stubs.LwSciSyncCoreObjValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjDup.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjFree

-- Test Case: TC_001.LwSciSyncObjFree.Successful_freesLwSciSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjFree.Successful_freesLwSciSyncObj
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjFree.Successful_freesLwSciSyncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjFree() Successful when destroys a valid LwSciSyncObj and frees any resources that were allocated for it.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjFree().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjFree() returns LwSciError_Success.
 * - Successfully destroys a valid LwSciSyncObj and frees resources that were allocated for it.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853050}
 *
 * @verify{18844707}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjFree.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjFree.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjFree
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  lwscisync_object_external.c.LwSciSyncObjFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjFree.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjFree.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjFree.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjFree.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncObjFree.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjFree() for Invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjFree().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncCoreObjValidate().}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjFree() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853053}
 *
 * @verify{18844707}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjFree.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjFree.syncObj[0].refObj.objPtr:<<null>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjFree
  uut_prototype_stubs.LwSciSyncCoreObjValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjFree.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjFree.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjFree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjFree.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjFree.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjFree() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjFree().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreObjValidate().}
 *
 * @testinput{- syncObj is set to NULL.}
 *
 * @testbehavior{LwSciSyncObjFree() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853056}
 *
 * @verify{18844707}
 */
TEST.END_NOTES:
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjFree
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_object_external.c.LwSciSyncObjFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjFree.syncObj
<<lwscisync_object_external.LwSciSyncObjFree.syncObj>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjGenerateFence

-- Test Case: TC_001.LwSciSyncObjGenerateFence.Successful_FenceGeneration
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjGenerateFence.Successful_FenceGeneration
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjGenerateFence.Successful_FenceGeneration}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() when generates next point on sync timeline of an LwSciSyncObj and fills
 * in the supplied LwSciSyncFence object.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj set to valid LwSciSyncObj with primitive allocated by LwSciSync and has CPU access perm.
 * - syncFence set to LwSciSyncFence pointer.}
 *
 * @testbehavior{- LwSciSyncObjGenerateFence() Returns LwSciError_Success.
 * - syncFence members are filled correctly.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853059}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncFenceUpdateFence.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT1
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveGetNewFence.return:10
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveGetId.return:11
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncFenceUpdateFence.id:11
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncFenceUpdateFence.value:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCorePrimitiveGetNewFence
  uut_prototype_stubs.LwSciSyncCorePrimitiveGetId
  uut_prototype_stubs.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciSyncFenceUpdateFence
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncFenceUpdateFence.syncObj
{{ <<uut_prototype_stubs.LwSciSyncFenceUpdateFence.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncFenceUpdateFence.syncFence
{{ <<uut_prototype_stubs.LwSciSyncFenceUpdateFence.syncFence>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveGetNewFence.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveGetNewFence.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveGetId.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveGetId.primitive>> == ( <<uut_prototype_stubs.LwSciSyncCorePrimitiveGetNewFence.primitive>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithExternalPrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithExternalPrimitive
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithExternalPrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() when syncObj is backed by external primitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj set to valid LwSciSyncObj backed by external primitive.
 * - syncFence set to LwSciSyncFence pointer.}
 *
 * @testbehavior{- LwSciSyncObjGenerateFence() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853065}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT2
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithoutCpuPerm
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithoutCpuPerm
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncObjGenerateFence.Ilwalid_syncObjWithoutCpuPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() attribute list does not have CPU signaler permissions.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj set to valid LwSciSyncObj without CPU access permission.
 * - syncFence set to LwSciSyncFence pointer.}
 *
 * @testbehavior{- LwSciSyncObjGenerateFence() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853068}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT2
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:false
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncObjGenerateFence.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_006.LwSciSyncObjGenerateFence.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncObjGenerateFence.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() for invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testsetup{None.}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * - syncFence set to LwSciSyncFence pointer.}
 *
 * @testbehavior{- LwSciSyncObjGenerateFence() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853074}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  uut_prototype_stubs.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncObjGenerateFence.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_007.LwSciSyncObjGenerateFence.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncObjGenerateFence.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to NULL.
 * - syncFence set to LwSciSyncFence pointer.}
 *
 * @testbehavior{- LwSciSyncObjGetAttrList() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853077}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncObjGenerateFence.NULL_syncFence
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGenerateFence
TEST.NEW
TEST.NAME:TC_008.LwSciSyncObjGenerateFence.NULL_syncFence
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncObjGenerateFence.NULL_syncFence}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGenerateFence() syncFence is NULL.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGenerateFence().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testsetup{None}
 *
 * @testinput{- syncFence set to NULL.
 * - syncObj set to valid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjGenerateFence() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853080}
 *
 * @verify{18844716}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.syncFence:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGenerateFence.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
  lwscisync_object_external.c.LwSciSyncObjGenerateFence
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncObjGetNumPrimitives

-- Test Case: TC_001.LwSciSyncObjGetNumPrimitives.Successful_RetrievalOfPrimitiveCount
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetNumPrimitives
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjGetNumPrimitives.Successful_RetrievalOfPrimitiveCount
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjGetNumPrimitives.Successful_RetrievalOfPrimitiveCount}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetNumPrimitives() for successful retrieval of number of primitives in LwSciSyncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetNumPrimitives().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{syncObj is set to a valid LwSciSyncObj.
 * numPrimitives is points to memory holding the output.}
 *
 * @testbehavior{- LwSciSyncObjGetNumPrimitives() returns LwSciError_Success.
 * - numPrimitives returns with primitives count associated with LwSciSyncObj.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853083}
 *
 * @verify{18844734}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives[0]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives[0]:1
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveCount
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjGetNumPrimitives.NULL_numPrimitives
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetNumPrimitives
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjGetNumPrimitives.NULL_numPrimitives
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjGetNumPrimitives.NULL_numPrimitives}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetNumPrimitives() for NULL numPrimitives.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetNumPrimitives().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- numPrimitives is set to NULL.
 * - syncObj is set to a valid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjGetNumPrimitives() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853089}
 *
 * @verify{18844734}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncObjGetNumPrimitives.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetNumPrimitives
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjGetNumPrimitives.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncObjGetNumPrimitives.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetNumPrimitives() syncObj is NULL}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetNumPrimitives().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to NULL.
 * - numPrimitives is points to memory holding the output.}
 *
 * @testbehavior{- LwSciSyncObjGetNumPrimitives() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853092}
 *
 * @verify{18844734}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives[0]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives[0]:0
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj:<<null>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncObjGetNumPrimitives.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetNumPrimitives
TEST.NEW
TEST.NAME:TC_005.LwSciSyncObjGetNumPrimitives.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncObjGetNumPrimitives.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetNumPrimitives() for invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetNumPrimitives().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * - numPrimitives is points to memory holding the output.}
 *
 * @testbehavior{- LwSciSyncObjGetNumPrimitives() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853095}
 *
 * @verify{18844734}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.numPrimitives[0]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetNumPrimitives.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetNumPrimitives
  uut_prototype_stubs.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGetNumPrimitives.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjGetPrimitiveType

-- Test Case: TC_001.LwSciSyncObjGetPrimitiveType.Successful_RetrievalOfPrimitiveType
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetPrimitiveType
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjGetPrimitiveType.Successful_RetrievalOfPrimitiveType
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjGetPrimitiveType.Successful_RetrievalOfPrimitiveType}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetPrimitiveType() for successful retrieval of primitive type from input LwSciSyncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetPrimitiveType().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - primitiveType is set to LwSciSyncInternalAttrValPrimitiveType pointer.}
 *
 * @testbehavior{- LwSciSyncObjGetPrimitiveType() returns LwSciError_Success.
 * - primitiveType returns the LwSciSyncInternalAttrValPrimitiveType associated with LwSciSyncObj.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853098}
 *
 * @verify{18844737}
 */
TEST.END_NOTES:
TEST.VALUE:<<OPTIONS>>.REFERENCED_GLOBALS:TRUE
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT3:MACRO=LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj[0].refObj.objPtr:VECTORCAST_INT2
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len[0]:1
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.key:LwSciSyncInternalAttrKey_SignalerPrimitiveInfo
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value.value[0]
<<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>>[0] = (  &VECTORCAST_INT3 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetSingleInternalAttr.len>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncInternalAttrValPrimitiveType) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &VECTORCAST_INT3  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjGetPrimitiveType.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetPrimitiveType
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjGetPrimitiveType.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjGetPrimitiveType.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetPrimitiveType() for invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetPrimitiveType().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * - primitiveType is set to LwSciSyncInternalAttrValPrimitiveType pointer.}
 *
 * @testbehavior{- LwSciSyncObjGetPrimitiveType() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853104}
 *
 * @verify{18844737}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_Unknown
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
  uut_prototype_stubs.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjGetPrimitiveType.NULL_primitiveType
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetPrimitiveType
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjGetPrimitiveType.NULL_primitiveType
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncObjGetPrimitiveType.NULL_primitiveType}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetPrimitiveType() for NULL primitiveType.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetPrimitiveType().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testsetup{None}
 *
 * @testinput{- primitiveType is set to NULL.
 * - syncObj is set to a valid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjGetPrimitiveType() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853107}
 *
 * @verify{18844737}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncObjGetPrimitiveType.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjGetPrimitiveType
TEST.NEW
TEST.NAME:TC_005.LwSciSyncObjGetPrimitiveType.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncObjGetPrimitiveType.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjGetPrimitiveType() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjGetPrimitiveType().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{- syncObj is set to NULL.
 * - primitiveType is set to LwSciSyncInternalAttrValPrimitiveType pointer.}
 *
 * @testbehavior{- LwSciSyncObjGetPrimitiveType() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853110}
 *
 * @verify{18844737}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.primitiveType:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjGetPrimitiveType.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj:<<null>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  lwscisync_object_external.c.LwSciSyncObjGetPrimitiveType
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjIpcExport

-- Test Case: TC_001.LwSciSyncObjIpcExport.Successful_ExportNewlyAllocatedSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjIpcExport.Successful_ExportNewlyAllocatedSyncObj
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjIpcExport.Successful_ExportNewlyAllocatedSyncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() exporting newly allocated LwSciSyncObj (i.e. exporting the syncObj for the first time).}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj which is never exported earlier.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_Success.
 * - desc points to buffer containing valid key-values in correct sequence.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853113}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:40
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:6
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1>> ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
else if(5== cnt)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcExport.desc
{{ <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> == ( <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncObjIpcExport.Fail_To_AppendSyncObjKeyInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_005.LwSciSyncObjIpcExport.Fail_To_AppendSyncObjKeyInDesc
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncObjIpcExport.Fail_To_AppendSyncObjKeyInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to append LwSciSyncObj key when no space is left in transport buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for append LwSciSyncObj key:
 * - LwSciCommonTransportAppendKeyValuePair()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_NoSpace.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853125}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[0..2]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[10]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[100]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(3==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(4==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(5==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
else if(5== cnt)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncObjIpcExport.Fail_To_AllocMemForDescBuffer
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_008.LwSciSyncObjIpcExport.Fail_To_AllocMemForDescBuffer
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncObjIpcExport.Fail_To_AllocMemForDescBuffer}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to allocate memory for desc buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for failed to allocate memory for desc buffer:
 * - LwSciCommonTransportAllocTxBufferForKeys().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853134}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
static ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_InsufficientMemory );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncObjIpcExport.Fail_To_AppendCorePrimitiveKeyInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_009.LwSciSyncObjIpcExport.Fail_To_AppendCorePrimitiveKeyInDesc
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncObjIpcExport.Fail_To_AppendCorePrimitiveKeyInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to append CorePrimitive key when no space is left in transport buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for append CorePrimitive key:
 * - LwSciCommonTransportAppendKeyValuePair()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_NoSpace.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853137}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(3==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(4==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncObjIpcExport.Fail_To_AppendIpcEndpointKeyInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_010.LwSciSyncObjIpcExport.Fail_To_AppendIpcEndpointKeyInDesc
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncObjIpcExport.Fail_To_AppendIpcEndpointKeyInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to append IpcEndpoint key when no space is left in transport buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for append IpcEndpoint key:
 * - LwSciCommonTransportAppendKeyValuePair()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_NoSpace.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853140}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(3==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=3)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncObjIpcExport.Fail_To_AppendModulecntKeyInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_011.LwSciSyncObjIpcExport.Fail_To_AppendModulecntKeyInDesc
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncObjIpcExport.Fail_To_AppendModulecntKeyInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to append ModuleCnt key when no space is left in transport buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for append ModuleCnt key:
 * - LwSciCommonTransportAppendKeyValuePair()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_NoSpace.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853143}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=3)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncObjIpcExport.Fail_To_AppendAccessPermKeyInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_012.LwSciSyncObjIpcExport.Fail_To_AppendAccessPermKeyInDesc
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncObjIpcExport.Fail_To_AppendAccessPermKeyInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when LwSciCommonTransportAppendKeyValuePair() fails to append AccessPerm key when no space is left in transport buffer.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonTransportAppendKeyValuePair()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_NoSpace.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853146}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[0..2]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[10]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[100]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc[0].payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=3)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncObjIpcExport.Successful_ExportAnImportedSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_013.LwSciSyncObjIpcExport.Successful_ExportAnImportedSyncObj
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncObjIpcExport.Successful_ExportAnImportedSyncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when exporting an imported LwSciSyncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid imported LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_Success.
 * - desc points to buffer containing valid key-values in correct sequence.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testcase{18853149}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:6
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 33 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 33 ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor)  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 33 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
else if(5== cnt)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncObjIpcExport.Ilwalid_InputPermLessThanPeerPermission
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_014.LwSciSyncObjIpcExport.Ilwalid_InputPermLessThanPeerPermission
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncObjIpcExport.Ilwalid_InputPermLessThanPeerPermission}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() where input permissions is less than requested by peer ipcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj bound to LwSciSyncAttrList which has higher peer perm for input ipcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853152}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncObjIpcExport.Successful_ExportSyncObjFromDescWithAccessPermAuto
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_015.LwSciSyncObjIpcExport.Successful_ExportSyncObjFromDescWithAccessPermAuto
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncObjIpcExport.Successful_ExportSyncObjFromDescWithAccessPermAuto}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when export LwSciSyncObj with LwSciSyncAccessPerm_Auto.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_Auto.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_Success.
 * - Exports an LwSciSyncObj into an LwSciIpc-transferable object binary descriptor.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853155}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:6
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 33 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 33 ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor)  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 33 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
else if(5== cnt)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_CorePrimitive ) }}
else if(5== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (5==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &VECTORCAST_INT1 ) }}
else if (5==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcExport.permissions>> = ( uint64_t )LwSciSyncAccessPerm_Auto ;
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcExport.desc
{{ <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> == ( <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncObjIpcExport.Fail_To_GetPermForInputIpcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_016.LwSciSyncObjIpcExport.Fail_To_GetPermForInputIpcEndpoint
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncObjIpcExport.Fail_To_GetPermForInputIpcEndpoint}}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when there are no valid permissions for given ipcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreAttrListGetIpcExportPerm()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to LwSciIpcEndpoint for which LwSciSyncObj doesn't have valid permissions.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853158}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncObjIpcExport.Successful_ExportOfSyncObjWithExternalPrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_017.LwSciSyncObjIpcExport.Successful_ExportOfSyncObjWithExternalPrimitive
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncObjIpcExport.Successful_ExportOfSyncObjWithExternalPrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() - LwSciSyncCorePrimitiveExport() when export LwSciSyncObj with external primitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj with external primitive.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_Success.
 * - desc points to buffer containing valid key-values in correct sequence.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853161}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:0
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:6
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.ipcEndpoint:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
static int count8=0;
count8++;

if (1==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 32 );
}
else if (2==count8)
{
 *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 33 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
static int count9=0;
count9++;

if (1==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 5 );
}
else if (2==count9)
{
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> = ( 6 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &VECTORCAST_INT1 ) }}
else if (3==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 33 ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncObjIpcExportDescriptor)  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( 33 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm.actualPerm>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
static int count2=0;
count2++;

if(count2 ==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 24 ) }}
else if(count2 == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
static int cnt=0;
cnt++;
if(cnt>=1 && cnt<=4)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( NULL ) }}
else if(5== cnt)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key
static int count5=0;
count5++;

if(1== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_AccessPerm ) }}
else if(2== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_ModuleCnt ) }}
else if(3== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreObjKey_IpcEndpoint ) }}
else if(4== count5)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key>> == ( LwSciSyncCoreDescKey_SyncObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int count6=0;
count6++;

if (1==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (2==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (3==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 8 ) }}
else if (4==count6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( 5 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int count7=0;
count7++;

if (1==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 1 ) }}
else if (2==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 21 ) }}
else if (3==count7)
{{ *(uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 10 ) }}
else if (4==count7)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( 32 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcExport.desc
{{ <<lwscisync_object_external.LwSciSyncObjIpcExport.desc>> == ( <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_018.LwSciSyncObjIpcExport.Fail_To_AllocMemFortransportBuf
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_018.LwSciSyncObjIpcExport.Fail_To_AllocMemFortransportBuf
TEST.NOTES:
/**
 * @testname{TC_018.LwSciSyncObjIpcExport.Fail_To_AllocMemFortransportBuf}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when fail to allocate memory for transport buf.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonTransportAllocTxBufferForKeys().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853164}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize>> == ( 32 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncObjIpcExport.Overflow_TotalValueSize
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_019.LwSciSyncObjIpcExport.Overflow_TotalValueSize
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncObjIpcExport.Overflow_TotalValueSize}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when overflow in subtraction of primitiveExportSize from total desc size.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_Success.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853167}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:4294967295
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonPanic
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciSyncCoreAttrListGetIpcExportPerm
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_021.LwSciSyncObjIpcExport.Fail_To_CreatePrimitiveExportDescriptor
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_021.LwSciSyncObjIpcExport.Fail_To_CreatePrimitiveExportDescriptor
TEST.NOTES:
/**
 * @testname{TC_021.LwSciSyncObjIpcExport.Fail_To_CreatePrimitiveExportDescriptor}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() when not enough memory to create export descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCorePrimitiveExport()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_InsufficientMemory.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853173}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.moduleCntr:21
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj.objId.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.ipcEndpoint:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
*<<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> = ( &VECTORCAST_INT1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count11=0;
count11++;

if (1==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (2==count11)
{{ (int *)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (3==count11)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if (4==count11)
{{ (uint64_t*)<<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.data>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveExport.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncObjIpcExport.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_023.LwSciSyncObjIpcExport.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncObjIpcExport.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() faor invalid syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncCoreObjValidate()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to invalid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853179}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjIpcExport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncObjIpcExport.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_024.LwSciSyncObjIpcExport.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncObjIpcExport.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport()- for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncCoreObjValidate()
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- syncObj is set to NULL.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853182}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:10
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:10
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncCoreObjValidate
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_025.LwSciSyncObjIpcExport.Ilwalid_IpcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_025.LwSciSyncObjIpcExport.Ilwalid_IpcEndpoint
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncObjIpcExport.Ilwalid_IpcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() - for invalid ipcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciIpcGetEndpointInfo().}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to invalid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853185}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncObjIpcExport.Ilwalid_Permissions
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_026.LwSciSyncObjIpcExport.Ilwalid_Permissions
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncObjIpcExport.Ilwalid_Permissions}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() for invalid permissions.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - permissions set to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853188}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcExport
  lwscisync_object_external.c.LwSciSyncObjIpcExport
TEST.END_FLOW
TEST.END

-- Test Case: TC_027.LwSciSyncObjIpcExport.NULL_Desc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcExport
TEST.NEW
TEST.NAME:TC_027.LwSciSyncObjIpcExport.NULL_Desc
TEST.NOTES:
/**
 * @testname{TC_027.LwSciSyncObjIpcExport.NULL_Desc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcExport() for NULL desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcExport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{desc is NULL.
 * - syncObj is set to valid LwSciSyncObj.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.}
 *
 * @testbehavior{- LwSciSyncObjIpcExport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853191}
 *
 * @verify{18844710}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.desc:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcExport.return:LwSciError_BadParameter
TEST.END

-- Subprogram: LwSciSyncObjIpcImport

-- Test Case: TC_001.LwSciSyncObjIpcImport.Successfully_ImportSyncObjFromDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjIpcImport.Successfully_ImportSyncObjFromDesc
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjIpcImport.Successfully_ImportSyncObjFromDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when LwSciSyncObj based on supplied binary descriptor is created successfully.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_WaitOnly.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_Success.
 * - LwSciSyncObj is associated with inputAttrList and other members are initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853194}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[1]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[1]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,(4)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,(4)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.len:8
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
else if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}
else if(3==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_ModuleCnt );
}
else if(4==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_IpcEndpoint );
}
else if(5==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_CorePrimitive );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(3==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(4==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(5==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
else if(3==count7)
{
*(uint64_t *)(*(uint64_t**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 3 );
}
else if(4==count7)
{
*(LwSciIpcEndpoint *)(*(LwSciIpcEndpoint**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 4 );
}
else if(5==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT2 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(3==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(4==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(5==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data>> == ( &VECTORCAST_INT2 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj
{{ *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjIpcImport.Successful_ImportWithAutoPerm
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjIpcImport.Successful_ImportWithAutoPerm
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncObjIpcImport.Successful_ImportWithAutoPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - successful import of LwSciSyncObj when permissions is set to LwSciSyncAccessPerm_Auto.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_Success.
 * - LwSciSyncObj is associated with inputAttrList and other members are initialized as per SWUD.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853197}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList[0].refAttrList.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.len[0]:5
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_ActualPerm
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,(4)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,(4)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_Auto
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.len:8
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
else if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}
else if(3==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_ModuleCnt );
}
else if(4==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_IpcEndpoint );
}
else if(5==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_CorePrimitive );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(3==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(4==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(5==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
else if(3==count7)
{
*(uint64_t *)(*(uint64_t**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 3 );
}
else if(4==count7)
{
*(LwSciIpcEndpoint *)(*(LwSciIpcEndpoint**)<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>) = ( 4 );
}
else if(5==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT2 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(3==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(4==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(5==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data>> == ( &VECTORCAST_INT2 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj
{{ *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncObjIpcImport.Ilwalid_DescFromIncompatibleLibraryVersion
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_006.LwSciSyncObjIpcImport.Ilwalid_DescFromIncompatibleLibraryVersion
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncObjIpcImport.Ilwalid_DescFromIncompatibleLibraryVersion}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - when input LwSciSyncObjIpcExportDescriptor is from incompatible library version.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor received from incompatible library version.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853209}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList[0].refAttrList.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934591 );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj
{{ *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncObjIpcImport.Ilwalid_lengthOfModuleCnt
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_007.LwSciSyncObjIpcImport.Ilwalid_lengthOfModuleCnt
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncObjIpcImport.Ilwalid_lengthOfModuleCnt}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - Invalid length of moduleCntr in obj descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with length of moduleCnt set to invalid value.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() Returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853212}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_ModuleCnt );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncObjIpcImport.Ilwalid_lengthOfIpcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_008.LwSciSyncObjIpcImport.Ilwalid_lengthOfIpcEndpoint
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncObjIpcImport.Ilwalid_lengthOfIpcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - Invalid length of IpcEndpoint in obj descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with length of ipcEndpoint set to invalid value.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853215}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_IpcEndpoint );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncObjIpcImport.Ilwalid_lengthOfCorePrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_009.LwSciSyncObjIpcImport.Ilwalid_lengthOfCorePrimitive
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncObjIpcImport.Ilwalid_lengthOfCorePrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - Invalid length of CorePrimitive in obj descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with length of corePrimitive set to invalid value.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853218}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.ipcEndpoint:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.len:5
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
else if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_CorePrimitive );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == (  &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.reconciledList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.data>> == ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive
{{ <<uut_prototype_stubs.LwSciSyncCorePrimitiveImport.primitive>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncObjIpcImport.Failure_DueToUrecognizedTag
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_010.LwSciSyncObjIpcImport.Failure_DueToUrecognizedTag
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncObjIpcImport.Failure_DueToUrecognizedTag}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - when there is unrecognized tag in object descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor containing unrecognized tags.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853221}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
else if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_CoreTimestamps );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncObjIpcImport.Ilwalid_RequestedPermHigherThanGrantedPerm
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_011.LwSciSyncObjIpcImport.Ilwalid_RequestedPermHigherThanGrantedPerm
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncObjIpcImport.Ilwalid_RequestedPermHigherThanGrantedPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - where input permissions is higher than granted perm.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor contains LwSciSyncAccessPerm value lower than that set in "permissions" input.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList with LwSciSyncAccessPerm_WaitSignal actual perm.
 * - permissions set to LwSciSyncAccessPerm_Auto).
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853224}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList[0].refAttrList.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.len[0]:5
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:true
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_ActualPerm
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_WaitSignal );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncObjIpcImport.NULL_attrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_012.LwSciSyncObjIpcImport.NULL_attrList
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncObjIpcImport.NULL_attrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - for NULL attrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure per respective SWUD:
 * - LwSciSyncAttrListGetAttr().
 * All other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to NULL.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853227}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.len[0]:5
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_ActualPerm
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncObjIpcImport.Ilwalid_inputAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_013.LwSciSyncObjIpcImport.Ilwalid_inputAttrList
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncObjIpcImport.Ilwalid_inputAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - for invalid inputAttrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncAttrListGetAttr().}
 * All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853230}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList[0].refAttrList.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.len[0]:5
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_ActualPerm
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListGetAttr
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncObjIpcImport.Ilwalid_lengthOfLwSciSyncAccessPerm
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_014.LwSciSyncObjIpcImport.Ilwalid_lengthOfLwSciSyncAccessPerm
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncObjIpcImport.Ilwalid_lengthOfLwSciSyncAccessPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - Invalid length of LwSciSyncAccessPerm in obj descriptor.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with length of LwSciSyncAccessPerm set to invalid value.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.}
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853233}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncObjIpcImport.Fail_ToGetValueForAccessPermKeyFromDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_015.LwSciSyncObjIpcImport.Fail_ToGetValueForAccessPermKeyFromDesc
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncObjIpcImport.Fail_ToGetValueForAccessPermKeyFromDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when LwSciCommonTransportGetNextKeyValuePair() fails to get value for access perm key.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for get next key value pair for LwSciSyncCoreDescKey_AccessPerm.
 * - LwSciCommonTransportGetNextKeyValuePair().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_Overflow.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853236}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 5 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return
static int ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return>> = ( LwSciError_Overflow );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncObjIpcImport.Fail_ToGetDescBufAndParams
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_016.LwSciSyncObjIpcImport.Fail_ToGetDescBufAndParams
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncObjIpcImport.Fail_ToGetDescBufAndParams}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - when LwSciCommonTransportGetRxBufferAndParams() fails to get rx buffer and param for input desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return success for all cases as per respective SWUD, except for fail to get rx buffer and param for input desc.
 * - LwSciCommonTransportGetRxBufferAndParams().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_InsufficientMemory.}
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return
static int ReturnCount=0;
ReturnCount++;

if(1==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return>> = ( LwSciError_Success );
}
else if(2==ReturnCount)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return>> = ( LwSciError_InsufficientMemory );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncObjIpcImport.Fail_ToAllocNewSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_017.LwSciSyncObjIpcImport.Fail_ToAllocNewSyncObj
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncObjIpcImport.Fail_ToAllocNewSyncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when LwSciCommonAllocObjWithRef() fails due to lack of system resource.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonAllocObjWithRef().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853239}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_ResourceError
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_020.LwSciSyncObjIpcImport.Ilwalid_keyInPlaceOfExpectedSyncObjKey
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_020.LwSciSyncObjIpcImport.Ilwalid_keyInPlaceOfExpectedSyncObjKey
TEST.NOTES:
/**
 * @testname{TC_020.LwSciSyncObjIpcImport.Ilwalid_keyInPlaceOfExpectedSyncObjKey}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when received invalid key in place of expected LwSciSyncCoreDescKey_SyncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor containing invalid key in place of LwSciSyncCoreDescKey_SyncObj.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853248}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:0
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_021.LwSciSyncObjIpcImport.Fail_ToImportTooBigDescResultingOverflow
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_021.LwSciSyncObjIpcImport.Fail_ToImportTooBigDescResultingOverflow
TEST.NOTES:
/**
 * @testname{TC_021.LwSciSyncObjIpcImport.Fail_ToImportTooBigDescResultingOverflow}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when importing too big desc resulting in overflow.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonTransportGetNextKeyValuePair().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of large LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_Overflow.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853251}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciSyncObjIpcImport.Ilwalid_DescChecksumValue
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_022.LwSciSyncObjIpcImport.Ilwalid_DescChecksumValue
TEST.NOTES:
/**
 * @testname{TC_022.LwSciSyncObjIpcImport.Ilwalid_DescChecksumValue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() for computed checksum value does not match checksum value stored in input desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonTransportGetRxBufferAndParams().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with corrupted checksum.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.}
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853254}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncObjIpcImport.Fail_ToCloneInputAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_023.LwSciSyncObjIpcImport.Fail_ToCloneInputAttrList
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncObjIpcImport.Fail_ToCloneInputAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when cloning of input LwSciSyncAttrList fails due to system lacks resource other than memory.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListClone().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_ResourceError.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853257}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncObjIpcImport.Fail_ToAllocMemForLocalDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_026.LwSciSyncObjIpcImport.Fail_ToAllocMemForLocalDesc
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncObjIpcImport.Fail_ToAllocMemForLocalDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() - failed to allocate memory for local desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonCalloc().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_InsufficientMemory.}
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853266}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_027.LwSciSyncObjIpcImport.Fail_ToImportWithUnreconciledAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_027.LwSciSyncObjIpcImport.Fail_ToImportWithUnreconciledAttrList
TEST.NOTES:
/**
 * @testname{TC_027.LwSciSyncObjIpcImport.Fail_ToImportWithUnreconciledAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when importing with unreconciled LwSciSyncAttrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to unreconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853269}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_028.LwSciSyncObjIpcImport.NULL_inputAttrList
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_028.LwSciSyncObjIpcImport.NULL_inputAttrList
TEST.NOTES:
/**
 * @testname{TC_028.LwSciSyncObjIpcImport.NULL_inputAttrList}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() for NULL inputAttrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncAttrListIsReconciled().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to NULL.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853272}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_029.LwSciSyncObjIpcImport.Ilwalid_ipcEndpoint
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_029.LwSciSyncObjIpcImport.Ilwalid_ipcEndpoint
TEST.NOTES:
/**
 * @testname{TC_029.LwSciSyncObjIpcImport.Ilwalid_ipcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when ipcEndpoint is not a valid LwSciIpcEndpoint.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciIpcGetEndpointInfo().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to invalid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853275}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:0
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_030.LwSciSyncObjIpcImport.Ilwalid_permissions
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_030.LwSciSyncObjIpcImport.Ilwalid_permissions
TEST.NOTES:
/**
 * @testname{TC_030.LwSciSyncObjIpcImport.Ilwalid_permissions}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when input permissions is invalid.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- permissions set to LwSciSyncAccessPerm_SignalOnly(invalid import perm).
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853278}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.END

-- Test Case: TC_031.LwSciSyncObjIpcImport.NULL_desc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_031.LwSciSyncObjIpcImport.NULL_desc
TEST.NOTES:
/**
 * @testname{TC_031.LwSciSyncObjIpcImport.NULL_desc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when input desc is NULL.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- desc is NULL.
 * - ipcEndpoint set to valid LwSciIpcEndpoint..
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853281}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions:LwSciSyncAccessPerm_Auto
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.END

-- Test Case: TC_032.LwSciSyncObjIpcImport.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_032.LwSciSyncObjIpcImport.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_032.LwSciSyncObjIpcImport.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() for NULL syncObj.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{N/A.}
 *
 * @testinput{- syncObj is set to NULL.
 * - ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of valid LwSciSyncObjIpcExportDescriptor.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853284}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions:LwSciSyncAccessPerm_Auto
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.END

-- Test Case: TC_033.LwSciSyncObjIpcImport.Fail_ToDueRepeatedTagInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_033.LwSciSyncObjIpcImport.Fail_ToDueRepeatedTagInDesc
TEST.NOTES:
/**
 * @testname{TC_033.LwSciSyncObjIpcImport.Fail_ToDueRepeatedTagInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when repeated tag in input desc.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to LwSciSyncObjIpcExportDescriptor pointer which contains repeated tags.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853287}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal:MACRO=LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList[0].refAttrList.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.len[0]:5
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListGetAttr.key:LwSciSyncAttrKey_ActualPerm
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListGetAttr
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
*<<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}
if(3==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
else if(3==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
else if(3==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(3==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListGetAttr.value
{{ <<uut_prototype_stubs.LwSciSyncAttrListGetAttr.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListSetActualPerm.attrList>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_WaitOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_034.LwSciSyncObjIpcImport.Ilwalid_PermissionInDesc
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_034.LwSciSyncObjIpcImport.Ilwalid_PermissionInDesc
TEST.NOTES:
/**
 * @testname{TC_034.LwSciSyncObjIpcImport.Ilwalid_PermissionInDesc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when value of LwSciSyncCoreObjKey_AccessPerm key in desc is invalid.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor which contains invalid value for LwSciSyncCoreObjKey_AccessPerm key.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853290}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024,4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
static int count3=0;
count3++;
if(1==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( LW_SCI_SYNC_TRANSPORT_OBJ_MAGIC );
}
if(2==count3)
{
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 41 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 42 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->keyCount = ( 43 );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
static int count5=0;
count5++;
if(1==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreDescKey_SyncObj );
}
if(2==count5)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> = ( LwSciSyncCoreObjKey_AccessPerm );
}

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
static int count3=0;
count3++;
if(1==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 4 );
}
else if(2==count3)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> = ( 8 );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static int count7=0;
count7++;
if(1==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &VECTORCAST_INT1 );
}
else if(2==count7)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<lwscisync_object_external.<<GLOBAL>>.Param64value>> );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
static int count8=0;
count8++;
if(1==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( false );
}
else if(2==count8)
{
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> = ( true );
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int count=0;
count++;
if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&VECTORCAST_INT1) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
static int count1=0;
count1++;
if(1==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}
else if(2==count1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 4 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_035.LwSciSyncObjIpcImport.Ilwalid_ExportDescriptorsMagicID
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjIpcImport
TEST.NEW
TEST.NAME:TC_035.LwSciSyncObjIpcImport.Ilwalid_ExportDescriptorsMagicID
TEST.NOTES:
/**
 * @testname{TC_035.LwSciSyncObjIpcImport.Ilwalid_ExportDescriptorsMagicID}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjIpcImport() when invalid export descriptor's magicID.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjIpcImport().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciCommonTransportGetNextKeyValuePair().
 * All the other stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{- ipcEndpoint set to valid LwSciIpcEndpoint.
 * - desc is set to pointer of LwSciSyncObjIpcExportDescriptor with invalid magic ID.
 * - inputAttrList is set to valid reconciled LwSciSyncAttrList.
 * - permissions set to LwSciSyncAccessPerm_Auto.
 * - timeoutUs set to non-zero value.
 * - syncObj is set to LwSciSyncObj pointer.}
 *
 * @testbehavior{- LwSciSyncObjIpcImport() returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{}
 *
 * @verify{18844713}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0..3]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[10]:0
TEST.VALUE:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[127]:0
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.ipcEndpoint:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.desc[0].payload[0]:5
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.timeoutUs:3
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncAttrListClone.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor.payload[0]:5
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjIpcImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:1024
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize:80
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize:64
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjIpcImport
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  uut_prototype_stubs.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncAttrListClone
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_object_external.c.LwSciSyncObjIpcImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjRec>>.refObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgVersion = ( 8589934592 );
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>>->msgMagic = ( 1 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info
{{ <<uut_prototype_stubs.LwSciIpcGetEndpointInfo.info>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.attrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled
{{ <<uut_prototype_stubs.LwSciSyncAttrListIsReconciled.isReconciled>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.origAttrList>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.inputAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListClone.newAttrList>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_object_external.LwSciSyncObjIpcImport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncObjIpcExportDescriptor>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize>> == ( 1024 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjFreeObjAndRef.syncObj>> == ( *<<lwscisync_object_external.LwSciSyncObjIpcImport.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.Param64value
<<lwscisync_object_external.<<GLOBAL>>.Param64value>> = ( (uint64_t)LwSciSyncAccessPerm_SignalOnly);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.<<GLOBAL>>.ParmattrVal
<<lwscisync_object_external.<<GLOBAL>>.ParmattrVal>> = ( (uint64_t)LwSciSyncAccessPerm_Auto );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_external.LwSciSyncObjIpcImport.permissions
<<lwscisync_object_external.LwSciSyncObjIpcImport.permissions>> = ( (uint64_t)LwSciSyncAccessPerm_Auto);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjSignal

-- Test Case: TC_001.LwSciSyncObjSignal.Successful_SignalingOfSyncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjSignal
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjSignal.Successful_SignalingOfSyncObj
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncObjSignal.Successful_SignalingOfSyncObj.}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjSignal() when signal operation on LwSciSyncObj is successful.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjSignal().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{syncObj set to valid LwSciSyncObj with primitive allocated by LwSciSync and has CPU access perm.}
 *
 * @testbehavior{- LwSciSyncObjSignal() Returns LwSciError_Success.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853293}
 *
 * @verify{18844719}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT2
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:false
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjSignal
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciSyncCoreSignalPrimitive
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_object_external.c.LwSciSyncObjSignal
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (&<<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreSignalPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreSignalPrimitive.primitive>> == ( <<lwscisync_object_external.<<GLOBAL>>.ParamLwSciSyncCoreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjSignal.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjSignal.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( &<<lwscisync_object_external.LwSciSyncObjSignal.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncObjSignal.Ilwalid_syncObjWithExternalPrimitive
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjSignal
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjSignal.Ilwalid_syncObjWithExternalPrimitive
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncObjSignal.Ilwalid_syncObjWithExternalPrimitive}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjSignal() when syncObj is backed by external primitive.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjSignal().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{syncObj set to valid LwSciSyncObj backed by external primitive.}
 *
 * @testbehavior{- LwSciSyncObjSignal() Returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853299}
 *
 * @verify{18844719}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT2
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.signalerUseExternalPrimitive[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjSignal
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive
  lwscisync_object_external.c.LwSciSyncObjSignal
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreGetSignalerUseExternalPrimitive.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjSignal.Ilwalid_syncObjWithoutCpuPerm
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjSignal
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjSignal.Ilwalid_syncObjWithoutCpuPerm
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncObjSignal.Ilwalid_syncObjWithoutCpuPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjSignal() syncObj's LwSciSyncAttrList.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjSignal().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{syncObj set to valid LwSciSyncObj without CPU access permission.}
 *
 * @testbehavior{- LwSciSyncObjSignal() Returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853302}
 *
 * @verify{18844719}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj[0].refObj.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList[0].refAttrList.objPtr:VECTORCAST_INT2
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.isCpuSignaler[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjSignal
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler
  lwscisync_object_external.c.LwSciSyncObjSignal
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj
{{ <<uut_prototype_stubs.LwSciSyncCoreObjValidate.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListTypeIsCpuSignaler.attrList>> == ( &<<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>>[0][0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncObjSignal.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjSignal
TEST.NEW
TEST.NAME:TC_006.LwSciSyncObjSignal.Ilwalid_syncObj
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncObjSignal.Ilwalid_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjSignal() - when input syncObj is invalid.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjSignal().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{syncObj set to invalid LwSciSyncObj.}
 *
 * @testbehavior{- LwSciSyncObjSignal() panics.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853308}
 *
 * @verify{18844719}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj[0].refObj.objPtr:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjSignal
  uut_prototype_stubs.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj>> == ( <<lwscisync_object_external.LwSciSyncObjSignal.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncObjSignal.NULL_syncObj
TEST.UNIT:lwscisync_object_external
TEST.SUBPROGRAM:LwSciSyncObjSignal
TEST.NEW
TEST.NAME:TC_007.LwSciSyncObjSignal.NULL_syncObj
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncObjSignal.NULL_syncObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncObjSignal() when input syncObj Is NULL.}
 *
 * @testpurpose{Unit testing of LwSciSyncObjSignal().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
 * - LwSciSyncObjGetAttrList().}
 *
 * @testinput{syncObj set to NULL.}
 *
 * @testbehavior{- LwSciSyncObjSignal() Returns LwSciError_BadParameter.
 * - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{18853311}
 *
 * @verify{18844719}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.syncObj:<<null>>
TEST.VALUE:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_object_external.LwSciSyncObjSignal.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncObj:<<null>>
TEST.FLOW
  lwscisync_object_external.c.LwSciSyncObjSignal
  uut_prototype_stubs.LwSciSyncObjGetAttrList
  lwscisync_object_external.c.LwSciSyncObjSignal
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList
{{ <<uut_prototype_stubs.LwSciSyncObjGetAttrList.syncAttrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

