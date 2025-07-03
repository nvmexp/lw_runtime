-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_ATTRIBUTE_RECONCILE
-- Unit(s) Under Test: lwscisync_attribute_reconcile
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

-- Subprogram: LwSciSyncAttrListIsReconciled

-- Test Case: TC_001.LwSciSyncAttrListIsReconciled.NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListIsReconciled
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListIsReconciled.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListIsReconciled.NormalOperation}
 *
 * @verifyFunction{This test-case checks attribute list is reconciled successfully}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreAttrListValidate() returns success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * test_objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled}
 *
 * @testinput{attrList set to valid memory
 * isReconciled set to vaid memory}
 *
 * @testbehavior{LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * isReconciled set to true
 * LwSciError_Success returned}
 *
 * @testcase{18851946}
 *
 * @verify{18844326}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr:<<null>>
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( & <<lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList>>[0].refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListIsReconciled.Unreconciled_NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListIsReconciled
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListIsReconciled.Unreconciled_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListIsReconciled.Unreconciled_NormalOperation}
 *
 * @verifyFunction{This test-case checks success case when LwSciSyncCoreAttrListState is Unreconciled}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreAttrListValidate() returns LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * test_objAttrList.state set to LwSciSyncCoreAttrListState_Unreconciled}
 *
 * @testinput{isReconciled set to vaid memory}
 *
 * @testbehavior{LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * isReconciled set to false
 * LwSciError_Success returned}
 *
 * @testcase{18851949}
 *
 * @verify{18844326}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr:<<null>>
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
test_objAttrList.state = LwSciSyncCoreAttrListState_Unreconciled;
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( & <<lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList>>[0].refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListIsReconciled.attrList_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListIsReconciled
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListIsReconciled.attrList_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListIsReconciled.attrList_NULL}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreAttrListValidate() returns error when attrList set to NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreAttrListValidate() returns LwSciError_BadParameter}
 *
 * @testinput{attrList set to NULL
 * isReconciled set to vaid memory and set to true}
 *
 * @testbehavior{LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18851952}
 *
 * @verify{18844326}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListIsReconciled.isReconciled_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListIsReconciled
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListIsReconciled.isReconciled_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListIsReconciled.isReconciled_NULL}
 *
 * @verifyFunction{This test-case checks error path when isReconciled is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{attrList set to valid memory
 * isReconciled set to NULL}
 *
 * @testbehavior{LwSciError_BadParameter returned}
 *
 * @testcase{18851955}
 *
 * @verify{18844326}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.attrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.isReconciled:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListIsReconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListIsReconciled
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncAttrListReconcile

-- Test Case: TC_001.LwSciSyncAttrListReconcile.IpcTable_Route_IsNotEmpty_NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListReconcile.IpcTable_Route_IsNotEmpty_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListReconcile.IpcTable_Route_IsNotEmpty_NormalOperation}
 *
 * @verifyFunction{This test-case checks success case when IPC Route is not empty}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports set to true
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreGetSupportedPrimitives.primitiveType set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * objAttrList.coreAttrList[0].ipcTable set to valid memory
 * test_objAttrList.coreAttrList[0].ipcTable.ipcRouteEntries set to 1
 * LwSciSyncCoreIpcTableRouteIsEmpty() returns to false
 * LwSciSyncCoreIpcTableTreeAlloc()  returns to LwSciError_Success}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableTreeAlloc() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableAddBranch() receieves correct arguments
 * test_newObjAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * test_newObjAttrList.writable set to false
 * newReconciledList set to  'attrList' from function LwSciSyncCoreAttrListCreateMultiSlot()
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18851958}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcRouteEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcRouteEntries:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Empty
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.valSize[5..7]:0
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:5787213827046133840
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_UBV
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_002.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_UBV
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_UBV}
 *
 * @verifyFunction{This test-case checks error path when inputCount set to upper boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports set to true
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreGetSupportedPrimitives.primitiveType set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * objAttrList.coreAttrList[0].ipcTable set to valid memory 
 * LwSciSyncCoreIpcTableRouteIsEmpty() returns to true
 * LwSciSyncCoreIpcTableTreeAlloc()  returns to LwSciError_Success}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to MAX
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableTreeAlloc() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * test_newObjAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * test_newObjAttrList.writable set to false
 * newReconciledList set to  'attrList' from function LwSciSyncCoreAttrListCreateMultiSlot()
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18851961}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:<<MAX>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Empty
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.valSize[5..7]:0
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:18446744073709551615
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0xFFFFFFFFFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0xFFFFFFFFFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0xFFFFFFFFFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0xFFFFFFFFFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LW
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_003.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LW
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LW}
 *
 * @verifyFunction{This test-case checks error path when inputCount set to nominal value}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports set to true
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreGetSupportedPrimitives.primitiveType set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * objAttrList.coreAttrList[0].ipcTable set to valid memory 
 * LwSciSyncCoreIpcTableRouteIsEmpty() returns to true
 * LwSciSyncCoreIpcTableTreeAlloc()  returns to LwSciError_Success}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableTreeAlloc() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * test_newObjAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * test_newObjAttrList.writable set to false
 * newReconciledList set to  'attrList' from function LwSciSyncCoreAttrListCreateMultiSlot()
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18851964}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Empty
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.valSize[5..7]:0
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LBV
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_004.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LBV
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncAttrListReconcile.IpcTable_Route_IsEmpty_NormalOperation_inputCount_set_to_LBV}
 *
 * @verifyFunction{This test-case checks error path when inputCount set to lower boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports set to true
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreGetSupportedPrimitives.primitiveType set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * objAttrList.coreAttrList[0].ipcTable set to valid memory 
 * LwSciSyncCoreIpcTableRouteIsEmpty() returns to true
 * LwSciSyncCoreIpcTableTreeAlloc()  returns to LwSciError_Success}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 1
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableTreeAlloc() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * test_newObjAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * test_newObjAttrList.writable set to false
 * newReconciledList set to  'attrList' from function LwSciSyncCoreAttrListCreateMultiSlot()
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18851967}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Empty
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.valSize[5..7]:0
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListReconcile.Finaltype_isNotthe_firsttype_by_signaler_NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListReconcile.Finaltype_isNotthe_firsttype_by_signaler_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListReconcile.Finaltype_isNotthe_firsttype_by_signaler_NormalOperation}
 *
 * @verifyFunction{This test-case checks success case when  Final type is not the first type proposed by signaler}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports set to true
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 4
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreGetSupportedPrimitives.primitiveType set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * objAttrList.coreAttrList[0].ipcTable set to valid memory
 * test_objAttrList.coreAttrList[0].ipcTable.ipcRouteEntries set to 1
 * LwSciSyncCoreIpcTableRouteIsEmpty() returns to false
 * LwSciSyncCoreIpcTableTreeAlloc()  returns to LwSciError_Success}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableTreeAlloc() receives correct arguments
 * LwSciSyncCoreIpcTableRouteIsEmpty() receieves correct arguments
 * LwSciSyncCoreIpcTableAddBranch() receieves correct arguments
 * test_newObjAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * test_newObjAttrList.writable set to false
 * newReconciledList set to  'attrList' from function LwSciSyncCoreAttrListCreateMultiSlot()
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18851970}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterContextInsensitiveFenceExports:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_VidmemSemaphore
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:4
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcRouteEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].ipcTable.ipcPermEntries:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Empty
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList[0].attrs.valSize[5..7]:0
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListReconcile.signalerandwaiter_attributelist_mismatch
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListReconcile.signalerandwaiter_attributelist_mismatch
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListReconcile.signalerandwaiter_attributelist_mismatch}
 *
 * @verifyFunction{This test-case checks error path when attribute list mismatch between signaler and waiters (no common primitive)}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_BadParameter
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * 'primitiveType' parameter of LwSciSyncCoreGetSupportedPrimitives() set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreGetSupportedPrimitives() receives correct arguments 
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_UnsupportedConfig returned}
 *
 * @testcase{18851979}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:5787213827046133840
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListReconcile.Ilwalid_number_of_signalers
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListReconcile.Ilwalid_number_of_signalers
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListReconcile.Ilwalid_number_of_signalers}
 *
 * @verifyFunction{This test-case checks error path when Invalid number of signalers found (Multi-signaler) }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_BadParameter
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 3
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * 'primitiveType' parameter of LwSciSyncCoreGetSupportedPrimitives() set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[1].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[2].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[2].attrs.requiredPerm set to LwSciSyncAccessPerm_SignalOnly}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreCopyCpuPrimitives() recives correct arguments
 * LwSciSyncCoreGetSupportedPrimitives() recives correct arguments
 * LwSciCommonMemcpy() receives correct arguments
 * LwSciCommonMemcpy() receives correct arguments(second event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_UnsupportedConfig returned}
 *
 * @testcase{18851982}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 3>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[1].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[2].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[2].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:3
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[1].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[2].attrs.needCpuAccess:true
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[2].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:3
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:5787213827046133840
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<null>>
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<null>>
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_UnsupportedConfig
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciSyncAccessPerm) ) }}
}
if( i == 1 ) 
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList[0].attrs.signalerPrimitiveInfo) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList[0].attrs.signalerPrimitiveInfo) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof( uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList[0].attrs.signalerPrimitiveInfo) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList[0].attrs.signalerPrimitiveInfo) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListReconcile.ActualPerm_signaler
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListReconcile.ActualPerm_signaler
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListReconcile.ActualPerm_signaler}
 *
 * @verifyFunction{This test-case checks error path when invalid signaler permssions found
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 0
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * }
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreCopyCpuPrimitives() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_ReconciliationFailed returned}
 *
 * @testcase{18851985}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:4
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:5787213827046133840
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncAttrListReconcile.ActualPerm_Waiter
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_011.LwSciSyncAttrListReconcile.ActualPerm_Waiter
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListReconcile.ActualPerm_Waiter}
 *
 * @verifyFunction{This test-case checks error path when invalid waiter permssions found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 0
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 2
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitOnly
 * }
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncCoreCopyCpuPrimitives() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_ReconciliationFailed returned}
 *
 * @testcase{18851988}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:1
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:4
TEST.EXPECTED:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:5787213827046133840
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncAttrListReconcile.numCoreAttrList_Ilwalid_NoReconcile_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_012.LwSciSyncAttrListReconcile.numCoreAttrList_Ilwalid_NoReconcile_permissions
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncAttrListReconcile.numCoreAttrList_Ilwalid_NoReconcile_permissions}
 *
 * @verifyFunction{This test-case checks error path when numCoreAttrList is zero and Invalid permissions received.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 0
 * test_newObjAttrList.coreAttrList set to valid memory}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_ReconciliationFailed returned}
 *
 * @testcase{18851991}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_newObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_ReconciliationFailed
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncAttrListReconcile.Key_signalerPrimitiveCounts_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_013.LwSciSyncAttrListReconcile.Key_signalerPrimitiveCounts_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncAttrListReconcile.Key_signalerPrimitiveCounts_Ilwalid}
 *
 * @verifyFunction{This test-case checks error path when value of LwSciSyncInternalAttrKey_SignalerPrimitiveCount is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_objAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked
 * test_objAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18851994}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncAttrListReconcile.Key_WaiterPrimitiveInfo_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_014.LwSciSyncAttrListReconcile.Key_WaiterPrimitiveInfo_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncAttrListReconcile.Key_WaiterPrimitiveInfo_Ilwalid}
 *
 * @verifyFunction{This test-case checks error path when value of LwSciSyncInternalAttrKey_WaiterPrimitiveInfo is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_LowerBound
 * test_objAttrList.coreAttrList[0].attrs.valSize[6] set to 8}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18851997}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[6]:8
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncAttrListReconcile.Key_SignalerPrimitiveInfo_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_015.LwSciSyncAttrListReconcile.Key_SignalerPrimitiveInfo_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncAttrListReconcile.Key_SignalerPrimitiveInfo_Ilwalid}
 *
 * @verifyFunction{This test-case checks error path when LwSciSyncInternalAttrKey_SignalerPrimitiveInfo is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_newObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_LowerBound
 * test_objAttrList.coreAttrList[0].attrs.valSize[5] set to 8}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852000}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.valSize[5]:8
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncAttrListReconcile.Key_RequiredPerm_Ilwalid_value
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_016.LwSciSyncAttrListReconcile.Key_RequiredPerm_Ilwalid_value
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncAttrListReconcile.Key_RequiredPerm_Ilwalid_value}
 *
 * @verifyFunction{This test-case checks error path when Invalid value for LwSciSyncAttrKey_RequiredPerm:}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * parameter 'attrList' LwSciSyncCoreAttrListCreateMultiSlot() set to valid memory
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (2nd event)
 * test_objAttrList.numCoreAttrList set to 1
 * test_objAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_Auto}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments (2nd event)
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852003}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_Auto
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_newObjAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
else if (cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0][0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_018.LwSciSyncAttrListReconcile.module_is_ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_018.LwSciSyncAttrListReconcile.module_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_018.LwSciSyncAttrListReconcile.module_is_ilwalid}
 *
 * @verifyFunction{This test-case checks where LwSciSyncAttrListReconcile() panics when module is invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * test_objAttrList.module set to '0xFFFFFFFFFFFFFFFF'(Invalid value)
 * LwSciSyncCoreAttrListCreateMultiSlot() panics}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncAttrListReconcile() panics}
 *
 * @testcase{18852009}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( 0xFFFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.module
<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncAttrListReconcile.LwSciSyncCoreAttrListCreateMultiSlot_InsufficientMemory
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_019.LwSciSyncAttrListReconcile.LwSciSyncCoreAttrListCreateMultiSlot_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncAttrListReconcile.LwSciSyncCoreAttrListCreateMultiSlot_InsufficientMemory}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreAttrListCreateMultiSlot() returns error when there is no memory to create a new LwSciSyncAttrList}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListsLock() returns to LwSciError_Success
 * parameter 'newUnreconciledAttrList[0]' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success
 * LwSciSyncCoreAttrListCreateMultiSlot() returns to LwSciError_InsufficientMemory}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0x5050505050505050
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListsLock() receives correct arguments
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments
 * LwSciSyncCoreAttrListCreateMultiSlot() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments (1st event)
 * LwSciSyncAttrListFree() receives correct arguments (2nd event)
 * LwSciSyncCoreAttrListsUnlock() receives correct arguments
 * LwSciError_InsufficientMemory returned}
 *
 * @testcase{18852012}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x5050505050505050
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5050505050505050
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.valueCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListsLock
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (  &(test_objAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.attrList>>[0] ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListCreateMultiSlot.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsLock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr.inputAttrListArr[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListsUnlock.inputAttrListArr>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncAttrListReconcile.inputArray_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_023.LwSciSyncAttrListReconcile.inputArray_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncAttrListReconcile.inputArray_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncAttrListReconcile() panics}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() panics}
 *
 * @testinput{inputArray set to 0xFFFFFFFFFFFFFFFF(Invalid value)
 * inputCount set to 1
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncAttrListReconcile() panics}
 *
 * @testcase{18852024}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray
<<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncAttrListReconcile.inputArray_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_024.LwSciSyncAttrListReconcile.inputArray_NULL
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncAttrListReconcile.inputArray_NULL}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreValidateAttrListArray() returns error when inputArray is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_BadParameter}
 *
 * @testinput{inputArray set to NULL
 * inputCount set to 1
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852027}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.END

-- Test Case: TC_025.LwSciSyncAttrListReconcile.inputCount_zero
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_025.LwSciSyncAttrListReconcile.inputCount_zero
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncAttrListReconcile.inputCount_zero}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreValidateAttrListArray() returns error when inputCount is zero}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_BadParameter}
 *
 * @testinput{inputArray set to valid memory
 * inputCount set to 0
 * newReconciledList set to valid memory
 * newConflictList set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852030}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray[0]:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:0x0
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newConflictList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x0
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:false
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncAttrListReconcile.newReconciledList_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListReconcile
TEST.NEW
TEST.NAME:TC_026.LwSciSyncAttrListReconcile.newReconciledList_NULL
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncAttrListReconcile.newReconciledList_NULL}
 *
 * @verifyFunction{This test-case checks error path when newReconciledList is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{newReconciledList set to NULL}
 *
 * @testbehavior{LwSciError_BadParameter returned}
 *
 * @testcase{18852033}
 *
 * @verify{18844329}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.newReconciledList:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListReconcile
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncAttrListValidateReconciled

-- Test Case: TC_001.LwSciSyncAttrListValidateReconciled.Successful_Validation_CpuSignalerandNonCpuWaiter
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_001.LwSciSyncAttrListValidateReconciled.Successful_Validation_CpuSignalerandNonCpuWaiter
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncAttrListValidateReconciled.Successful_Validation_CpuSignalerandNonCpuWaiter}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList against a set of input
 * Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_Success.
 * - isReconciledListValid set to true
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
 * @testcase{18852036}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncAttrListValidateReconciled.Ilwalid_primitiveInfo_in_reconciled_attrlist
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_005.LwSciSyncAttrListValidateReconciled.Ilwalid_primitiveInfo_in_reconciled_attrlist
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncAttrListValidateReconciled.Ilwalid_primitiveInfo_in_reconciled_attrlist}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList containing invalid primitive info
 * against a set of input Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists .}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852048}
 *
 * @verify{18844332}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncAttrListValidateReconciled.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_006.LwSciSyncAttrListValidateReconciled.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncAttrListValidateReconciled.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList containing different waiterPrimitiveInfo and signalerPrimitiveInfo
 * against a set of input Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists .}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852051}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_SignalerPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_007.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_SignalerPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_SignalerPrimitiveInfo}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList containing multiple primitives in signalerPrimitiveInfo
 * against a set of input Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists .}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore and LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852054}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[1]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[1]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_WaiterPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_008.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_WaiterPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncAttrListValidateReconciled.Too_many_primitives_in_WaiterPrimitiveInfo}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList containing multiple primitives in waiterPrimitiveInfo
 * against a set of input Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists .}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint and LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852057}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[1]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[1]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncAttrListValidateReconciled.ilwalid_reconciledSignalerPrimitiveCountForCpuSignaler
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_009.LwSciSyncAttrListValidateReconciled.ilwalid_reconciledSignalerPrimitiveCountForCpuSignaler
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncAttrListValidateReconciled.ilwalid_reconciledSignalerPrimitiveCountForCpuSignaler}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList containing invalid signalerPrimitiveCount against a set of input
 * Cpu Signaler and non-Cpu Waiter unreconciled LwSciSyncAttrLists.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint and signalerPrimitiveCount set to 2
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852060}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:2
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncAttrListValidateReconciled.Ilwalid_reconciledAttrListState
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_010.LwSciSyncAttrListValidateReconciled.Ilwalid_reconciledAttrListState
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncAttrListValidateReconciled.Ilwalid_reconciledAttrListState}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() for invalid state  of input reconciled LwSciSyncAttrList}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with state as LwSciSyncCoreAttrListState_Unreconciled, actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to true, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852063}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Unreconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList.coreAttrList[0].attrs.valSize.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList.coreAttrList[0].attrs.valSize.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList.coreAttrList[0].attrs.valSize.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList.coreAttrList[0].attrs.valSize.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList.coreAttrList[0].attrs.valSize.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList.coreAttrList[0].attrs.valSize.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList.coreAttrList[0].attrs.valSize.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList.coreAttrList[0].attrs.valSize.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList.coreAttrList[0].attrs.valSize.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList.coreAttrList[0].attrs.valSize.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncAttrListValidateReconciled.ReconciledList_has_insufficient_cpu_access_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_011.LwSciSyncAttrListValidateReconciled.ReconciledList_has_insufficient_cpu_access_permissions
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncAttrListValidateReconciled.ReconciledList_has_insufficient_cpu_access_permissions}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() for insufficient cpuAccess permissions of input reconciled LwSciSyncAttrList}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with state as LwSciSyncCoreAttrListState_Reconciled, actualPerm set to LwSciSyncAccessPerm_WaitSignal, 
 *  needCpuAccess set to false, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly and needCpuAccess set to true
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to false and waiterPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_BadParameter.
 * - isReconciledListValid set to false
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
 * @testcase{18852066}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[6]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitSignal ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == (  sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[6] = ( sizeof(LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncAttrListValidateReconciled.Insufficient_Reconciled_list_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_012.LwSciSyncAttrListValidateReconciled.Insufficient_Reconciled_list_permissions
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncAttrListValidateReconciled.Insufficient_Reconciled_list_permissions}
 *
 * @verifyFunction{This test-case checks eror path where LwSciSyncCoreAttrListPermLessThan() returns true (Insufficient Reconciled list permissions)}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitSignal
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * LwSciSyncCoreAttrListPermLessThan() returns to true}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852069}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncAttrListValidateReconciled.Reconciledattributelist_doesnot_has_atleast_waitpermissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_013.LwSciSyncAttrListValidateReconciled.Reconciledattributelist_doesnot_has_atleast_waitpermissions
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncAttrListValidateReconciled.Reconciledattributelist_doesnot_has_atleast_waitpermissions}
 *
 * @verifyFunction{This test-case checks eror path when LwSciSyncCoreAttrListPermLessThan() returns true (Reconciled attri list must have at least wait permissions)}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to true}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852072}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (NULL );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncAttrListValidateReconciled.Reconciliation_failed
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_014.LwSciSyncAttrListValidateReconciled.Reconciliation_failed
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncAttrListValidateReconciled.Reconciliation_failed}
 *
 * @verifyFunction{This test-case checks Reconciliation failure when there is Invalid permissions }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_ReconciliationFailed returned}
 *
 * @testcase{18852075}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncAttrListValidateReconciled.Modules_not_duplicated
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_015.LwSciSyncAttrListValidateReconciled.Modules_not_duplicated
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncAttrListValidateReconciled.Modules_not_duplicated}
 *
 * @verifyFunction{This test-case checks error path when attr list not belong to same module }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to false
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852078}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncAttrListValidateReconciled.reconciledObjAttrList_module_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_016.LwSciSyncAttrListValidateReconciled.reconciledObjAttrList_module_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncAttrListValidateReconciled.reconciledObjAttrList_module_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncAttrListValidateReconciled() panics when reconciledObjAttrList module is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * test_reconciledObjAttrList.module set to 0xFFFFFFFFFFFFFFFF (Invalid value)
 * LwSciSyncCoreModuleIsDup() panics
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncAttrListValidateReconciled() panics}
 *
 * @testcase{18852081}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.module
<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_module_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_017.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_module_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_module_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncAttrListValidateReconciled() panics when unreconciledObjAttrList module is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * test_unreconciledObjAttrList.module set to 0xFFFFFFFFFFFFFFFF (Invalid value)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() panics
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncAttrListValidateReconciled() panics}
 *
 * @testcase{18852084}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.module
<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_NULL_NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_019.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_NULL_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncAttrListValidateReconciled.unreconciledObjAttrList_NULL_NormalOperation}
 *
 * @verifyFunction{This test-case checks error path when unreconciledObjAttrList is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success 
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments  (unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments  (reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852090}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( NULL );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciSyncAttrListValidateReconciled.reconciledAttrList_is_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_022.LwSciSyncAttrListValidateReconciled.reconciledAttrList_is_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_022.LwSciSyncAttrListValidateReconciled.reconciledAttrList_is_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncAttrListValidateReconciled() panics when reconciledObjAttrList is not a valid reference object}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success 
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() panics (objAttrList)
 * }
 *
 * @testinput{reconciledAttrList set to 0xFFFFFFFFFFFFFFFF
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncAttrListValidateReconciled() panics}
 *
 * @testcase{18852099}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncAttrListValidateReconciled.isReconciledListValid_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_023.LwSciSyncAttrListValidateReconciled.isReconciledListValid_NULL
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncAttrListValidateReconciled.isReconciledListValid_NULL}
 *
 * @verifyFunction{This test-case checks the error path when isReconciledListValid is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to NULL
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852102}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_024.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncAttrListValidateReconciled() panics when inputUnreconciledAttrListArray is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() panics}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to 0xFFFFFFFFFFFFFFFF
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncAttrListValidateReconciled() panics}
 *
 * @testcase{18852105}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_025.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount_Zero
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_025.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount_Zero
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount_Zero}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreValidateAttrListArray() returns error when inputUnreconciledAttrListCount set to zero}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_BadParameter}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 0
 * isReconciledListValid set to valid memory
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852108}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:0
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.END

-- Test Case: TC_026.LwSciSyncAttrListValidateReconciled.reconciledAttrList_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_026.LwSciSyncAttrListValidateReconciled.reconciledAttrList_NULL
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncAttrListValidateReconciled.reconciledAttrList_NULL}
 *
 * @verifyFunction{This test-case checks the error path when reconciledAttrList is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{reconciledAttrList set to NULL
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * acquireLocks set to false}
 *
 * @testbehavior{LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852111}
 *
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]:<<malloc 1>>
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( <<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_032.LwSciSyncAttrListValidateReconciled.Successful_Validation_NonCpuSignalerandCpuWaiter
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncAttrListValidateReconciled
TEST.NEW
TEST.NAME:TC_032.LwSciSyncAttrListValidateReconciled.Successful_Validation_NonCpuSignalerandCpuWaiter
TEST.NOTES:
/**
 * @testname{TC_032.LwSciSyncAttrListValidateReconciled.Successful_Validation_NonCpuSignalerandCpuWaiter}
 *
 * @verifyFunction{This test-case verifies the functionality of the API-LwSciSyncAttrListValidateReconciled() successfullly validates a reconciled LwSciSyncAttrList against a set of input
 * non-Cpu Signaler and Cpu waiter unreconciled LwSciSyncAttrLists.}
 *
 * @testpurpose{Unit testing of LwSciSyncAttrListValidateReconciled().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{All stub functions are simulated to return success as per respective SWUD.}
 *
 * @testinput{reconciledAttrList set to valid LwSciSyncAttrList with actualPerm set to LwSciSyncAccessPerm_WaitSignal, signalerPrimitiveCount set to 5,
 *  needCpuAccess set to true, waiterPrimitiveInfo and signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * inputUnreconciledAttrListArray set to array of following LwSciSyncAttrList:
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_SignalOnly, needCpuAccess set to false, signalerPrimitiveInfo set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 *  and signalerPrimitiveCount set to 5
 *  - unreconciled LwSciSyncAttrList with requiredperm set to LwSciSyncAccessPerm_WaitOnly, needCpuAccess set to true
 * inputUnreconciledAttrListCount set to number of entries in LwSciSyncAttrList array
 * isReconciledListValid set to pointer of boolean}
 *
 * @testbehavior{- LwSciSyncAttrListValidateReconciled() returns LwSciError_Success.
 * - isReconciledListValid set to true
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
 * @verify{18844332}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:5
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.signalerPrimitiveCount:5
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[5]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.coreAttrList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.signalerPrimitiveCount:5
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[0]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[2]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.keyState[5..7]:LwSciSyncCoreAttrKeyState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.state:LwSciSyncCoreAttrListState_Reconciled
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.needCpuAccess:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.signalerPrimitiveCount:5
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[5]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.needCpuAccess:true
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.keyState[0..1]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj.numCoreAttrList:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListCount:2
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.isReconciledListValid[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.acquireLocks:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncAttrListValidateReconciled
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[0] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.coreAttrList[1] = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0];
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.module = <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module;

*<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return
<<uut_prototype_stubs.LwSciSyncCorePermLessThan.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permA>> < <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.return
<<uut_prototype_stubs.LwSciSyncCorePermLEq.return>> = ( <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> <= <<uut_prototype_stubs.LwSciSyncCorePermLEq.permB>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( malloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>> * <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType
*<<uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> = ( LwSciSyncInternalAttrValPrimitiveType_Syncpoint );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> != ( NULL ) }}

  LwSciSyncCoreAttrListObj* syncCoreAttrListObj;

       LwSciObj* objAttrListParam = <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> -> refAttrList.objPtr ;
     size_t addr = (size_t)(&(((LwSciSyncCoreAttrListObj *)0)->objAttrList)) ;
     syncCoreAttrListObj = (LwSciSyncCoreAttrListObj*) (void*) ((char*)(void*)objAttrListParam - addr);

{{ syncCoreAttrListObj->header != ( 0xFFFFFFFF ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount>> != ( 0 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( 0x1234 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.actualPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList[0].refAttrList.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledAttrList>>[0].refAttrList.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.appendedUnreconciledattrListObj>>.objAttrList );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[2]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[2] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[5] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[6]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[6] = ( sizeof( LwSciSyncInternalAttrValPrimitiveType ) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrListObj>>.coreAttrList[0].attrs.valSize[7] = ( sizeof( uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[5]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[5] = ( sizeof(   LwSciSyncInternalAttrValPrimitiveType) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1.coreAttrList[0].attrs.valSize[7]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj1>>.coreAttrList[0].attrs.valSize[7] = ( sizeof(   uint32_t) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.module
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.module = ( 0x1234 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[0]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[0] = ( sizeof(bool) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2.coreAttrList[0].attrs.valSize[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrListObj2>>.coreAttrList[0].attrs.valSize[1] = ( sizeof( LwSciSyncAccessPerm) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.reconciledAttrList>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.reconciledattrList>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[0]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray[1]
<<lwscisync_attribute_reconcile.LwSciSyncAttrListValidateReconciled.inputUnreconciledAttrListArray>>[1] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.unreconciledattrList2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreAttrListValidateReconciledWithLocks

-- Test Case: TC_001.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation}
 *
 * @verifyFunction{This test-case checks successful validatation of a reconciled attribute list}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 2
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6] set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 5
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852114}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:5
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LBV
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LBV
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LBV}
 *
 * @verifyFunction{This test-case checks error path when inputUnreconciledAttrListCount set to lower boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 2
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6] set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852117}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LW
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LW
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation_LW}
 *
 * @verifyFunction{This test-case checks error path when inputUnreconciledAttrListCount set to nominal value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 2
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6] set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 50
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852120}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:50
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x32
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x32
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation.UBV
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation.UBV
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreAttrListValidateReconciledWithLocks.NormalOperation.UBV}
 *
 * @verifyFunction{This test-case checks error path when inputUnreconciledAttrListCount set to upper boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 2
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6] set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to MAX
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852123}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:<<MAX>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:false
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:<<MAX>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:<<MAX>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:EXPECTED_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_primitivetype_in_input_attrlist
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_primitivetype_in_input_attrlist
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_primitivetype_in_input_attrlist}
 *
 * @verifyFunction{This test-case checks error path when Invalid primitive type in input attr list}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 2
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5] set to 0
 * test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6] set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreFillCpuPrimitiveInfo() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852126}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5..6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreAttrListValidateReconciledWithLocks.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreAttrListValidateReconciledWithLocks.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreAttrListValidateReconciledWithLocks.Mismatch_in_signalerPrimitiveInfo_vs_waiterPrimitiveInfo}
 *
 * @verifyFunction{This test-case checks error path when Mismatch in signalerPrimitiveInfo vs waiterPrimitiveInfo}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4
 * test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852129}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_SignalerPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_SignalerPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_SignalerPrimitiveInfo}
 *
 * @verifyFunction{This test-case checks error path when Too many primitives in SignalerPrimitiveInfo found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 4}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852132}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_WaiterPrimitiveInfo
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_WaiterPrimitiveInfo
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreAttrListValidateReconciledWithLocks.Too_many_primitives_in_WaiterPrimitiveInfo}
 *
 * @verifyFunction{This test-case checks error path when Too many primitives in WaiterPrimitiveInfo found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5] set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint
 * test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6] set to 1}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852135}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[6]:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledsignalerPrimitiveCount_notequals_to_unreconciledsignalerPrimitiveCount
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledsignalerPrimitiveCount_notequals_to_unreconciledsignalerPrimitiveCount
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledsignalerPrimitiveCount_notequals_to_unreconciledsignalerPrimitiveCount}
 *
 * @verifyFunction{This test-case checks eror path when reconciled attributed list does not satisfy the unreconciled lists requirements.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * test_objAttrList.objAttrList.state set to LwSciSyncCoreAttrListState_Reconciled
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_Empty
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 1
 * test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount set to 0}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852138}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLessThan.return:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciSyncCorePermLessThan
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList>>.coreAttrList );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLessThan.permB
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_WaitOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLessThan.permB>> == ( LwSciSyncAccessPerm_SignalOnly ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCorePermLEq.permA
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> == ( LwSciSyncAccessPerm_SignalOnly ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncCorePermLEq.permA>> == ( LwSciSyncAccessPerm_WaitOnly  ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.module>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleIsDup.otherModule>> == ( <<lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList>>.module ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrList) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_signalerprimitiveCount_ZERO
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_signalerprimitiveCount_ZERO
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_signalerprimitiveCount_ZERO}
 *
 * @verifyFunction{This test-case checks eror path when LwSciSyncInternalAttrKey_SignalerPrimitiveCount:reconciledattrList is zero}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false
 * test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7] set to LwSciSyncCoreAttrKeyState_SetLocked}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852141}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_SetLocked
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_cpu_access_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_cpu_access_permissions
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_cpu_access_permissions}
 *
 * @verifyFunction{This test-case checks eror path when Insufficient Reconciled cpu access permissions found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * LwSciSyncCoreAttrListPermLessThan() returns to false (second event)
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to true
 * test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess set to false}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852144}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_list_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_list_permissions
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCoreAttrListValidateReconciledWithLocks.Insufficient_Reconciled_list_permissions}
 *
 * @verifyFunction{This test-case checks eror path where LwSciSyncCoreAttrListPermLessThan() returns true (Insufficient Reconciled list permissions)}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitSignal
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * LwSciSyncCoreAttrListPermLessThan() returns to true}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852147}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciledattributelist_doesnot_has_atleast_waitpermissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciledattributelist_doesnot_has_atleast_waitpermissions
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciledattributelist_doesnot_has_atleast_waitpermissions}
 *
 * @verifyFunction{This test-case checks eror path when LwSciSyncCoreAttrListPermLessThan() returns true (Reconciled attri list must have at least wait permissions)}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User GLOBAL variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * test_unreconciledObjAttrList.numCoreAttrList set to 1
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm set to LwSciSyncAccessPerm_WaitOnly
 * test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * LwSciSyncCoreAttrListPermLessThan() returns to true}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852150}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (NULL );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciliation_failed
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_014.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciliation_failed
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncCoreAttrListValidateReconciledWithLocks.Reconciliation_failed}
 *
 * @verifyFunction{This test-case checks Reconciliation failure when there is Invalid permissions }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to true
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * isReconciledListValid[0] set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_ReconciliationFailed returned}
 *
 * @testcase{18852153}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid[0]:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncCoreAttrListValidateReconciledWithLocks.Modules_not_duplicated
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_015.LwSciSyncCoreAttrListValidateReconciledWithLocks.Modules_not_duplicated
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncCoreAttrListValidateReconciledWithLocks.Modules_not_duplicated}
 *
 * @verifyFunction{This test-case checks error path when attr list not belong to same module }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * 'isDup[0]' parameter of LwSciSyncCoreModuleIsDup() set to false
 * LwSciSyncCoreModuleIsDup() returns to LwSciError_Success
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852156}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_module_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_016.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_module_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledObjAttrList_module_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncCoreAttrListValidateReconciledWithLocks() panics when reconciledObjAttrList module is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * test_reconciledObjAttrList.module set to 0xFFFFFFFFFFFFFFFF (Invalid value)
 * LwSciSyncCoreModuleIsDup() panics
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreAttrListValidateReconciledWithLocks() panics}
 *
 * @testcase{18852159}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.module
<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_module_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_017.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_module_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_module_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncCoreAttrListValidateReconciledWithLocks() panics when unreconciledObjAttrList module is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_unreconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (2nd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * test_unreconciledObjAttrList.module set to 0xFFFFFFFFFFFFFFFF (Invalid value)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() panics
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments(reconciledObjAttrList)
 * LwSciSyncCoreModuleIsDup() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreAttrListValidateReconciledWithLocks() panics}
 *
 * @testcase{18852162}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.module
<<lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList>>.module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_NULL_NormalOperation
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_019.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_NULL_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncCoreAttrListValidateReconciledWithLocks.unreconciledObjAttrList_NULL_NormalOperation}
 *
 * @verifyFunction{This test-case checks error path when unreconciledObjAttrList is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success 
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * parameter 'newUnreconciledAttrList' of LwSciSyncCoreAttrListAppendUnreconciledWithLocks() set to valid memory
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to NULL(unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (unreconciledObjAttrList)
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_reconciledObjAttrList-User global variable  of type LwSciSyncCoreAttrListObj ) (3rd event)
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success (reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() returns to false
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments  (unreconciledObjAttrList)
 * LwSciCommonGetObjFromRef() receives correct arguments  (reconciledObjAttrList)
 * LwSciSyncCoreAttrListPermLessThan() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_Success returned}
 *
 * @testcase{18852168}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( NULL );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_020.LwSciSyncCoreAttrListValidateReconciledWithLocks.LwSciSyncCoreAttrListAppendUnreconciledWithLocks_InsufficientMemory
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_020.LwSciSyncCoreAttrListValidateReconciledWithLocks.LwSciSyncCoreAttrListAppendUnreconciledWithLocks_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_020.LwSciSyncCoreAttrListValidateReconciledWithLocks.LwSciSyncCoreAttrListAppendUnreconciledWithLocks_InsufficientMemory}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns error when any of the input LwSciSyncAttrLists are not unreconciled}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() returns to LwSciError_Success(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() returns to LwSciError_InsufficientMemory
 * }
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciSyncCoreAttrListAppendUnreconciledWithLocks() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_InsufficientMemory returned}
 *
 * @testcase{18852171}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_objAttrList.state:LwSciSyncCoreAttrListState_Reconciled
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList:<<malloc 2>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[5]:1
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.coreAttrList[1].attrs.valSize[6]:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_unreconciledObjAttrList.numCoreAttrList:2
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.signalerPrimitiveCount:0
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.keyState[7]:LwSciSyncCoreAttrKeyState_Empty
TEST.VALUE:lwscisync_attribute_reconcile.<<GLOBAL>>.test_reconciledObjAttrList.coreAttrList[0].attrs.valSize[5..6]:4
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:5
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.return:LwSciError_InsufficientMemory
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListCount:0x5
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x5
TEST.ATTRIBUTES:lwscisync_attribute_reconcile.LwSciSyncAttrListReconcile.inputCount:INPUT_BASE=16
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_unreconciledObjAttrList.objAttrList) );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = (&(test_reconciledObjAttrList.objAttrList) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.newUnreconciledAttrList>>[0][0].refAttrList) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray.inputUnreconciledAttrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListAppendUnreconciledWithLocks.inputUnreconciledAttrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_reconciledAttrList
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_022.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_reconciledAttrList
TEST.NOTES:
/**
 * @testname{TC_022.LwSciSyncCoreAttrListValidateReconciledWithLocks.Ilwalid_reconciledAttrList}
 *
 * @verifyFunction{This test-case checks LwSciSyncCoreAttrListValidateReconciledWithLocks() panics when reconciledObjAttrList is not a valid reference object}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success
 * LwSciSyncCoreAttrListValidate() returns to LwSciError_Success 
 * parameter 'objPtr' of LwSciCommonGetObjFromRef() set to valid memory of type 'LwSciSyncCoreAttrListObj'(test_objAttrList-User global variable  of type LwSciSyncCoreAttrListObj )
 * LwSciCommonGetObjFromRef() panics (objAttrList)
 * }
 *
 * @testinput{reconciledAttrList set to 0xFFFFFFFFFFFFFFFF
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments
 * LwSciSyncCoreAttrListValidate() receives correct arguments 
 * LwSciCommonGetObjFromRef() receives correct arguments(objAttrList)
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreAttrListValidateReconciledWithLocks() panics}
 *
 * @testcase{18852177}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList
<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_023.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid_NULL
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid_NULL}
 *
 * @verifyFunction{This test-case checks the error path when isReconciledListValid is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_Success}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to NULL
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852180}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray_Ilwalid
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_024.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray_Ilwalid}
 *
 * @verifyFunction{This test-case checks LwSciSyncCoreAttrListValidateReconciledWithLocks() panics when inputUnreconciledAttrListArray is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() panics}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to 0xFFFFFFFFFFFFFFFF
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreAttrListValidateReconciledWithLocks() panics}
 *
 * @testcase{18852183}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:1
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.allowEmpty:true
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray
<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_025.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount_Zero_inputUnreconciledAttrListArray_valid_memory
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_025.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount_Zero_inputUnreconciledAttrListArray_valid_memory
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount_Zero_inputUnreconciledAttrListArray_valid_memory}
 *
 * @verifyFunction{This test-case checks where LwSciSyncCoreValidateAttrListArray() returns error when inputUnreconciledAttrListCount set to zero,but inputUnreconciledAttrListArray is non-NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciSyncCoreValidateAttrListArray() returns to LwSciError_BadParameter}
 *
 * @testinput{reconciledAttrList set to valid memory
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 0
 * isReconciledListValid set to valid memory
 * acquireLocks set to true}
 *
 * @testbehavior{LwSciSyncCoreValidateAttrListArray() receives correct arguments
 * LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852186}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:0
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.acquireLocks:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x0
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList_NULL
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.NEW
TEST.NAME:TC_026.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList_NULL
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList_NULL}
 *
 * @verifyFunction{This test-case checks the error path when reconciledAttrList is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{reconciledAttrList set to NULL
 * inputUnreconciledAttrListArray set to valid memory
 * inputUnreconciledAttrListCount set to 1
 * isReconciledListValid set to valid memory
 * acquireLocks set to false}
 *
 * @testbehavior{LwSciSyncAttrListFree() receives correct arguments
 * LwSciError_BadParameter returned}
 *
 * @testcase{18852189}
 *
 * @verify{18844338}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList:<<null>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListCount:1
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.acquireLocks:false
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.isReconciledListValid:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.return:LwSciError_Success
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncAttrListFree.attrList:<<null>>
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListCount:0x1
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
  uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncAttrListFree
  lwscisync_attribute_reconcile.c.LwSciSyncCoreAttrListValidateReconciledWithLocks
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &(test_objAttrList.objAttrList) );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.reconciledAttrList>>[0].refAttrList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray.attrListArray[0]
{{ <<uut_prototype_stubs.LwSciSyncCoreValidateAttrListArray.attrListArray>>[0] == ( <<lwscisync_attribute_reconcile.LwSciSyncCoreAttrListValidateReconciledWithLocks.inputUnreconciledAttrListArray>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreFillCpuPrimitiveInfo

-- Test Case: TC_001.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList_Ilwalid_panics
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreFillCpuPrimitiveInfo
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList_Ilwalid_panics
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList_Ilwalid_panics}
 *
 * @verifyFunction{This test-case checks  LwSciSyncCoreFillCpuPrimitiveInfo() panics when objAttrList is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{objAttrList set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.
 * LwSciSyncCoreFillCpuPrimitiveInfo() panics}
 *
 * @testcase{18852192}
 *
 * @verify{18844341}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList:<<null>>
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_waiter_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreFillCpuPrimitiveInfo
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_waiter_permissions
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_waiter_permissions}
 *
 * @verifyFunction{This test-case copies Cpu Primitives when needCpuAccess set to TRUE and actualPerm set to LwSciSyncAccessPerm_WaitOnly}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{objAttrList set to valid memory
 * objAttrList[0].numCoreAttrList set to 1
 * objAttrList[0].coreAttrList[0].attrs.needCpuAccess set to TRUE
 * objAttrList[0].coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_WaitOnly
 * objAttrList[0].coreAttrList[0].attrs.waiterPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore}
 *
 * @testbehavior{objAttrList[0].coreAttrList[0].attrs.valSize[7] set to 4
 * LwSciSyncCoreCopyCpuPrimitives() receives correct arguments}
 *
 * @testcase{18852195}
 *
 * @verify{18844341}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.waiterPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.valSize[6]:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_signaler_permissions
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreFillCpuPrimitiveInfo
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_signaler_permissions
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_True_signaler_permissions}
 *
 * @verifyFunction{This test-case copies Cpu Primitives when needCpuAccess set to TRUE and actualPerm set to LwSciSyncAccessPerm_SignalOnly}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{objAttrList set to valid memory
 * objAttrList[0].numCoreAttrList set to 1
 * objAttrList[0].coreAttrList[0].attrs.needCpuAccess set to TRUE
 * objAttrList[0].coreAttrList[0].attrs.actualPerm set to LwSciSyncAccessPerm_SignalOnly
 * objAttrList[0].coreAttrList[0].attrs.signalerPrimitiveInfo[0] set to LwSciSyncInternalAttrValPrimitiveType_Syncpoint}
 *
 * @testbehavior{objAttrList[0].coreAttrList[0].attrs.signalerPrimitiveCount set to 1
 * objAttrList[0].coreAttrList[0].attrs.valSize[5] set to 4
 * objAttrList[0].coreAttrList[0].attrs.valSize[6] set to 4
 * LwSciSyncCoreCopyCpuPrimitives() receives correct arguments}
 *
 * @testcase{18852198}
 *
 * @verify{18844341}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.needCpuAccess:true
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.actualPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.signalerPrimitiveInfo[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.signalerPrimitiveCount:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.valSize[5]:4
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.valSize[7]:4
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
  uut_prototype_stubs.LwSciSyncCoreCopyCpuPrimitives
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_False
TEST.UNIT:lwscisync_attribute_reconcile
TEST.SUBPROGRAM:LwSciSyncCoreFillCpuPrimitiveInfo
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_False
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreFillCpuPrimitiveInfo.needCpuAccess_False}
 *
 * @verifyFunction{This test-case checks NoOperation when needCpuAccess set to FALSE}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{objAttrList set to valid memory
 * objAttrList[0].numCoreAttrList set to 1
 * objAttrList[0].coreAttrList[0].attrs.needCpuAccess set to FALSE}
 *
 * @testbehavior{returns nothing}
 *
 * @testcase{18852201}
 *
 * @verify{18844341}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList:<<malloc 1>>
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.needCpuAccess:false
TEST.VALUE:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].numCoreAttrList:1
TEST.EXPECTED:lwscisync_attribute_reconcile.LwSciSyncCoreFillCpuPrimitiveInfo.objAttrList[0].coreAttrList[0].attrs.needCpuAccess:false
TEST.FLOW
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
  lwscisync_attribute_reconcile.c.LwSciSyncCoreFillCpuPrimitiveInfo
TEST.END_FLOW
TEST.END
