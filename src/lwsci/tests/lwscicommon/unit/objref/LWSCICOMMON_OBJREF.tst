-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
--
-- Environment    : LWSCICOMMON_OBJREF
-- Unit(s) Under Test: lwscicommon_objref
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Subprogram: LwSciCommonAllocObjWithRef

-- Test Case: TC_001.LwSciCommonAllocObjWithRef.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_001.LwSciCommonAllocObjWithRef.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonAllocObjWithRef.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Success use case when allocates memory for user specific reference container of requested size.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{test_lwSciObj writes memory into 'test_lwSciRef.objPtr'.
 * LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_Success.
 * LwSciCommonCalloc() returns memory to test_lwSciObj.
 * LwSciCommonMutexCreate() returns LwSciError_Success.}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciObj).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciObj.objLock.
 * objPtr pointing to valid address.
 * refPtr pointing to valid address.
 * returns LwSciError_Success.}
 *
 * @testcase{18859896}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_inputRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.refCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:1
TEST.EXPECTED:lwscicommon_objref.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = ( sizeof(LwSciObj) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr
{{ *<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr
{{ *<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.size
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.size == ( sizeof(LwSciRef) ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_002.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to system lacks resource other than memory to initialize the lock.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{test_lwSciObj writes memory into 'test_lwSciRef.objPtr'.
 * LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_Success.
 * LwSciCommonCalloc() returns memory to test_lwSciObj.
 * LwSciCommonMutexCreate() returns LwSciError_ResourceError.
 * LwSciCommonMutexDestroy()returns LwSciError_Success.}
 *
 * @testinput{objSize driven as SIZE_MAX.
 * refSize  driven as SIZE_MAX.
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciObj).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonFree()receives ptr as test_lwSciRef.objPtr.
 * LwSciCommonMutexDestroy() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonFree()receives ptr as test_lwSciRef.
 * returns LwSciError_ResourceError.}
 *
 * @testcase{18859881}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_inputRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize:<<MAX>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize:<<MAX>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.size:<<MAX>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.refCount:0
TEST.EXPECTED:lwscicommon_objref.LwSciCommonAllocObjWithRef.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonMutexCreate.return>> = ( LwSciError_Success );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonMutexCreate.return>> = ( LwSciError_ResourceError );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.objPtr ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexDestroy.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_003.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to memory allocation failed.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_Success.
 * LwSciCommonCalloc() returns NULL.
 * LwSciCommonMutexDestroy()returns LwSciError_Success.}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciObj).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexDestroy() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonFree()receives ptr as test_lwSciRef.
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859875}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_inputRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.EXPECTED:lwscicommon_objref.LwSciCommonAllocObjWithRef.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexDestroy.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = ( sizeof(LwSciObj) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_004.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonAllocObjWithRef.Failure_due_to_resource_error}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to system lacks resource other than memory to initialize the lock.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_ResourceError.}
 *
 * @testinput{objSize driven as 123.
 * refSize  driven as 123.
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciObj).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonFree()receives ptr as test_lwSciRef.
 * returns LwSciError_ResourceError.}
 *
 * @testcase{18859878}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_inputRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize:123
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize:123
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonMutexCreate.return:LwSciError_ResourceError
TEST.EXPECTED:lwscicommon_objref.LwSciCommonAllocObjWithRef.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_005.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonAllocObjWithRef.Failure_due_to_memory_allocation_failed}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to memory allocation failed.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL.}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciObj).
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859872}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_inputRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscicommon_objref.LwSciCommonAllocObjWithRef.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = ( sizeof(LwSciObj) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciCommonAllocObjWithRef.Failure_due_to_refPtr_is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_006.LwSciCommonAllocObjWithRef.Failure_due_to_refPtr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonAllocObjWithRef.Failure_due_to_refPtr_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to refPtr is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonAllocObjWithRef panics.}
 *
 * @testcase{18859890}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = (sizeof(LwSciObj));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef));

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciCommonAllocObjWithRef.Failure_due_to_objPtr_is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_007.LwSciCommonAllocObjWithRef.Failure_due_to_objPtr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonAllocObjWithRef.Failure_due_to_objPtr_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to objPtr is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonAllocObjWithRef panics.}
 *
 * @testcase{18859884}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<null>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = (sizeof(LwSciObj));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef));

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciCommonAllocObjWithRef.Failure_due_to_objSize_is_too_small
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_008.LwSciCommonAllocObjWithRef.Failure_due_to_objSize_is_too_small
TEST.NOTES:
/**
 * @testname{TC_008.LwSciCommonAllocObjWithRef.Failure_due_to_objSize_is_too_small}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to refSize being smaller than sizeof(LwSciObj).}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{objSize driven as (sizeof(LwSciObj)-1).
 * refSize  driven as sizeof(LwSciRef).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonAllocObjWithRef panics.}
 *
 * @testcase{18859887}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = (sizeof(LwSciObj) - 1);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef));

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciCommonAllocObjWithRef.Failure_due_to_refSize_is_too_small
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonAllocObjWithRef
TEST.NEW
TEST.NAME:TC_009.LwSciCommonAllocObjWithRef.Failure_due_to_refSize_is_too_small
TEST.NOTES:
/**
 * @testname{TC_009.LwSciCommonAllocObjWithRef.Failure_due_to_refSize_is_too_small}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonAllocObjWithRef() - Failure due to refSize being smaller than sizeof(LwSciRef).}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{objSize driven as sizeof(LwSciObj).
 * refSize  driven as (sizeof(LwSciRef)-1).
 * objPtr updated to valid memory.
 * refPtr updated to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonAllocObjWithRef panics.}
 *
 * @testcase{18859893}
 *
 * @verify{18851091}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refPtr:<<malloc 1>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonAllocObjWithRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.objSize>> = (sizeof(LwSciObj));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize
<<lwscicommon_objref.LwSciCommonAllocObjWithRef.refSize>> = ( sizeof(LwSciRef) - 1);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonDuplicateRef

-- Test Case: TC_001.LwSciCommonDuplicateRef.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_001.LwSciCommonDuplicateRef.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonDuplicateRef.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Success use case when creates new LwSciRef structure pointed by newRef.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{oldRef.objptr set to test_lwSciObj.
 * LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_Success.
 * LwSciCommonMutexLock()returns LwSciError_Success.
 * LwSciCommonMutexLock()returns LwSciError_Success.
 * LwSciCommonMutexUnlock()returns LwSciError_Success.
 * LwSciCommonMutexUnlock()returns LwSciError_Success.}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * oldRef.size driven as sizeof(LwSciRef).
 * newRef updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonMutexLock() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciRef.RefLock
 * newRef pointing to valid address of test_lwSciRef.
 * returns LwSciError_Success.}
 *
 * @testcase{18859911}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.refCount:1
TEST.EXPECTED:lwscicommon_objref.LwSciCommonDuplicateRef.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].size
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].objPtr
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef.newRef[0]
{{ <<lwscicommon_objref.LwSciCommonDuplicateRef.newRef>>[0] == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.size
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.size == ( sizeof(LwSciRef) ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.objPtr
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.objPtr == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonDuplicateRef.Failure_due_to_obj_refcount_Is_MAX
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_002.LwSciCommonDuplicateRef.Failure_due_to_obj_refcount_Is_MAX
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonDuplicateRef.Failure_due_to_obj_refcount_Is_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to refcount is INT32_MAX.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{oldRef.objptr set to test_lwSciObj.
 * test_lwSciObj.refCount set to SIZE_MAX.
 * LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_Success.
 * LwSciCommonMutexLock()returns LwSciError_Success.
 * LwSciCommonMutexLock()returns LwSciError_Success.
 * LwSciCommonMutexUnlock()returns LwSciError_Success.
 * LwSciCommonMutexUnlock()returns LwSciError_Success.}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * oldRef.size driven as sizeof(LwSciRef).
 * newRef updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonMutexLock() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciRef.RefLock.
 * returns LwSciError_IlwalidState.}
 *
 * @testcase{22060242}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:<<MAX>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.refCount:0
TEST.EXPECTED:lwscicommon_objref.LwSciCommonDuplicateRef.return:LwSciError_IlwalidState
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].size
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].objPtr
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.size
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.size == ( sizeof(LwSciRef) ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef.objPtr
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.objPtr == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonDuplicateRef.Failure_due_to_resource_error
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_003.LwSciCommonDuplicateRef.Failure_due_to_resource_error
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonDuplicateRef.Failure_due_to_resource_error}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to resource error.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to test_lwSciRef.
 * LwSciCommonMutexCreate() returns LwSciError_ResourceError.}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * oldRef.size driven as sizeof(LwSciRef).
 * newRef updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * LwSciCommonMutexCreate() receives mutex as test_lwSciRef.RefLock.
 * LwSciCommonFree() receives mutex as test_lwSciRef.
 * returns LwSciError_ResourceError.}
 *
 * @testcase{18859905}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonMutexCreate.return:LwSciError_ResourceError
TEST.EXPECTED:lwscicommon_objref.LwSciCommonDuplicateRef.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMutexCreate
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexCreate.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexCreate.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciRef>>.refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].size
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonDuplicateRef.Failure_due_to_memory_allocation_failed
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_004.LwSciCommonDuplicateRef.Failure_due_to_memory_allocation_failed
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonDuplicateRef.Failure_due_to_memory_allocation_failed}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to memory allocation failed.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL.}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * oldRef.size driven as sizeof(LwSciRef).
 * newRef updated to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives numItems as 1.
 * LwSciCommonCalloc() receives size as sizeof(LwSciRef).
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859899}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<malloc 1>>
TEST.EXPECTED:lwscicommon_objref.LwSciCommonDuplicateRef.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
  uut_prototype_stubs.LwSciCommonCalloc
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( <<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef.oldRef[0].size
<<lwscicommon_objref.LwSciCommonDuplicateRef.oldRef>>[0].size = ( sizeof(LwSciRef) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonDuplicateRef.Failure_due_to_oldRef_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_005.LwSciCommonDuplicateRef.Failure_due_to_oldRef_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonDuplicateRef.Failure_due_to_oldRef_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to oldRef.magicNUmber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as 1.
 * newRef updated to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonDuplicateRef panics.}
 *
 * @testcase{22060246}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:1
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<malloc 1>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciCommonDuplicateRef.Failure_due_to_oldRef_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_006.LwSciCommonDuplicateRef.Failure_due_to_oldRef_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonDuplicateRef.Failure_due_to_oldRef_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to oldRef is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{oldRef updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonDuplicateRef panics.}
 *
 * @testcase{18859908}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciCommonDuplicateRef.Failure_due_to_newRef_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonDuplicateRef
TEST.NEW
TEST.NAME:TC_007.LwSciCommonDuplicateRef.Failure_due_to_newRef_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonDuplicateRef.Failure_due_to_newRef_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonDuplicateRef() - Failure due to newRef is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{oldRef updated to valid memory.
 * oldRef.magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * newRef updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonDuplicateRef panics.}
 *
 * @testcase{18859902}
 *
 * @verify{18851109}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.oldRef[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonDuplicateRef.newRef:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonDuplicateRef
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonFreeObjAndRef

-- Test Case: TC_001.LwSciCommonFreeObjAndRef.Failed_due_to_ref_refCount_Is_0
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_001.LwSciCommonFreeObjAndRef.Failed_due_to_ref_refCount_Is_0
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonFreeObjAndRef.Failed_due_to_ref_refCount_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Panics due to obj_refcount is 0.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * objCleanupCallback set to valid function address of objCleanupStub.
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.}
 *
 * @testcase{18859914}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonFreeObjAndRef.Success_use_case2_ref_refcount_Is_1_obj_refcount_is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_002.LwSciCommonFreeObjAndRef.Success_use_case2_ref_refcount_Is_1_obj_refcount_is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonFreeObjAndRef.Success_use_case2_ref_refcount_Is_1_obj_refcount_is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Success_use_case2 when ref_refcount and obj_refcount is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.
 * test_lwSciObj.refCount set to 1.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * ref[0].refCount driven as 1.
 * objCleanupCallback set to valid function address of objCleanupStub.
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexDestroy() receives mutex as ref[0].refLock.
 * LwSciCommonFree() receives ptr as ref.
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexDestroy() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonFree() receives ptr as ref.objPtr.}
 *
 * @testcase{18859920}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:1
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].refCount:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexDestroy.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonFreeObjAndRef.Success_use_case3_ref_refcount_Is_2_obj_refcount_is_2
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_003.LwSciCommonFreeObjAndRef.Success_use_case3_ref_refcount_Is_2_obj_refcount_is_2
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonFreeObjAndRef.Success_use_case3_ref_refcount_Is_2_obj_refcount_is_2}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Success_use_case3 when ref_refcount and obj_refcount is 2.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.
 * test_lwSciObj.refCount set to 2.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * ref[0].refCount driven as 2.
 * objCleanupCallback set to valid function address of objCleanupStub.
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.}
 *
 * @testcase{18859923}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:2
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].refCount:2
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonFreeObjAndRef.Success_use_case4_objCleanupCallback_Is_NULL_refCleanupCallback_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_004.LwSciCommonFreeObjAndRef.Success_use_case4_objCleanupCallback_Is_NULL_refCleanupCallback_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonFreeObjAndRef.Success_use_case4_objCleanupCallback_Is_NULL_refCleanupCallback_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Success_use_case4 when ref_refcount and obj_refcount is 2.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.
 * test_lwSciObj.refCount set to 1.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * ref[0].refCount driven as 1.
 * objCleanupCallback set to NULL.
 * refCleanupCallback set to NULL.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock.
 * LwSciCommonMutexDestroy() receives mutex as ref[0].refLock.
 * LwSciCommonFree() receives ptr as ref.
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexDestroy() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonFree() receives ptr as ref.objPtr.}
 *
 * @testcase{22060250}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:1
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].refCount:1
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexDestroy.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonFreeObjAndRef.Failed_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_005.LwSciCommonFreeObjAndRef.Failed_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonFreeObjAndRef.Failed_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as 1.
 * ref[0].refCount driven as 1.
 * objCleanupCallback set to valid function address of objCleanupStub..
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonFreeObjAndRef panics.}
 *
 * @testcase{18859917}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciCommonFreeObjAndRef.Failed_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_006.LwSciCommonFreeObjAndRef.Failed_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonFreeObjAndRef.Failed_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to NULL.
 * objCleanupCallback set to valid function address.
 * refCleanupCallback set to valid function address.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonFreeObjAndRef panics.}
 *
 * @testcase{22060253}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciCommonFreeObjAndRef.Failed_due_to_ref_objPtr_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_007.LwSciCommonFreeObjAndRef.Failed_due_to_ref_objPtr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonFreeObjAndRef.Failed_due_to_ref_objPtr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Failure due to ref.objPtr is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * ref[0].refCount driven as 1.
 * ref[0].objPtr set to NULL.
 * objCleanupCallback set to valid function address of objCleanupStub..
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonFreeObjAndRef panics.}
 *
 * @testcase{22060255}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].objPtr:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciCommonFreeObjAndRef.Failed_due_to_obj_refCount_Is_0
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonFreeObjAndRef
TEST.NEW
TEST.NAME:TC_008.LwSciCommonFreeObjAndRef.Failed_due_to_obj_refCount_Is_0
TEST.NOTES:
/**
 * @testname{TC_008.LwSciCommonFreeObjAndRef.Failed_due_to_obj_refCount_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Panics due to obj_refcount is 0.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * objCleanupCallback set to valid function address of objCleanupStub.
 * refCleanupCallback set to valid function address of refCleanupStub.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.}
 *
 * @testcase{22060258}
 *
 * @verify{18851094}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:0
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref[0].refCount:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexDestroy
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexDestroy.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexDestroy.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.objCleanupCallback>> = (&objCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback
<<lwscicommon_objref.LwSciCommonFreeObjAndRef.refCleanupCallback>> = (&refCleanupStub);

TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonGetObjFromRef

-- Test Case: TC_001.LwSciCommonGetObjFromRef.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonGetObjFromRef
TEST.NEW
TEST.NAME:TC_001.LwSciCommonGetObjFromRef.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonGetObjFromRef.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Success use case when retrieves LwSciObj object associated with the input LwSciRef object successfully.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * objPtr set to valid memory.}
 *
 * @testbehavior{objPtr is pointing to valid address.}
 *
 * @testcase{18859932}
 *
 * @verify{18851100}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.objPtr:<<malloc 1>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonGetObjFromRef
  lwscicommon_objref.c.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscicommon_objref.LwSciCommonGetObjFromRef.objPtr
{{ *<<lwscicommon_objref.LwSciCommonGetObjFromRef.objPtr>> == ( <<lwscicommon_objref.LwSciCommonGetObjFromRef.ref>>[0].objPtr ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonGetObjFromRef.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonGetObjFromRef
TEST.NEW
TEST.NAME:TC_002.LwSciCommonGetObjFromRef.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonGetObjFromRef.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonGetObjFromRef panics.}
 *
 * @testcase{22060261}
 *
 * @verify{18851100}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonGetObjFromRef.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonGetObjFromRef
TEST.NEW
TEST.NAME:TC_003.LwSciCommonGetObjFromRef.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonGetObjFromRef.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFreeObjAndRef() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonGetObjFromRef panics.}
 *
 * @testcase{22060263}
 *
 * @verify{18851100}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonGetObjFromRef.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonGetObjFromRef
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonIncrAllRefCounts

-- Test Case: TC_001.LwSciCommonIncrAllRefCounts.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonIncrAllRefCounts
TEST.NEW
TEST.NAME:TC_001.LwSciCommonIncrAllRefCounts.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonIncrAllRefCounts.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonIncrAllRefCounts() - Success use case when increments reference count of LwSciRef object and LwSciObj object by one successfully.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock
 * returns LwSciError_Success.}
 *
 * @testcase{18859941}
 *
 * @verify{18851097}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:lwscicommon_objref.LwSciCommonIncrAllRefCounts.return:LwSciError_Success
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_refcount_Is_MAX
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonIncrAllRefCounts
TEST.NEW
TEST.NAME:TC_002.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_refcount_Is_MAX
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_refcount_Is_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonIncrAllRefCounts() - Failure due to ref.refCount is MAX.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.
 * ref[0].refCount driven as SIZE_MAX.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock
 * returns LwSciError_IlwalidState.}
 *
 * @testcase{18859935}
 *
 * @verify{18851097}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref[0].refCount:<<MAX>>
TEST.EXPECTED:lwscicommon_objref.LwSciCommonIncrAllRefCounts.return:LwSciError_IlwalidState
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonIncrAllRefCounts.Failure_due_to_obj_refcount_Is_MAX
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonIncrAllRefCounts
TEST.NEW
TEST.NAME:TC_003.LwSciCommonIncrAllRefCounts.Failure_due_to_obj_refcount_Is_MAX
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonIncrAllRefCounts.Failure_due_to_obj_refcount_Is_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonIncrAllRefCounts() - Failure due to test_lwSciObj.refCount is MAX.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr updated to test_lwSciObj.
 * test_lwSciObj.refCount updated to SIZE_MAX.}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock
 * LwSciCommonMutexLock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as test_lwSciObj.objLock.
 * LwSciCommonMutexUnlock() receives mutex as ref[0].refLock
 * returns LwSciError_IlwalidState.}
 *
 * @testcase{22060266}
 *
 * @verify{18851097}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj.refCount:<<MAX>>
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.EXPECTED:lwscicommon_objref.LwSciCommonIncrAllRefCounts.return:LwSciError_IlwalidState
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexLock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].refLock) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonIncrAllRefCounts
TEST.NEW
TEST.NAME:TC_004.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonIncrAllRefCounts() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref set to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonIncrAllRefCounts panics.}
 *
 * @testcase{18859938}
 *
 * @verify{18851097}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonIncrAllRefCounts
TEST.NEW
TEST.NAME:TC_005.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonIncrAllRefCounts.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonIncrAllRefCounts() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonIncrAllRefCounts panics.}
 *
 * @testcase{22060269}
 *
 * @verify{18851097}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonIncrAllRefCounts.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonIncrAllRefCounts
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonObjLock

-- Test Case: TC_001.LwSciCommonObjLock.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjLock
TEST.NEW
TEST.NAME:TC_001.LwSciCommonObjLock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonObjLock.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjLock() - Success use case when acquires thread synchronization lock on the LwSciObj object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr set to test_lwSciObj.}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.}
 *
 * @testcase{18859944}
 *
 * @verify{18851112}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjLock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonObjLock.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonMutexLock
  lwscicommon_objref.c.LwSciCommonObjLock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonObjLock.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonObjLock.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonObjLock.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjLock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonObjLock.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonObjLock.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjLock() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr set to test_lwSciObj.}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonObjLock panics.}
 *
 * @testcase{22060272}
 *
 * @verify{18851112}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjLock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonObjLock.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjLock
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonObjLock.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjLock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonObjLock.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonObjLock.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjLock() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonObjLock panics.}
 *
 * @testcase{18859947}
 *
 * @verify{18851112}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjLock.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjLock
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonObjUnlock

-- Test Case: TC_001.LwSciCommonObjUnlock.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjUnlock
TEST.NEW
TEST.NAME:TC_001.LwSciCommonObjUnlock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonObjUnlock.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjUnlock() - Success use case when acquires thread synchronization lock on the LwSciObj object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr set to test_lwSciObj.}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.}
 *
 * @testcase{18859950}
 *
 * @verify{18851115}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjUnlock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonObjUnlock.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonObjUnlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>>.objLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_objref.LwSciCommonObjUnlock.ref.ref[0].objPtr
<<lwscicommon_objref.LwSciCommonObjUnlock.ref>>[0].objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_lwSciObj>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonObjUnlock.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjUnlock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonObjUnlock.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonObjUnlock.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjUnlock() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{ref[0].objPtr set to test_lwSciObj.}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonObjUnlock panics.}
 *
 * @testcase{22060275}
 *
 * @verify{18851115}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjUnlock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonObjUnlock.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjUnlock
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonObjUnlock.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonObjUnlock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonObjUnlock.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonObjUnlock.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonObjUnlock() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonObjUnlock panics.}
 *
 * @testcase{18859953}
 *
 * @verify{18851115}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonObjUnlock.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonObjUnlock
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonRefLock

-- Test Case: TC_001.LwSciCommonRefLock.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefLock
TEST.NEW
TEST.NAME:TC_001.LwSciCommonRefLock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonRefLock.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefLock() - Success use case when acquires thread synchronization lock on the input LwSciRef object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.}
 *
 * @testcase{18859956}
 *
 * @verify{18851103}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefLock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonRefLock.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefLock
  uut_prototype_stubs.LwSciCommonMutexLock
  lwscicommon_objref.c.LwSciCommonRefLock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexLock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexLock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonRefLock.ref>>[0].refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonRefLock.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefLock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonRefLock.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonRefLock.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefLock() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonRefLock panics.}
 *
 * @testcase{22060278}
 *
 * @verify{18851103}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefLock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonRefLock.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefLock
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonRefLock.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefLock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonRefLock.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonRefLock.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefLock() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonRefLock panics.}
 *
 * @testcase{18859959}
 *
 * @verify{18851103}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefLock.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefLock
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonRefUnlock

-- Test Case: TC_001.LwSciCommonRefUnlock.Success_use_case
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefUnlock
TEST.NEW
TEST.NAME:TC_001.LwSciCommonRefUnlock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonRefUnlock.Failure_due_to_ref_refcount_Is_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefUnlock() - Success use case when acquires thread synchronization lock on the input LwSciRef object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as LW_SCI_COMMON_REF_MAGIC.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives mutex as ref[0].refLock.}
 *
 * @testcase{18859962}
 *
 * @verify{18851106}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefUnlock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonRefUnlock.ref[0].magicNumber:MACRO=LW_SCI_COMMON_REF_MAGIC
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefUnlock
  uut_prototype_stubs.LwSciCommonMutexUnlock
  lwscicommon_objref.c.LwSciCommonRefUnlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMutexUnlock.mutex
{{ <<uut_prototype_stubs.LwSciCommonMutexUnlock.mutex>> == ( &<<lwscicommon_objref.LwSciCommonRefUnlock.ref>>[0].refLock ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonRefUnlock.Failure_due_to_ref_magicNumber_Is_1
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefUnlock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonRefUnlock.Failure_due_to_ref_magicNumber_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonRefUnlock.Failure_due_to_ref_magicNumber_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefUnlock() - Failure due to ref.magicNumber is 1.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to valid memory.
 * ref[0].magicNumber driven as 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonRefUnlock panics.}
 *
 * @testcase{18859965}
 *
 * @verify{18851106}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefUnlock.ref:<<malloc 1>>
TEST.VALUE:lwscicommon_objref.LwSciCommonRefUnlock.ref[0].magicNumber:1
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefUnlock
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonRefUnlock.Failure_due_to_ref_Is_NULL
TEST.UNIT:lwscicommon_objref
TEST.SUBPROGRAM:LwSciCommonRefUnlock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonRefUnlock.Failure_due_to_ref_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonRefUnlock.Failure_due_to_ref_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonRefUnlock() - Failure due to ref is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ref updated to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called with no args.
 * LwSciCommonRefUnlock panics.}
 *
 * @testcase{22060281}
 *
 * @verify{18851106}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_objref.LwSciCommonRefUnlock.ref:<<null>>
TEST.FLOW
  lwscicommon_objref.c.LwSciCommonRefUnlock
TEST.END_FLOW
TEST.END
