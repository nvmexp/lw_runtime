-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCICOMMON_PLATFORM_UTILITIES
-- Unit(s) Under Test: lwscicommon_aarch64 lwscicommon_libc lwscicommon_posix
-- 
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscicommon_libc

-- Subprogram: LwSciCommonCalloc

-- Test Case: TC_001.LwSciCommonCalloc.Success_use_case
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonCalloc
TEST.NEW
TEST.NAME:TC_001.LwSciCommonCalloc.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonCalloc.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonCalloc() - Success case when successfully allocated memory.}
 *
 * @casederiv{Analysis of Requirements.
 *  Generation and Analysis of Equivalence Classes.}
 *
 * @testsetup{calloc() returns memory to 'test_ptr2'.}
 *
 * @testinput{numItems driven as '6'.
 *  size driven as '5'.}
 *
 * @testbehavior{calloc() receives '__nmemb' as '1'.
 *  calloc() receives '__size' as '18'.
 *  'test_ptr[0].magic' pointing to valid value.
 *  'test_ptr[0].allocSize' pointing to valid value.
 *  returns memory.}
 *
 * @testcase{18859971}
 *
 * @verify{18851223}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.calloc
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.numItems:6
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.size:5
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:0x10293847
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.allocSize:30
TEST.EXPECTED:uut_prototype_stubs.calloc.__nmemb:1
TEST.EXPECTED:uut_prototype_stubs.calloc.__size:46
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonCalloc
  uut_prototype_stubs.calloc
  lwscicommon_libc.c.LwSciCommonCalloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.calloc.return
<<uut_prototype_stubs.calloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonCalloc.return
{{ <<lwscicommon_libc.LwSciCommonCalloc.return>> != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonCalloc.Failed_due_to_Allocate_memory
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonCalloc
TEST.NEW
TEST.NAME:TC_002.LwSciCommonCalloc.Failed_due_to_Allocate_memory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonCalloc.Failed_due_to_Allocate_memory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonCalloc() - Failure case when failed to allocate memory.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{calloc() returns NULL.}
 *
 * @testinput{numItems driven as '0'.
 *  size driven as '0'.}
 *
 * @testbehavior{calloc() receives '__nmemb' as '1'.
 *  calloc() receives '__size' as '16'.
 *  returns NULL.}
 *
 * @testcase{18859974}
 *
 * @verify{18851223}
 */
 
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.calloc
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.numItems:1
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.size:1
TEST.EXPECTED:uut_prototype_stubs.calloc.__nmemb:1
TEST.EXPECTED:uut_prototype_stubs.calloc.__size:17
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonCalloc
  uut_prototype_stubs.calloc
  lwscicommon_libc.c.LwSciCommonCalloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.calloc.return
<<uut_prototype_stubs.calloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonCalloc.return
{{ <<lwscicommon_libc.LwSciCommonCalloc.return>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonCalloc.Failed_due_to_size_Is_SIZE_MAX_numitems_Is_SIZE_MAX
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonCalloc
TEST.NEW
TEST.NAME:TC_003.LwSciCommonCalloc.Failed_due_to_size_Is_SIZE_MAX_numitems_Is_SIZE_MAX
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonCalloc.Failed_due_to_size_Is_SIZE_MAX_numitems_Is_SIZE_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonCalloc() - Failed due to size and numitems is SIZE_MAX.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{numItems driven as 'SIZE_MAX'.
 *  size driven as 'SIZE_MAX'.}
 *
 * @testbehavior{returns NULL.}
 *
 * @testcase{18859977}
 *
 * @verify{18851223}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.numItems:<<MAX>>
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.size:<<MAX>>
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonCalloc
  lwscicommon_libc.c.LwSciCommonCalloc
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonCalloc.return
{{ <<lwscicommon_libc.LwSciCommonCalloc.return>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonCalloc.Failed_due_to_size_Is_0
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonCalloc
TEST.NEW
TEST.NAME:TC_004.LwSciCommonCalloc.Failed_due_to_size_Is_0
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonCalloc.Failed_due_to_size_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonCalloc() - Failed due to size is 0.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{numItems driven as '1'.
 *  size driven as '0'.
 *
 * @testbehavior{returns NULL.}
 *
 * @testcase{22060293}
 *
 * @verify{18851223}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.numItems:1
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.size:0
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonCalloc
  lwscicommon_libc.c.LwSciCommonCalloc
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonCalloc.return
{{ <<lwscicommon_libc.LwSciCommonCalloc.return>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonCalloc.Failed_due_to_numItems_Is_0
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonCalloc
TEST.NEW
TEST.NAME:TC_005.LwSciCommonCalloc.Failed_due_to_numItems_Is_0
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonCalloc.Failed_due_to_numItems_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonCalloc() - Failed due to numItems is 0.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{numItems driven as '0'.
 *  size driven as '1'.}
 *
 * @testbehavior{returns NULL.}
 *
 * @testcase{22060296}
 *
 * @verify{18851223}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.numItems:0
TEST.VALUE:lwscicommon_libc.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonCalloc
  lwscicommon_libc.c.LwSciCommonCalloc
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonCalloc.return
{{ <<lwscicommon_libc.LwSciCommonCalloc.return>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonFree

-- Test Case: TC_001.LwSciCommonFree.Success_use_case
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonFree
TEST.NEW
TEST.NAME:TC_001.LwSciCommonFree.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonFree.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Success case when pointer memory is freed successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{'test_ptr[0].magic' set to 'LWSCICOMMON_ALLOC_MAGIC'.
 *  'test_ptr[0].allocSize' set to '1'.}
 *
 * @testinput{'ptr' set to valid memory.}
 *
 * @testbehavior{memset() receives '__s' as valid address.
 *  memset() receives '__c' as '0'.
 *  memset() receives '__n' as valid address.
 *  'test_ptr[0].magic' pointing to valid value.
 *  'test_ptr[0].allocSize' pointing to valid value.
 *  free() receives '__ptr' as valid address.}
 *
 * @testcase{18859980}
 *
 * @verify{18851226}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.free
TEST.STUB:uut_prototype_stubs.memset
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:MACRO=LWSCICOMMON_ALLOC_MAGIC
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:0x0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:0x0
TEST.EXPECTED:uut_prototype_stubs.memset.__c:0x0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonFree
  uut_prototype_stubs.memset
  uut_prototype_stubs.free
  lwscicommon_libc.c.LwSciCommonFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.free.__ptr
{{ <<uut_prototype_stubs.free.__ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.memset.__s
{{ <<uut_prototype_stubs.memset.__s>> == ( <<lwscicommon_libc.LwSciCommonFree.ptr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.memset.__n
{{ <<uut_prototype_stubs.memset.__n>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr>>[0].allocSize ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonFree.ptr
<<lwscicommon_libc.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr>>[1] );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonFree.Panic_due_to_hdr_magic_Is_Not_equal_to_LWSCICOMMON_ALLOC_MAGIC
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonFree
TEST.NEW
TEST.NAME:TC_002.LwSciCommonFree.Panic_due_to_hdr_magic_Is_Not_equal_to_LWSCICOMMON_ALLOC_MAGIC
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonFree.Panic_due_to_hdr_magic_Is_Not_equal_to_LWSCICOMMON_ALLOC_MAGIC}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Panics when the memory was not allocated via call to LwSciCommonCalloc().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'ptr' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18859983}
 *
 * @verify{18851226}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonFree
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonFree.ptr
<<lwscicommon_libc.LwSciCommonFree.ptr>> = ( &VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonFree.Failed_due_to_ptr_Is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonFree
TEST.NEW
TEST.NAME:TC_003.LwSciCommonFree.Failed_due_to_ptr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonFree.Failed_due_to_ptr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Failed due to ptr is equal to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'ptr' set to NULL.}
 *
 * @testbehavior{returns NULL.}
 *
 * @testcase{18859986}
 *
 * @verify{18851226}
 */
TEST.END_NOTES:
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonFree
  lwscicommon_libc.c.LwSciCommonFree
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonFree.ptr
<<lwscicommon_libc.LwSciCommonFree.ptr>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_libc.LwSciCommonFree.ptr
{{ <<lwscicommon_libc.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonMemcmp

-- Test Case: TC_001.LwSciCommonMemcmp.Success_use_case
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcmp
TEST.NEW
TEST.NAME:TC_001.LwSciCommonMemcmp.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMemcmp.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcmp() - Success case when ptr1 and ptr2 are equal.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{'VECTORCAST_INT1' set to '1'.
 *  'VECTORCAST_INT2' set to '1'.}
 *
 * @testinput{'ptr1' set to valid memory.
 *  'ptr2' set to valid memory.
 *  'size' driven as '0'.}
 *
 * @testbehavior{return '0'.}
 *
 * @testcase{18859989}
 *
 * @verify{18851232}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr1:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr2:VECTORCAST_INT2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.size:0
TEST.EXPECTED:lwscicommon_libc.LwSciCommonMemcmp.return:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcmp
  lwscicommon_libc.c.LwSciCommonMemcmp
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Lessthen_ptr2
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcmp
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Lessthen_ptr2
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Lessthen_ptr2}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcmp() - Panic due to ptr1 is less than ptr2.}
 *
 * @casederiv{Analysis of Requirements.
 *  Generation and Analysis of Equivalence Classes.}
 *
 * @testsetup{'VECTORCAST_INT1' set to '1'.
 *  'VECTORCAST_INT2' set to '2'.}
 *
 * @testinput{'ptr1' set to valid memory.
 *  'ptr2' set to valid memory.
 *  'size' driven as '1234'.}
 *
 * @testbehavior{return '-1'.}
 *
 * @testcase{18859992}
 *
 * @verify{18851232}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr1:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr2:VECTORCAST_INT2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.size:1234
TEST.EXPECTED:lwscicommon_libc.LwSciCommonMemcmp.return:-1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcmp
  lwscicommon_libc.c.LwSciCommonMemcmp
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Greaterthan_ptr2
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcmp
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Greaterthan_ptr2
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_Greaterthan_ptr2}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcmp() - Panic due to ptr1 is greater than ptr2.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{'VECTORCAST_INT1' set to '2'.
 *  'VECTORCAST_INT2' set to '1'.}
 *
 * @testinput{'ptr1' set to valid memory.
 *  'ptr2' set to valid memory.
 *  'size' driven as 'UINT64_MAX'.}
 *
 * @testbehavior{return '1'.}
 *
 * @testcase{18859995}
 *
 * @verify{18851232}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr1:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr2:VECTORCAST_INT2
TEST.EXPECTED:lwscicommon_libc.LwSciCommonMemcmp.return:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcmp
  lwscicommon_libc.c.LwSciCommonMemcmp
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcmp.size
<<lwscicommon_libc.LwSciCommonMemcmp.size>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonMemcmp.Panic_due_to_ptr2_Is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcmp
TEST.NEW
TEST.NAME:TC_004.LwSciCommonMemcmp.Panic_due_to_ptr2_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonMemcmp.Panic_due_to_ptr2_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcmp() - Panic due to ptr2 is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'ptr1' set to valid memory.
 *  'ptr2' set to 'NULL'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18859998}
 *
 * @verify{18851232}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr1:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr2:<<null>>
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcmp
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonMemcmp.Panic_due_to_ptr1_Ia_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcmp
TEST.NEW
TEST.NAME:TC_005.LwSciCommonMemcmp.Panic_due_to_ptr1_Ia_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonMemcmp.Panic_due_to_ptr1_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcmp() - Panic due to ptr1 is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'ptr1' set to NULL.
 *  'ptr2' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860001}
 *
 * @verify{18851232}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr1:<<null>>
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcmp.ptr2:VECTORCAST_INT1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcmp
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonMemcpyS

-- Test Case: TC_001.LwSciCommonMemcpyS.Success_use_case
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_001.LwSciCommonMemcpyS.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMemcpyS.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Success case when memory copied to destination successfully.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{'VECTORCAST_INT1' set tp '1'.
 *  'VECTORCAST_INT2' set tp '3'.}
 *
 * @testinput{'dest' set to valid memory.
 *  'src' set to valid memory.
 *  'destSize' driven as '2'.
 *  'n' driven as '1'.}
 *
 * @testbehavior{'dest' points to valid value.}
 *
 * @testcase{22060298}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:3
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.dest:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.destSize:2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.src:VECTORCAST_INT2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.n:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:3
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Panic due to dest and src buffers are overlap.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'dest' set to 'UINT32_MAX'.
 *  'src' set to 'UINT32_MAX'.
 *  'destSize' driven as '2'.
 *  'n' driven as '2'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060300}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.destSize:2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.n:2
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.dest
<<lwscicommon_libc.LwSciCommonMemcpyS.dest>> = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.src
<<lwscicommon_libc.LwSciCommonMemcpyS.src>> = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonMemcpyS.Panic_due_to_dest_buffer_in_MemcpyS_too_small
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMemcpyS.Panic_due_to_dest_buffer_in_MemcpyS_too_small
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMemcpyS.Panic_due_to_dest_buffer_in_MemcpyS_too_small}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Panic due to destSize is 2.}
 *
 * @casederiv{Analysis of Requirements.
 *  Boundary value analysis.}
 *
 * @testsetup{None.}
 *
 * @testinput{'dest' set to valid memory.
 *  'src' set to valid memory.
 *  'destSize' driven as '2'.
 *  'n' driven as 'UINT64_MAX'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060305}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.dest:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.destSize:2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.src:VECTORCAST_INT2
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.n
<<lwscicommon_libc.LwSciCommonMemcpyS.n>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonMemcpyS.Panic_due_to_dest_is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_004.LwSciCommonMemcpyS.Panic_due_to_dest_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonMemcpyS.Panic_due_to_dest_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Panic due to dest is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'dest' set to NULL.
 *  'src' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060306}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.dest:<<null>>
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.src:VECTORCAST_INT2
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonMemcpyS.Panic_due_to_src_is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_005.LwSciCommonMemcpyS.Panic_due_to_src_is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonMemcpyS.Panic_due_to_src_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Panic due to src is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'dest' set to valid memory.
 *  'src' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060309}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.dest:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.src:<<null>>
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciCommonMemcpyS.Success_use_case.destSize_Size_MAX
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_006.LwSciCommonMemcpyS.Success_use_case.destSize_Size_MAX
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonMemcpyS.Success_use_case.destSize_Size_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Success case when memory copied to destination successfully.}
 *
 * @casederiv{Analysis of Requirements.
 *  Boundary value analysis.}
 *
 * @testsetup{'VECTORCAST_INT1' set tp '1'.
 *  'VECTORCAST_INT2' set tp '3'.}
 *
 * @testinput{'dest' set to valid memory.
 *  'src' set to valid memory.
 *  'destSize' driven as 'UINT64_MAX'.
 *  'n' driven as '0'.}
 *
 * @testbehavior{'dest' points to valid value.}
 *
 * @testcase{22060314}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:3
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.dest:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.src:VECTORCAST_INT2
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.n:0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.memset.__c:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.destSize
<<lwscicommon_libc.LwSciCommonMemcpyS.destSize>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonMemcpyS
TEST.NEW
TEST.NAME:TC_007.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonMemcpyS.Panic_due_to_dest_and_src_buffers_are_overlap}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMemcpyS() - Panic due to dest and src buffers are overlap.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'dest' set to 'UINT32_MAX'.
 *  'src' set to 'UINT32_MAX-1'.
 *  'destSize' driven as '5'.
 *  'n' driven as '5'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060316}
 *
 * @verify{18851229}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.destSize:5
TEST.VALUE:lwscicommon_libc.LwSciCommonMemcpyS.n:5
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.dest
<<lwscicommon_libc.LwSciCommonMemcpyS.dest>> = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonMemcpyS.src
<<lwscicommon_libc.LwSciCommonMemcpyS.src>> = ( UINT32_MAX-1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonSort

-- Test Case: TC_001.LwSciCommonSort.Success_use_case
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_001.LwSciCommonSort.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonSort.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Success case when Sorts elements successfully.}
 *
 * @casederiv{Analysis of Requirements.
 *  Boundary value analysis.}
 *
 * @testsetup{None.}
 *
 * @testinput{'base' set to valid memory
 *  'nmemb' driven as '1'.
 *  'size' driven as '1'.
 *  'compare' set to valid memory.}
 *
 * @testbehavior{None.}
 *
 * @testcase{18860013}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.nmemb:1
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.size:1
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
extern int CompareUint32(const void* elem1, const void* elem2);
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( CompareUint32 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonSort.Success_use_case2_aStartAddr_Is_Lessthan_bStartAddr
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_002.LwSciCommonSort.Success_use_case2_aStartAddr_Is_Lessthan_bStartAddr
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonSort.Success_use_case2_aStartAddr_Is_Lessthan_bStartAddr}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Succes case when aStartAddr is Lessthan bStartAddr.}
 *
 * @casederiv{Analysis of Requirements.
 *  Generation and Analysis of Equivalence Classes.}
 *
 * @testsetup{'VECTORCAST_BUFFER[0]' set to '12'.
 *  'VECTORCAST_BUFFER[1]' set to '2'.
 *  'VECTORCAST_BUFFER[2]' set to '9'.
 *  'VECTORCAST_BUFFER[3]' set to '1'.
 *  'VECTORCAST_BUFFER[4]' set to '5'.
 *  'VECTORCAST_BUFFER[5]' set to '21'.}
 *
 * @testinput{'base' set to valid memory
 *  'nmemb' driven as '6'.
 *  'size' driven as '8'.
 *  'compare' set to valid function.}
 *
 * @testbehavior{None.}
 *
 * @testcase{18860016}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:12
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[1]:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[2]:9
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[3]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[4]:5
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[5]:21
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.nmemb:6
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.size:8
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
extern int CompareUint32(const void* elem1, const void* elem2);
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( CompareUint32 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonSort.Panic_due_to_base_Is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_003.LwSciCommonSort.Panic_due_to_base_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonSort.Panic_due_to_base_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Panic due to base is NULL.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'base' set to NULL.
 *  'nmemb' driven as '1'.
 *  'size' driven as '1'.
 *  'compare' set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860019}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:<<null>>
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.nmemb:1
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.size:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
extern int CompareUint32(const void* elem1, const void* elem2);
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( CompareUint32 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonSort.Panic_due_to_nmemb_Is_0
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_004.LwSciCommonSort.Panic_due_to_nmemb_Is_0
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonSort.Panic_due_to_nmemb_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Panic due to nmemb is 0.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'base' set to valid memory.
 *  'nmemb' driven as '0'.
 *  'size' driven as '1'.
 *  'compare' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860022}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.nmemb:0
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.size:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
extern int CompareUint32(const void* elem1, const void* elem2);
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( CompareUint32 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonSort.Panic_due_to_size_Is_0
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_005.LwSciCommonSort.Panic_due_to_size_Is_0
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonSort.Panic_due_to_size_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Panic due to size is 0.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'base' set to valid memory.
 *  'nmemb' driven as '1'.
 *  'size' driven as '0'.
 *  'compare' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860025}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:VECTORCAST_INT1
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.nmemb:1
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.size:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
extern int CompareUint32(const void* elem1, const void* elem2);
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( CompareUint32 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciCommonSort.Panic_due_to_compare_Is_NULL
TEST.UNIT:lwscicommon_libc
TEST.SUBPROGRAM:LwSciCommonSort
TEST.NEW
TEST.NAME:TC_006.LwSciCommonSort.Panic_due_to_compare_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonSort.Panic_due_to_compare_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSort() - Panic due to compare is NULL.}
 *
 * @casederiv{Analysis of Requirements.
 *  Analysis of Boundary Values.}
 *
 * @testsetup{None.}
 *
 * @testinput{'base' set to valid memory.
 *  'nmemb' driven as 'UINT64_MAX'.
 *  'size' driven as 'UINT64_MAX'.
 *  'compare' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860028}
 *
 * @verify{18851283}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.VALUE:lwscicommon_libc.LwSciCommonSort.base:VECTORCAST_INT1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_libc.c.LwSciCommonSort
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.nmemb
<<lwscicommon_libc.LwSciCommonSort.nmemb>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.size
<<lwscicommon_libc.LwSciCommonSort.size>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_libc.LwSciCommonSort.compare
<<lwscicommon_libc.LwSciCommonSort.compare>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Unit: lwscicommon_posix

-- Subprogram: LwSciCommonMutexCreate

-- Test Case: TC_001.LwSciCommonMutexCreate.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexCreate
TEST.NEW
TEST.NAME:TC_001.LwSciCommonMutexCreate.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMutexCreate.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexCreate() - Success case when initializes a LwSciCommonMutex object used for thread synchronization.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_init() returns '0'.}
 *
 * @testinput{mutex set to valid memory.}
 *
 * @testbehavior{pthread_mutex_init() receives '__mutex' as valid address.
 *  pthread_mutex_init() receives '__mutezattr' as NULL.
 *  returns LwSciError_Success.}
 *
 * @testcase{18860031}
 *
 * @verify{18851235}
 */
 
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.pthread_mutex_init
TEST.VALUE:uut_prototype_stubs.pthread_mutex_init.return:0
TEST.EXPECTED:lwscicommon_posix.LwSciCommonMutexCreate.return:LwSciError_Success
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexCreate
  uut_prototype_stubs.pthread_mutex_init
  lwscicommon_posix.c.LwSciCommonMutexCreate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutexattr
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutexattr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexCreate.mutex
<<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexCreate
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexCreate() - Panic due to when failed to acquire lock on a mutex.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_init() returns '1'.}
 *
 * @testinput{mutex set to valid memory.}
 *
 * @testbehavior{pthread_mutex_init() receives '__mutex' as valid address.
 *  pthread_mutex_init() receives '__mutezattr' as NULL.
 *  LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860034}
 *
 * @verify{18851235}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.STUB:uut_prototype_stubs.pthread_mutex_init
TEST.VALUE:uut_prototype_stubs.pthread_mutex_init.return:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexCreate
  uut_prototype_stubs.pthread_mutex_init
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutexattr
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutexattr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexCreate.mutex
<<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_ENOMEM
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexCreate
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_ENOMEM
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_ENOMEM}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexCreate() - pthread_mutex_init() returns ENOMEM.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_init() returns 'ENOMEM'.}
 *
 * @testinput{mutex set to valid memory.}
 *
 * @testbehavior{pthread_mutex_init() receives '__mutex' as valid address.
 *  pthread_mutex_init() receives '__mutezattr' as NULL.
 *  returns LwSciError_InsufficientMemory when memory allocation failed.}
 *
 * @testcase{18860037}
 *
 * @verify{18851235}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.pthread_mutex_init
TEST.EXPECTED:lwscicommon_posix.LwSciCommonMutexCreate.return:LwSciError_InsufficientMemory
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexCreate
  uut_prototype_stubs.pthread_mutex_init
  lwscicommon_posix.c.LwSciCommonMutexCreate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.pthread_mutex_init.return
<<uut_prototype_stubs.pthread_mutex_init.return>> = ( ENOMEM );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutexattr
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutexattr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexCreate.mutex
<<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_EAGAIN
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexCreate
TEST.NEW
TEST.NAME:TC_004.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_EAGAIN
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonMutexCreate.Panic_due_to_pthread_mutex_init()_returns_EAGAIN}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexCreate() - pthread_mutex_init() returns EAGAIN.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_init() returns 'EAGAIN'.}
 *
 * @testinput{mutex set to valid memory.}
 *
 * @testbehavior{pthread_mutex_init() receives '__mutex' as valid address.
 *  pthread_mutex_init() receives '__mutezattr' as NULL.
 *  returns LwSciError_ResourceError when system lacks resource other than memory to initialize the lock.}
 *
 * @testcase{18860040}
 *
 * @verify{18851235}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.pthread_mutex_init
TEST.EXPECTED:lwscicommon_posix.LwSciCommonMutexCreate.return:LwSciError_ResourceError
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexCreate
  uut_prototype_stubs.pthread_mutex_init
  lwscicommon_posix.c.LwSciCommonMutexCreate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.pthread_mutex_init.return
<<uut_prototype_stubs.pthread_mutex_init.return>> = ( EAGAIN );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_init.__mutexattr
{{ <<uut_prototype_stubs.pthread_mutex_init.__mutexattr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexCreate.mutex
<<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonMutexCreate.Panic_due_to_mutex_Is_NULL
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexCreate
TEST.NEW
TEST.NAME:TC_005.LwSciCommonMutexCreate.Panic_due_to_mutex_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonMutexCreate.Panic_due_to_mutex_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexCreate() - Panic due to mutex is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{abort() the function.}
 *
 * @testinput{'mutex' set to 'NULL'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860043}
 *
 * @verify{18851235}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexCreate
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexCreate.mutex
<<lwscicommon_posix.LwSciCommonMutexCreate.mutex>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonMutexDestroy

-- Test Case: TC_001.LwSciCommonMutexDestroy.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexDestroy
TEST.NEW
TEST.NAME:TC_001.LwSciCommonMutexDestroy.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMutexDestroy.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexDestroy() - Success case when mutex destroy successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_destroy() returns '0'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexDestroy() receives '_mutex' as valid address.}
 *
 * @testcase{18860046}
 *
 * @verify{18851244}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.pthread_mutex_destroy
TEST.VALUE:uut_prototype_stubs.pthread_mutex_destroy.return:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.memset.__c:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexDestroy
  uut_prototype_stubs.pthread_mutex_destroy
  lwscicommon_posix.c.LwSciCommonMutexDestroy
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_destroy.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_destroy.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexDestroy.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexDestroy.mutex
<<lwscicommon_posix.LwSciCommonMutexDestroy.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonMutexDestroy.Panic_due_to_pthread_mutex_destroy()_returns_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexDestroy
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMutexDestroy.Panic_due_to_pthread_mutex_destroy()_returns_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMutexDestroy.Panic_due_to_pthread_mutex_destroy()_returns_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexDestroy() - Panic due to pthread_mutex_destroy() returns 1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_destroy() returns '1'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexDestroy() receives '_mutex' as valid address.
 *  LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860049}
 *
 * @verify{18851244}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.STUB:uut_prototype_stubs.pthread_mutex_destroy
TEST.VALUE:uut_prototype_stubs.pthread_mutex_destroy.return:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexDestroy
  uut_prototype_stubs.pthread_mutex_destroy
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_destroy.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_destroy.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexDestroy.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexDestroy.mutex
<<lwscicommon_posix.LwSciCommonMutexDestroy.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonMutexDestroy.Panic_due_to_mutex_Is_NULL
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexDestroy
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMutexDestroy.Panic_due_to_mutex_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMutexDestroy.Panic_due_to_mutex_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexDestroy() - Panic due to mutex is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{abort() the function.}
 *
 * @testinput{'mutex' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060284}
 *
 * @verify{18851244}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexDestroy
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexDestroy.mutex
<<lwscicommon_posix.LwSciCommonMutexDestroy.mutex>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonMutexLock

-- Test Case: TC_001.LwSciCommonMutexLock.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexLock
TEST.NEW
TEST.NAME:TC_001.LwSciCommonMutexLock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMutexLock.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexLock() - Success case when mutex locks successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_lock() returns '0'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives '_mutex' as valid address.}
 *
 * @testcase{18860055}
 *
 * @verify{18851238}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.pthread_mutex_lock
TEST.VALUE:uut_prototype_stubs.pthread_mutex_lock.return:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.memset.__c:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexLock
  uut_prototype_stubs.pthread_mutex_lock
  lwscicommon_posix.c.LwSciCommonMutexLock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_lock.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_lock.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexLock.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexLock.mutex
<<lwscicommon_posix.LwSciCommonMutexLock.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonMutexLock.Panic_due_to_pthread_mutex_lock()_returns_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexLock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMutexLock.Panic_due_to_pthread_mutex_lock()_returns_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMutexLock.Panic_due_to_pthread_mutex_lock()_returns_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexLock() - Panic due to pthread_mutex_lock() returns 1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_lock() returns '1'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexLock() receives '_mutex' as valid address.
 *  LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860058}
 *
 * @verify{18851238}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.STUB:uut_prototype_stubs.pthread_mutex_lock
TEST.VALUE:uut_prototype_stubs.pthread_mutex_lock.return:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexLock
  uut_prototype_stubs.pthread_mutex_lock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_lock.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_lock.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexLock.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexLock.mutex
<<lwscicommon_posix.LwSciCommonMutexLock.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonMutexLock.Panic_due_to_mutex_Is_NULL
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexLock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMutexLock.Panic_due_to_mutex_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMutexLock.Panic_due_to_mutex_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexLock() - Panic due to mutex is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{abort() the function.}
 *
 * @testinput{'mutex' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860061}
 *
 * @verify{18851238}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexLock
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexLock.mutex
<<lwscicommon_posix.LwSciCommonMutexLock.mutex>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonMutexUnlock

-- Test Case: TC_001.MutexUnlock.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexUnlock
TEST.NEW
TEST.NAME:TC_001.MutexUnlock.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMutexUnlock.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexUnlock() - Success case when mutex locks successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_unlock() returns '0'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexUnlock() receives '_mutex' as valid address.}
 *
 * @testcase{18860064}
 *
 * @verify{18851241}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.pthread_mutex_unlock
TEST.VALUE:uut_prototype_stubs.pthread_mutex_unlock.return:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.memset.__c:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexUnlock
  uut_prototype_stubs.pthread_mutex_unlock
  lwscicommon_posix.c.LwSciCommonMutexUnlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_unlock.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_unlock.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexUnlock.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexUnlock.mutex
<<lwscicommon_posix.LwSciCommonMutexUnlock.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonMutexUnlock.Panic_due_to_pthread_mutex_unlock()_returns_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexUnlock
TEST.NEW
TEST.NAME:TC_002.LwSciCommonMutexUnlock.Panic_due_to_pthread_mutex_unlock()_returns_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonMutexUnlock.Panic_due_to_pthread_mutex_unlock()_returns_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexUnlock() - Panic due to pthread_mutex_unlock() returns 1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{pthread_mutex_unlock() returns '1'.}
 *
 * @testinput{'mutex' set to valid memory.}
 *
 * @testbehavior{LwSciCommonMutexUnlock() receives '_mutex' as valid address.
 *  LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860067}
 *
 * @verify{18851241}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.STUB:uut_prototype_stubs.pthread_mutex_unlock
TEST.VALUE:uut_prototype_stubs.pthread_mutex_unlock.return:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexUnlock
  uut_prototype_stubs.pthread_mutex_unlock
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.pthread_mutex_unlock.__mutex
{{ <<uut_prototype_stubs.pthread_mutex_unlock.__mutex>> == ( <<lwscicommon_posix.LwSciCommonMutexUnlock.mutex>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexUnlock.mutex
<<lwscicommon_posix.LwSciCommonMutexUnlock.mutex>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonMutexUnlock.Panic_due_to_mutex_Is_NULL
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonMutexUnlock
TEST.NEW
TEST.NAME:TC_003.LwSciCommonMutexUnlock.Panic_due_to_mutex_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonMutexUnlock.Panic_due_to_mutex_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonMutexUnlock() - Panic due to mutex is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{abort() the function.}
 *
 * @testinput{'mutex' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860070}
 *
 * @verify{18851241}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonMutexUnlock
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonMutexUnlock.mutex
<<lwscicommon_posix.LwSciCommonMutexUnlock.mutex>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonPanic

-- Test Case: TC_001.LwSciCommonPanic.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonPanic
TEST.NEW
TEST.NAME:TC_001.LwSciCommonPanic.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonPanic.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonPanic() - Success case when abort the function successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{abort() the function.}
 *
 * @testinput{None.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860073}
 *
 * @verify{18851247}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.abort
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.memset.__c:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonSleepNs

-- Test Case: TC_001.LwSciCommonSleepNs.Success_use_case
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonSleepNs
TEST.NEW
TEST.NAME:TC_001.LwSciCommonSleepNs.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonSleepNs.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSleepNs() - Success case when suspends exelwtion of a thread successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{nanosleep() returns '0'.}
 *
 * @testinput{timeNs deiven as '5'.}
 *
 * @testbehavior{nanosleep() receives '_reuested_time' as valid address.
 *  nanosleep() receives '_remaining' as NULL.}
 *
 * @testcase{18860076}
 *
 * @verify{18851250}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.memset
TEST.STUB:uut_prototype_stubs.nanosleep
TEST.VALUE:uut_prototype_stubs.nanosleep.return:0
TEST.VALUE:lwscicommon_posix.LwSciCommonSleepNs.timeNs:5
TEST.EXPECTED:uut_prototype_stubs.memset.__c:0
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonSleepNs
  uut_prototype_stubs.nanosleep
  uut_prototype_stubs.memset
  lwscicommon_posix.c.LwSciCommonSleepNs
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.memset.__s
{{ <<uut_prototype_stubs.memset.__s>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.memset.__n
{{ <<uut_prototype_stubs.memset.__n>> == ( sizeof(struct timespec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.nanosleep.__requested_time
{{ <<uut_prototype_stubs.nanosleep.__requested_time>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.nanosleep.__remaining
{{ <<uut_prototype_stubs.nanosleep.__remaining>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonSleepNs.Panic_due_to_nanosleep()_returns_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonSleepNs
TEST.NEW
TEST.NAME:TC_002.LwSciCommonSleepNs.Panic_due_to_nanosleep()_returns_1
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonMemcmp.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSleepNs() - Panics when failed to suspends exelwtion of a thread.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{nanosleep() returns '1'.}
 *
 * @testinput{timeNs deiven as '5'.}
 *
 * @testbehavior{nanosleep() receives '_reuested_time' as valid address.
 *  nanosleep() receives '_remaining' as NULL.
 *  LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{18860079}
 *
 * @verify{18851250}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.STUB:uut_prototype_stubs.nanosleep
TEST.VALUE:uut_prototype_stubs.nanosleep.return:1
TEST.VALUE:lwscicommon_posix.LwSciCommonSleepNs.timeNs:5
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].magic:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr[0].allocSize:EXPECTED_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.test_ptr2.magic:EXPECTED_BASE=16
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonSleepNs
  uut_prototype_stubs.nanosleep
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.nanosleep.__requested_time
{{ <<uut_prototype_stubs.nanosleep.__requested_time>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.nanosleep.__remaining
{{ <<uut_prototype_stubs.nanosleep.__remaining>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonSleepNs.timeNs_Is____LONG_MAX___+_1
TEST.UNIT:lwscicommon_posix
TEST.SUBPROGRAM:LwSciCommonSleepNs
TEST.NEW
TEST.NAME:TC_003.LwSciCommonSleepNs.timeNs_Is____LONG_MAX___+_1
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonSleepNs.timeNs_Is____LONG_MAX___+_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonSleepNs() - Panics when timeNs is ____LONG_MAX___+_1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{none.}
 *
 * @testinput{timeNs deiven as '____LONG_MAX___+_1'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060318}
 *
 * @verify{18851250}
 */
TEST.END_NOTES:
TEST.STUB:uut_prototype_stubs.abort
TEST.FLOW
  lwscicommon_posix.c.LwSciCommonSleepNs
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_posix.LwSciCommonSleepNs.timeNs
<<lwscicommon_posix.LwSciCommonSleepNs.timeNs>> = ( __LONG_MAX__ + 1 );
TEST.END_VALUE_USER_CODE:
TEST.END
