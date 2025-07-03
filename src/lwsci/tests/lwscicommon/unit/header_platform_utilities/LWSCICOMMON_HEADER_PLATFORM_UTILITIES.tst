-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCICOMMON_HEADER_PLATFORM_UTILITIES
-- Unit(s) Under Test: lwscilist
-- 
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Subprogram: lwListAppend

-- Test Case: TC_001.lwListAppend.Success_use_case
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListAppend
TEST.NEW
TEST.NAME:TC_001.lwListAppend.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.lwListAppend.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - lwListAppend() - Success case when test appends a new node to the tail of the linklist head node successfully.
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'entry' set to valid memory.
 *  - 'head' set to valid memory.}
 *
 * @testbehavior{- 'head[0]->prev' pointing to valid address.
 *  - 'head[0]->prev[0]->prev[0]->next' pointing to valid address.
 *  - 'entry[0]->next' pointing to valid address.
 *  - 'entry[0]->prev' pointing to valid address.}
 *
 * @testcase{18860082}
 *
 * @verify{18851256}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListAppend.entry:<<malloc 1>>
TEST.VALUE:lwscilist.lwListAppend.head:<<malloc 1>>
TEST.VALUE:lwscilist.lwListAppend.head[0].prev:<<malloc 1>>
TEST.VALUE:lwscilist.lwListAppend.head[0].prev[0].next:<<malloc 1>>
TEST.VALUE:lwscilist.lwListAppend.head[0].prev[0].prev:<<malloc 1>>
TEST.FLOW
  lwscilist.h.lwListAppend
  lwscilist.h.lwListAppend
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscilist.lwListAppend.entry.entry[0].next
{{ <<lwscilist.lwListAppend.entry>>[0].next == ( &<<lwscilist.lwListAppend.head>>[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListAppend.entry.entry[0].prev
{{ <<lwscilist.lwListAppend.entry>>[0].prev == ( &<<lwscilist.lwListAppend.head>>[0].prev[0].prev[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListAppend.head.head[0].prev
{{ <<lwscilist.lwListAppend.head>>[0].prev == ( &<<lwscilist.lwListAppend.entry>>[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListAppend.head.head[0].prev.prev[0].prev.prev[0].next
{{ <<lwscilist.lwListAppend.head>>[0].prev[0].prev[0].next == ( &<<lwscilist.lwListAppend.entry>>[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.lwListAppend.Panic_due_to_entry_Is_NULL
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListAppend
TEST.NEW
TEST.NAME:TC_002.lwListAppend.Panic_due_to_entry_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.lwListAppend.Panic_due_to_entry_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Panics when the entry is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'entry' set to NULL.
 *  - 'head' set to valid memory.}
 *
 * @testbehavior{- LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060287}
 *
 * @verify{18851256}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListAppend.entry:<<null>>
TEST.VALUE:lwscilist.lwListAppend.head:<<malloc 1>>
TEST.FLOW
  lwscilist.h.lwListAppend
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.lwListAppend.Panic_due_to_head_Is_NULL
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListAppend
TEST.NEW
TEST.NAME:TC_003.lwListAppend.Panic_due_to_head_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.lwListAppend.Panic_due_to_head_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Panics when the head is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'head' set to NULL.
 *  - 'entry' set to valid memory.}
 *
 * @testbehavior{- LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060290}
 *
 * @verify{18851256}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListAppend.entry:<<malloc 1>>
TEST.VALUE:lwscilist.lwListAppend.head:<<null>>
TEST.FLOW
  lwscilist.h.lwListAppend
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: lwListDel

-- Test Case: TC_001.lwListDel.Success_use_case
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListDel
TEST.NEW
TEST.NAME:TC_001.lwListDel.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.lwListDel.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - lwListDel() - Success case when reset the list as empty list successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'entry' set to valid memory.}
 *
 * @testbehavior{- 'entry[0]->next' pointing to valid address.
 *  - 'entry[0]->next[0]->prev' pointing to valid address.
 *  - 'entry[0]->prev' pointing to valid address.
 *  - 'entry[0]->prev[0]->next' pointing to valid address.}
 *
 * @testcase{18860085}
 *
 * @verify{18851259}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListDel.entry:<<malloc 1>>
TEST.VALUE:lwscilist.lwListDel.entry[0].next:<<malloc 1>>
TEST.VALUE:lwscilist.lwListDel.entry[0].prev:<<malloc 1>>
TEST.FLOW
  lwscilist.h.lwListDel
  lwscilist.h.lwListDel
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscilist.lwListDel.entry.entry[0].next
{{ <<lwscilist.lwListDel.entry>>[0].next == ( <<lwscilist.lwListDel.entry>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListDel.entry.entry[0].next.next[0].prev
{{ <<lwscilist.lwListDel.entry>>[0].next[0].prev == ( &<<lwscilist.lwListDel.entry>>[0].prev[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListDel.entry.entry[0].prev
{{ <<lwscilist.lwListDel.entry>>[0].prev == ( <<lwscilist.lwListDel.entry>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListDel.entry.entry[0].prev.prev[0].next
{{ <<lwscilist.lwListDel.entry>>[0].prev[0].next == ( &<<lwscilist.lwListDel.entry>>[0].next[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.lwListDel.entry_Is_NULL
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListDel
TEST.NEW
TEST.NAME:TC_002.lwListDel.entry_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.lwListDel.entry_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Panics when the entry is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'entry' set to NULL.}
 *
 * @testbehavior{- LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060323}
 *
 * @verify{18851259}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListDel.entry:<<null>>
TEST.FLOW
  lwscilist.h.lwListDel
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: lwListInit

-- Test Case: TC_001.lwListInit.Success_use_case
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListInit
TEST.NEW
TEST.NAME:TC_001.lwListInit.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.lwListInit.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - lwListInit() - Success case when initialized the list as an empty list successfully.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'list' set to valid memory.}
 *
 * @testbehavior{- 'list[0]->next' pointing to valid address.
 *  - 'list[0]->prev' pointing to valid address.}
 *
 * @testcase{18860088}
 *
 * @verify{18851253}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListInit.list:<<malloc 1>>
TEST.FLOW
  lwscilist.h.lwListInit
  lwscilist.h.lwListInit
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscilist.lwListInit.list.list[0].next
{{ <<lwscilist.lwListInit.list>>[0].next == ( <<lwscilist.lwListInit.list>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscilist.lwListInit.list.list[0].prev
{{ <<lwscilist.lwListInit.list>>[0].prev == ( <<lwscilist.lwListInit.list>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.lwListInit.list_Is_NULL
TEST.UNIT:lwscilist
TEST.SUBPROGRAM:lwListInit
TEST.NEW
TEST.NAME:TC_002.lwListInit.list_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.lwListInit.list_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonFree() - Panics when the list is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'list' set to NULL.}
 *
 * @testbehavior{- LwSciCommonPanic() is called and abort from the test case.}
 *
 * @testcase{22060325}
 *
 * @verify{18851253}
 */
TEST.END_NOTES:
TEST.VALUE:lwscilist.lwListInit.list:<<null>>
TEST.FLOW
  lwscilist.h.lwListInit
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END
