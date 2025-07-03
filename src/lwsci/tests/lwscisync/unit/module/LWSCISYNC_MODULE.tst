-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCISYNC_MODULE
-- Unit(s) Under Test: lwscisync_module
-- 
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Subprogram: LwSciSyncCoreModuleCntrGetNextValue

-- Test Case: TC_004.LwSciSyncCoreModuleCntrGetNextValue.Overflow
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleCntrGetNextValue
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreModuleCntrGetNextValue.Overflow
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreModuleCntrGetNextValue.Overflow}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleCntrGetNextValue() when counter value exceeds UINT64_MAX.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD}
 *
 * @testinput{ - module - points to valid LwSciSyncModule with module counter value set to UINT32_MAX.
 *  - cntrValue - points to memory holding sizeof(uint64_t) bytes.}
 *
 * @testbehavior{ - Function should return LwSciError_Overflow.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853860}
 *
 * @verify{18844605}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01[0].moduleCounter:<<MAX>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module
<<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreModuleCntrGetNextValue.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleCntrGetNextValue
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreModuleCntrGetNextValue.Success
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreModuleCntrGetNextValue.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleCntrGetNextValue() success case.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule with module counter value set to 4.
 *  - cntrValue - points to memory holding sizeof(uint64_t) bytes.}
 *
 * @testbehavior{ - Function should return LwSciError_Success.
 * - cntrValue is set to 4 (i.e module counter value)
 * - module counter value is increased to 5.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853863}
 *
 * @verify{18844605}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01[0].moduleCounter:4
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01[0].moduleCounter:5
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue[0]:4
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.return:LwSciError_Success
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonObjLock
  uut_prototype_stubs.LwSciCommonObjUnlock
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjLock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjLock.ref>> == ( <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonObjUnlock.ref
{{ <<uut_prototype_stubs.LwSciCommonObjUnlock.ref>> == ( <<uut_prototype_stubs.LwSciCommonObjLock.ref>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module
<<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>->moduleCounter == 5 }}
{{ *<<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue>> == 4 }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreModuleCntrGetNextValue.cntrValue_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleCntrGetNextValue
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreModuleCntrGetNextValue.cntrValue_is_null
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreModuleCntrGetNextValue.cntrValue_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleCntrGetNextValue() when input cntrValue is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule with module counter value set to 4.
 *  - cntrValue - points to NULL.}
 *
 * @testbehavior{ - Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853866}
 *
 * @verify{18844605}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01[0].moduleCounter:4
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module
<<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreModuleCntrGetNextValue.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleCntrGetNextValue
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreModuleCntrGetNextValue.module_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreModuleCntrGetNextValue.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleCntrGetNextValue() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{ - module - points to NULL.
 *  - cntrValue - points to memory holding sizeof(uint64_t) bytes.}
 *
 * @testbehavior{Function should Panic.}
 *
 * @testcase{18853869}
 *
 * @verify{18844605}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module:<<null>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciSyncCoreModuleCntrGetNextValue.Panic_due_to_ilwalid_module
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleCntrGetNextValue
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreModuleCntrGetNextValue.Panic_due_to_ilwalid_module
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreModuleCntrGetNextValue.Panic_due_to_ilwalid_module}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleCntrGetNextValue() when input module is invalid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD}
 *
 * @testinput{ - module - points to invalid LwSciSyncModule.
 *  - cntrValue - points to memory holding sizeof(uint64_t) bytes.}
 *
 * @testbehavior{ - Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060339}
 *
 * @verify{18844605}
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01[0].moduleCounter:4
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.cntrValue:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleCntrGetNextValue
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( 13 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module
<<lwscisync_module.LwSciSyncCoreModuleCntrGetNextValue.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreModuleDup

-- Test Case: TC_002.LwSciSyncCoreModuleDup.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreModuleDup.Success
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreModuleDup.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() success case.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{Function should return LwSciError_Success.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.
 * - LwSciSyncModule pointed by dupModule should be same as input module.}
 *
 * @testcase{18853875}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_module.c.LwSciSyncCoreModuleDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
*<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.return
<<uut_prototype_stubs.LwSciCommonDuplicateRef.return>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == (void*)<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? LwSciError_Success
 : LwSciError_BadParameter
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule
{{ *<<lwscisync_module.LwSciSyncCoreModuleDup.dupModule>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreModuleDup.dupModule_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreModuleDup.dupModule_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreModuleDup.dupModule_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when input dupModule is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - dupModule - points to NULL.}
 *
 * @testbehavior{Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853878}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreModuleDup.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreModuleDup.module_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreModuleDup.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{ - module - points to NULL.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{ - Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.}
 *
 * @testcase{18853881}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.module:<<null>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreModuleDup.InsufficientMemory
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreModuleDup.InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreModuleDup.InsufficientMemory}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when LwSciCommonDuplicateRef() returns
 * LwSciError_InsufficientMemory.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060342}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleDup.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_module.c.LwSciSyncCoreModuleDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
*<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.return
<<uut_prototype_stubs.LwSciCommonDuplicateRef.return>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> != (void*)<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? LwSciError_Success
 : LwSciError_InsufficientMemory
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreModuleDup.ResourceError
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreModuleDup.ResourceError
TEST.NOTES:
/*
 * @testname{TC_006.LwSciSyncCoreModuleDup.ResourceError}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when LwSciCommonDuplicateRef() returns
 * LwSciError_ResourceError.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{Function should return LwSciError_ResourceError
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060345}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleDup.return:LwSciError_ResourceError
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_module.c.LwSciSyncCoreModuleDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
*<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.return
<<uut_prototype_stubs.LwSciCommonDuplicateRef.return>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> != (void*)<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? LwSciError_Success
 : LwSciError_ResourceError
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreModuleDup.IlwalidState
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreModuleDup.IlwalidState
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreModuleDup.IlwalidState}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when LwSciCommonDuplicateRef() returns
 * LwSciError_IlwalidState.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{Function should return LwSciError_IlwalidState.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060348}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleDup.return:LwSciError_IlwalidState
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscisync_module.c.LwSciSyncCoreModuleDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
*<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.return
<<uut_prototype_stubs.LwSciCommonDuplicateRef.return>> = (
(<<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> != (void*)<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? LwSciError_Success
 : LwSciError_IlwalidState
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreModuleDup.Panic_due_to_ilwalid_module
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleDup
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreModuleDup.Panic_due_to_ilwalid_module
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreModuleDup.Panic_due_to_ilwalid_module}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleDup() when input module is invalid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to invalid LwSciSyncModule.
 *  - dupModule - points to allocated memory.}
 *
 * @testbehavior{- Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060351}
 *
 * @verify{18844611}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleDup.dupModule:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( 13 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleDup.module
<<lwscisync_module.LwSciSyncCoreModuleDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreModuleGetRmBackEnd

-- Test Case: TC_002.LwSciSyncCoreModuleGetRmBackEnd.backEnd_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleGetRmBackEnd
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreModuleGetRmBackEnd.backEnd_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreModuleGetRmBackEnd.backEnd_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleGetRmBackEnd() when input backEnd is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - backEnd - points to NULL.}
 *
 * @testbehavior{Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853887}
 *
 * @verify{18844617}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module
<<lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreModuleGetRmBackEnd.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleGetRmBackEnd
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreModuleGetRmBackEnd.Success
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreModuleGetRmBackEnd.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleGetRmBackEnd() success case.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - backEnd - points to allocated memory.}
 *
 * @testbehavior{Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.
 * - backEnd points to LwSciSyncCoreRmBackEnd present in module.}
 *
 * @testcase{18853890}
 *
 * @verify{18844617}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_module.c.LwSciSyncCoreModuleGetRmBackEnd
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module
<<lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
{{ <<lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].backEnd ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreModuleGetRmBackEnd.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleGetRmBackEnd
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreModuleGetRmBackEnd.module_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreModuleGetRmBackEnd.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleGetRmBackEnd() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to NULL.
 *  - backEnd - points to allocated memory.}
 *
 * @testbehavior{Function should Panic.}
 *
 * @testcase{22060354}
 *
 * @verify{18844617}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module:<<null>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreModuleGetRmBackEnd.Panic_due_to_ilwalid_module
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleGetRmBackEnd
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreModuleGetRmBackEnd.Panic_due_to_ilwalid_module
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreModuleGetRmBackEnd.Panic_due_to_ilwalid_module}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleGetRmBackEnd() when input module is invalid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD}
 *
 * @testinput{ - module - points to invalid LwSciSyncModule.
 *  - backEnd - points to allocated memory.}
 *
 * @testbehavior{ - Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060358}
 *
 * @verify{18844617}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.backEnd:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( 13 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module
<<lwscisync_module.LwSciSyncCoreModuleGetRmBackEnd.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreModuleIsDup

-- Test Case: TC_001.LwSciSyncCoreModuleIsDup.input_module_is_not_valid
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreModuleIsDup.input_module_is_not_valid
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreModuleIsDup.input_module_is_not_valid}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input module is not valid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to invalid LwSciSyncModule.
 *  - otherModule - points to valid LwSciSyncModule different from input module.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853893}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count = 0;
if( count == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
}
if( count == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule
<<lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreModuleIsDup.input_otherModule_is_not_valid
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreModuleIsDup.input_otherModule_is_not_valid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreModuleIsDup.input_otherModule_is_not_valid}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input otherModule is not valid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - otherModule - points to invalid LwSciSyncModule different from input module.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853896}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count = 0;
if( count == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
}
if( count == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule
<<lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreModuleIsDup.Success_input_modules_are_duplicate
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreModuleIsDup.Success_input_modules_are_duplicate
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreModuleIsDup.Success_input_modules_are_duplicate}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input otherModule is duplicate of input module.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - otherModule - points to valid LwSciSyncModule same as input module.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{- Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.
 * - isDup output parameter should be equal to true.}
 *
 * @testcase{18853899}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup[0]:true
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule
<<lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreModuleIsDup.Success_input_modules_are_not_duplicate
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreModuleIsDup.Success_input_modules_are_not_duplicate
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreModuleIsDup.Success_input_modules_are_not_duplicate}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input module and input otherModule are not duplicate.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - otherModule - points to valid LwSciSyncModule different from input module.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{- Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.
 * - isDup output parameter should be equal to true.}
 *
 * @testcase{18853902}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup[0]:false
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count = 0;
if( count == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
}
if( count == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02.coreModule02[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule
<<lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreModuleIsDup.isDup_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreModuleIsDup.isDup_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreModuleIsDup.isDup_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input isDup is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - otherModule - points to valid LwSciSyncModule different from input module.
 *  - isDup - points to NULL.}
 *
 * @testbehavior{- Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853905}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count = 0;
if( count == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
}
if( count == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02.coreModule02[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule02>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule
<<lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreModuleIsDup.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreModuleIsDup.module_is_null
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreModuleIsDup.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{ - module - points to NULL.
 *  - otherModule - points to valid LwSciSyncModule.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{- Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.}
 *
 * @testcase{18853908}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.module:<<null>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciSyncCoreModuleIsDup.otherModule_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleIsDup
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreModuleIsDup.otherModule_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreModuleIsDup.otherModule_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleIsDup() when input otherModule is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.
 *  - otherModule - points to NULL.
 *  - isDup - points to allocated memory.}
 *
 * @testbehavior{Function should Panic.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853911}
 *
 * @verify{18844614}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.otherModule:<<null>>
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleIsDup.isDup:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleIsDup
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = (
 (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>)
 ? (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>)
 : NULL
);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count = 0;
if( count == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> ) }}
}
if( count == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef2>> ) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleIsDup.module
<<lwscisync_module.LwSciSyncCoreModuleIsDup.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreModuleValidate

-- Test Case: TC_002.LwSciSyncCoreModuleValidate.Panic_due_to_ilwalid_module
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleValidate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreModuleValidate.Panic_due_to_ilwalid_module
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreModuleValidate.Panic_due_to_ilwalid_module}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleValidate() when input module is invalid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to invalid LwSciSyncModule.}
 *
 * @testbehavior{Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853917}
 *
 * @verify{18844608}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleValidate.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = (13);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleValidate.module
<<lwscisync_module.LwSciSyncCoreModuleValidate.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreModuleValidate.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleValidate
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreModuleValidate.module_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreModuleValidate.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleValidate() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{module points to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18853923}
 *
 * @verify{18844608}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncCoreModuleValidate.module:<<null>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleValidate
  lwscisync_module.c.LwSciSyncCoreModuleValidate
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreModuleValidate.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncCoreModuleValidate
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreModuleValidate.Success
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreModuleValidate.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncCoreModuleValidate() success case.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.}
 *
 * @testbehavior{Function should return LwSciError_Success.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853926}
 *
 * @verify{18844608}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_module.c.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_module.c.LwSciSyncCoreModuleValidate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncCoreModuleValidate.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncCoreModuleValidate.module
<<lwscisync_module.LwSciSyncCoreModuleValidate.module>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncModuleClose

-- Test Case: TC_001.LwSciSyncModuleClose.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleClose
TEST.NEW
TEST.NAME:TC_001.LwSciSyncModuleClose.Success
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncModuleClose.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleClose() success case.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.}
 *
 * @testbehavior{- Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853932}
 *
 * @verify{18844599}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreRmFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_module.c.LwSciSyncModuleClose
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncModuleClose.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmFree.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmFree.backEnd>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].backEnd ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1.moduleRef1[0].refModule.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>[0].refModule.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( ((uint64_t)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>) & (~0xFFFFULL)) | 0xABULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncModuleClose.module
<<lwscisync_module.LwSciSyncModuleClose.module>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncModuleClose.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleClose
TEST.NEW
TEST.NAME:TC_002.LwSciSyncModuleClose.module_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncModuleClose.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleClose() when input module is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{module points to NULL.}
 *
 * @testbehavior{The function just returns when input module is NULL.}
 *
 * @testcase{18853935}
 *
 * @verify{18844599}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncModuleClose.module:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleClose
  lwscisync_module.c.LwSciSyncModuleClose
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncModuleClose.Panic_due_to_ilwalid_module
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleClose
TEST.NEW
TEST.NAME:TC_003.LwSciSyncModuleClose.Panic_due_to_ilwalid_module
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncModuleClose.Panic_due_to_ilwalid_module}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleClose() when input module is invalid.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD}
 *
 * @testinput{ - module - points to valid LwSciSyncModule.}
 *
 * @testbehavior{Function aborts exelwtion/panics.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{22060361}
 *
 * @verify{18844599}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
if (<<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>) {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>;
} else {
  *<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = NULL;
}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_module.LwSciSyncModuleClose.module>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1.moduleRef1[0].refModule.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>[0].refModule.objPtr = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01.coreModule01[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>>[0].header = ( 13 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncModuleClose.module
<<lwscisync_module.LwSciSyncModuleClose.module>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncModuleOpen

-- Test Case: TC_002.LwSciSyncModuleOpen.LwSciSyncCoreRmAlloc_return_error
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleOpen
TEST.NEW
TEST.NAME:TC_002.LwSciSyncModuleOpen.LwSciSyncCoreRmAlloc_return_error
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncModuleOpen.LwSciSyncCoreRmAlloc_return_error}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleOpen() when LwSciSyncCoreRmAlloc() returns
 * value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - newModule - points to valid memory.}
 *
 * @testbehavior{ - Function should return LwSciError_ResourceError.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853941}
 *
 * @verify{18844596}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreRmAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:lwscisync_module.LwSciSyncModuleOpen.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonFreeObjAndRef.refCleanupCallback:<<null>>
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreRmAlloc
  uut_prototype_stubs.LwSciSyncCoreRmFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_module.c.LwSciSyncModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = malloc(sizeof(LwSciSyncCoreModule));
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = malloc(sizeof(struct LwSciSyncModuleRec));

CastRefToSyncModule(*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>)->refModule.objPtr = *<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> ;

<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreModule) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncModuleRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>[0].refModule ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncModuleOpen.newModule
<<lwscisync_module.LwSciSyncModuleOpen.newModule>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncModuleOpen.Success
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleOpen
TEST.NEW
TEST.NAME:TC_003.LwSciSyncModuleOpen.Success
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncModuleOpen.Success}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleOpen() when LwSciSyncCoreRmAlloc() returns
 * LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - newModule - points to valid memory.}
 *
 * @testbehavior{ - Function should return LwSciError_Success.
 * - newModule points to valid LwSciSyncModule with its members correctly initialized.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853944}
 *
 * @verify{18844596}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01:<<malloc 1>>
TEST.VALUE:lwscisync_module.LwSciSyncModuleOpen.newModule:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncModuleOpen.return:LwSciError_Success
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciSyncCoreRmAlloc
  lwscisync_module.c.LwSciSyncModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> = (  <<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
*<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreModule01>> );
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_Success );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreModule) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncModuleRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_module.LwSciSyncModuleOpen.newModule.newModule[0]
{{ <<lwscisync_module.LwSciSyncModuleOpen.newModule>>[0] == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>>[0] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncModuleOpen.Insufficient_memory
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleOpen
TEST.NEW
TEST.NAME:TC_004.LwSciSyncModuleOpen.Insufficient_memory
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncModuleOpen.Insufficient_memory}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleOpen() when LwSciCommonAllocObjWithRef() returns
 * value other than LwSciError_Success.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Mock expected behavior of stubs as per its SWUD.}
 *
 * @testinput{ - newModule - points to valid memory.}
 *
 * @testbehavior{Function should return LwSciError_InsufficientMemory.
 * - Function call sequence expected as per sequence diagram/code.
 * - Stub functions receive correct arguments.}
 *
 * @testcase{18853947}
 *
 * @verify{18844596}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1:<<malloc 1>>
TEST.EXPECTED:lwscisync_module.LwSciSyncModuleOpen.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  lwscisync_module.c.LwSciSyncModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.return>> = ( LwSciError_InsufficientMemory );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciSyncCoreModule) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(struct LwSciSyncModuleRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_module.LwSciSyncModuleOpen.newModule
<<lwscisync_module.LwSciSyncModuleOpen.newModule>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.moduleRef1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncModuleOpen.module_is_null
TEST.UNIT:lwscisync_module
TEST.SUBPROGRAM:LwSciSyncModuleOpen
TEST.NEW
TEST.NAME:TC_005.LwSciSyncModuleOpen.module_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncModuleOpen.module_is_null}
 *
 * @verifyFunction{This test case verifies the behavior of LwSciSyncModuleOpen() when input newModule is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed}
 *
 * @testinput{newModule points to NULL.}
 *
 * @testbehavior{Function should return LwSciError_BadParameter.}
 *
 * @testcase{18853950}
 *
 * @verify{18844596}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_module.LwSciSyncModuleOpen.newModule:<<null>>
TEST.EXPECTED:lwscisync_module.LwSciSyncModuleOpen.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_module.c.LwSciSyncModuleOpen
  lwscisync_module.c.LwSciSyncModuleOpen
TEST.END_FLOW
TEST.END
