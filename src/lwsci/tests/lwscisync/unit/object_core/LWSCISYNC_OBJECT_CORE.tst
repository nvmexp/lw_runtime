-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_OBJECT_CORE
-- Unit(s) Under Test: lwscisync_object_core
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

-- Subprogram: LwSciSyncCoreObjGetId

-- Test Case: TC_001.LwSciSyncCoreObjGetId.Successfully_GetsObjIdFromSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetId
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreObjGetId.Successfully_GetsObjIdFromSyncObj
TEST.BASIS_PATH:4 of 4
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncCoreObjGetId.Successfully_GetsObjIdFromSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjGetId() for successful retrieval of LwSciSyncCoreObjId from input LwSciSyncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to pointer of LwSciSyncCoreObjId.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - "objId" returned is same as LwSciSyncCoreObjId of "syncObj"}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854094}
*
* @verify{18844665}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].objId.moduleCntr:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].objId.ipcEndpoint:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetId.objId:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjGetId.objId[0].moduleCntr:1
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjGetId.objId[0].ipcEndpoint:2
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncCoreObjGetId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>->header =  LwSciSyncCoreGenerateObjHeader(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].moduleCntr == 1}}
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].ipcEndpoint == 2}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreObjGetId.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetId
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreObjGetId.Ilwalid_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncCoreObjGetId.Ilwalid_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetId() for invalid syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to invalid LwSciSyncObj.
* - objId set to pointer of LwSciSyncCoreObjId.}
*
* @testbehavior{- LwSciSyncCoreObjGetId() panics.
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
* @testcase{18854097}
*
* @verify{18844665}
*/

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetId.objId:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].moduleCntr == 1}}
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].ipcEndpoint == 2}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreObjGetId.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetId
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreObjGetId.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncCoreObjGetId.NULL_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetId() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.
* - objId set to pointer of LwSciSyncCoreObjId.}
*
* @testbehavior{- LwSciSyncCoreObjGetId() panics.
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
* @testcase{18854100}
*
* @verify{18844665}
*/

TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetId.objId:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:<<testcase>>
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].moduleCntr == 1}}
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].ipcEndpoint == 2}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreObjGetId.NULL_objId
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetId
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreObjGetId.NULL_objId
TEST.NOTES:
/**
* @testname{TC_004.LwSciSyncCoreObjGetId.NULL_objId}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetId() for NULL objId.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to NULL.}
*
* @testbehavior{- LwSciSyncCoreObjGetId() panics.
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
* @testcase{18854103}
*
* @verify{18844665}
*/

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetId.objId:<<null>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>->header =  LwSciSyncCoreGenerateObjHeader(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].moduleCntr == 1}}
{{<<lwscisync_object_core.LwSciSyncCoreObjGetId.objId>>[0].ipcEndpoint == 2}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreObjGetModule

-- Test Case: TC_001.LwSciSyncCoreObjGetModule.Successfully_GetModuleFromSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetModule
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreObjGetModule.Successfully_GetModuleFromSyncObj
TEST.BASIS_PATH:4 of 4 (partial)
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncCoreObjGetModule.Successfully_GetModuleFromSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjGetModule() for successful retrieval of LwSciSyncModule from input LwSciSyncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetModule().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - module set to pointer of LwSciSyncModule.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - "module" points to LwSciSyncModule from input "syncObj".
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854106}
*
* @verify{18844671}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetModule.module:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  lwscisync_object_core.c.LwSciSyncCoreObjGetModule
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetModule.module
{{ <<lwscisync_object_core.LwSciSyncCoreObjGetModule.module>> == ( <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreObjGetModule.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetModule
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreObjGetModule.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncCoreObjGetModule.NULL_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetModule() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetModule().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.
* - module set to pointer of LwSciSyncModule.}
*
* @testbehavior{- LwSciSyncCoreObjMatchId() panics.
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
* @testcase{18854109}
*
* @verify{18844671}
*/


TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetModule.module:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreObjGetModule.NULL_module
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetModule
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreObjGetModule.NULL_module
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncCoreObjGetModule.NULL_module}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetModule() for NULL module.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetModule().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - module set to NULL.}
*
* @testbehavior{- LwSciSyncCoreObjGetModule() panics.
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
* @testcase{18854112}
*
* @verify{18844671}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetModule.module:<<null>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetModule
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetModule.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreObjGetPrimitive

-- Test Case: TC_001.LwSciSyncCoreObjGetPrimitive.Successfully_GetPrimitiveFromSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreObjGetPrimitive.Successfully_GetPrimitiveFromSyncObj
TEST.BASIS_PATH:2 of 2 (partial)
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncCoreObjGetPrimitive.Successfully_GetPrimitiveFromSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjGetPrimitive() for successful retrieval of LwSciSyncCorePrimitive from input LwSciSyncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetPrimitive().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - primitive set to pointer of LwSciSyncCorePrimitive.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - "primitive" returned is same as LwSciSyncCorePrimitive of "syncObj"}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854115}
*
* @verify{18844674}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncCoreObjGetPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].primitive
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].primitive = ( malloc(10) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.primitive.primitive[0]
{{ <<lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.primitive>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].primitive ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreObjGetPrimitive.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreObjGetPrimitive.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncCoreObjGetPrimitive.NULL_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetPrimitive() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetPrimitive().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.
* - primitive set to pointer of LwSciSyncCorePrimitive.}
*
* @testbehavior{- LwSciSyncCoreObjGetPrimitive() panics.
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
* @testcase{18854118}
*
* @verify{18844674}
*/


TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreObjGetPrimitive.NULL_primitive
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjGetPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreObjGetPrimitive.NULL_primitive
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncCoreObjGetPrimitive.NULL_primitive}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjGetPrimitive() for NULL primitive.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjGetPrimitive().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - primitive set to NULL.}
*
* @testbehavior{- LwSciSyncCoreObjGetPrimitive() panics.
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
* @testcase{18854121}
*
* @verify{18844674}
*/


TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.primitive:<<null>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjGetPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjGetPrimitive.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreObjMatchId

-- Test Case: TC_001.LwSciSyncCoreObjMatchId.Success_InputObjIdMatchesSyncObjId
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjMatchId
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreObjMatchId.Success_InputObjIdMatchesSyncObjId
TEST.BASIS_PATH:5 of 5 (partial)
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncCoreObjMatchId.Success_InputObjIdMatchesSyncObjId}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjMatchId() when input LwSciSyncCoreObjId struct members matches with LwSciSyncCoreObjId of syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjMatchId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to pointer of LwSciSyncCoreObjId with its members initialized to values same as LwSciSyncCoreObjId of input syncObj.}
* - isEqual set to pointer of boolean.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - isEqual boolean value is updated to true}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854124}
*
* @verify{18844668}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.moduleCntr:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.ipcEndpoint:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].moduleCntr:1
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].ipcEndpoint:2
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual[0]:true
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[1].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual>>[0] == (true) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreObjMatchId.Success_InputObjIdmisMatchesSyncObjId
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjMatchId
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreObjMatchId.Success_InputObjIdmisMatchesSyncObjId
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncCoreObjMatchId.Success_InputObjIdmisMatchesSyncObjId}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjMatchId() when input LwSciSyncCoreObjId struct members do not match with LwSciSyncCoreObjId of syncObj}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjMatchId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to pointer of LwSciSyncCoreObjId with its members initialized to values different from LwSciSyncCoreObjId of input syncObj.}
* - isEqual set to pointer of boolean.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - isEqual boolean value is updated to false}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854127}
*
* @verify{18844668}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].objId.moduleCntr:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].objId.ipcEndpoint:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].moduleCntr:1
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].ipcEndpoint:2
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual[0]:false
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual>>[0] == (false) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreObjMatchId.NULL_objId
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjMatchId
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreObjMatchId.NULL_objId
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncCoreObjMatchId.NULL_objId}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjMatchId() for NULL objId.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjMatchId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to NULL.
* - isEqual set to pointer of boolean.}
*
* @testbehavior{- LwSciSyncCoreObjMatchId() panics.
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
* @testcase{18854130}
*
* @verify{18844668}
*/


TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.moduleCntr:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.ipcEndpoint:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[1].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual>>[0] == (true) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreObjMatchId.NULL_isEqual
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjMatchId
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreObjMatchId.NULL_isEqual
TEST.NOTES:
/**
* @testname{TC_004.LwSciSyncCoreObjMatchId.NULL_isEqual}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjMatchId() for NULL isEqual.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjMatchId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - objId set to pointer of LwSciSyncCoreObjId with its members initialized to values same as LwSciSyncCoreObjId of input syncObj.}
* - isEqual set to NULL.}
*
* @testbehavior{- LwSciSyncCoreObjMatchId() panics.
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
* @testcase{18854133}
*
* @verify{18844668}
*/


TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.moduleCntr:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[1].objId.ipcEndpoint:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].moduleCntr:1
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId[0].ipcEndpoint:2
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual:<<null>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[1].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1].header = (LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[1];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual>>[0] == (true) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreObjMatchId.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjMatchId
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreObjMatchId.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_005.LwSciSyncCoreObjMatchId.NULL_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjMatchId() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjMatchId().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.
* - objId set to pointer of LwSciSyncCoreObjId.}
* - isEqual set to pointer of boolean.}
*
* @testbehavior{- LwSciSyncCoreObjMatchId() panics.
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
* @testcase{18854136}
*
* @verify{18844668}
*/


TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.syncObj:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.objId:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjMatchId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscisync_object_core.LwSciSyncCoreObjMatchId.isEqual>>[0] == (true) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreObjValidate

-- Test Case: TC_001.LwSciSyncCoreObjValidate.Successfully_ValidatesSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjValidate
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreObjValidate.Successfully_ValidatesSyncObj
TEST.BASIS_PATH:3 of 4 (partial)
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncCoreObjValidate.Successfully_ValidatesSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjValidate() for successful validation of LwSciSyncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjValidate().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncCoreObjValidate() returns LwSciError_Success.
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
* @testcase{18854139}
*
* @verify{18844662}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncCoreObjValidate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>->header =  LwSciSyncCoreGenerateObjHeader(<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>;
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreObjValidate.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjValidate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreObjValidate.Ilwalid_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncCoreObjValidate.Ilwalid_syncObj}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreObjValidate() for invalid syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjValidate().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to invalid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncCoreObjValidate() panics.
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
* @testcase{18854142}
*
* @verify{18844662}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj
<<lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj>> = <<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>;
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreObjValidate.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncCoreObjValidate
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreObjValidate.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncCoreObjValidate.NULL_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncCoreObjValidate() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncCoreObjValidate().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.}
*
* @testbehavior{- LwSciSyncCoreObjValidate() returns LwSciError_BadParameter.
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
* @testcase{18854145}
*
* @verify{18844662}
*/
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncCoreObjValidate.syncObj:<<null>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncCoreObjValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncCoreObjValidate
  lwscisync_object_core.c.LwSciSyncCoreObjValidate
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncObjFreeObjAndRef

-- Test Case: TC_001.LwSciSyncObjFreeObjAndRef.Successfully_FreeSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjFreeObjAndRef
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjFreeObjAndRef.Successfully_FreeSyncObj
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncObjFreeObjAndRef.Successfully_FreeSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjFreeObjAndRef() for successful free of syncObj}
*
* @testpurpose{Unit testing of LwSciSyncObjFreeObjAndRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.}
*
* @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - syncObj members are deinited and its memory is freed.
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
* @verify{18844677}
*/
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciSyncCoreDeinitPrimitive
  uut_prototype_stubs.LwSciSyncAttrListFree
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscisync_object_core.c.LwSciSyncObjFreeObjAndRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.refCleanupCallback
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.refCleanupCallback>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncAttrListFree.attrList
{{ <<uut_prototype_stubs.LwSciSyncAttrListFree.attrList>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.attrList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreDeinitPrimitive.primitive
{{ <<uut_prototype_stubs.LwSciSyncCoreDeinitPrimitive.primitive>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.primitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj.header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.header = ( ((uint64_t)&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>> & (~0xFFFFULL)) | 0xEFULL );


TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj[0].refObj.objPtr
<<lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj>>[0].refObj.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.coreObj );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjFreeObjAndRef.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjFreeObjAndRef
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjFreeObjAndRef.Ilwalid_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncObjFreeObjAndRef.Ilwalid_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjFreeObjAndRef() for invalid syncObj}
*
* @testpurpose{Unit testing of LwSciSyncObjFreeObjAndRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to invalid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncObjFreeObjAndRef() panics.
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
* @verify{18844677}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj.header:0x1234ABCD
TEST.VALUE:lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjFreeObjAndRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.coreObj );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj>>[0].refObj ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj[0].refObj.objPtr
<<lwscisync_object_core.LwSciSyncObjFreeObjAndRef.syncObj>>[0].refObj.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_coreObj>>.coreObj );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjGetAttrList

-- Test Case: TC_001.LwSciSyncObjGetAttrList.Successfully_GetAttrListFromSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjGetAttrList
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjGetAttrList.Successfully_GetAttrListFromSyncObj
TEST.BASIS_PATH:4 of 4 (partial)
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncObjGetAttrList.Successfully_GetAttrListFromSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjGetAttrList() for successful retrieval of LwSciSyncAttrList from input LwSciSyncObj.}
*
* @testpurpose{Unit testing of LwSciSyncObjGetAttrList().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - syncAttrList set to pointer of LwSciSyncAttrList.}
*
* @testbehavior{- LwSciSyncObjGetAttrList() returns LwSciError_Success.
* - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.
* - "syncAttrList" returned is same as LwSciSyncAttrList in "syncObj"}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{18854148}
*
* @verify{18844656}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjGetAttrList.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].attrList
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].attrList = ( malloc(1) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj
<<lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList.syncAttrList[0]
{{ <<lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].attrList ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjGetAttrList.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjGetAttrList
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjGetAttrList.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncObjGetAttrList.NULL_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjGetAttrList() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncObjGetAttrList().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to NULL.
* - syncAttrList set to pointer of LwSciSyncAttrList.}
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
* @testcase{18854151}
*
* @verify{18844656}
*/
TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj:<<null>>
TEST.VALUE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList:<<malloc 1>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncObjGetAttrList.NULL_syncAttrList
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjGetAttrList
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjGetAttrList.NULL_syncAttrList
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncObjGetAttrList.NULL_syncAttrList}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjGetAttrList() for NULL syncAttrList.}
*
* @testpurpose{Unit testing of LwSciSyncObjGetAttrList().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to valid LwSciSyncObj.
* - syncAttrList set to NULL.}
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
* @testcase{18854154}
*
* @verify{18844656}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList:<<null>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjGetAttrList.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj
<<lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjGetAttrList.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjGetAttrList
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjGetAttrList.Ilwalid_syncObj
TEST.NOTES:
/**
* @testname{TC_004.LwSciSyncObjGetAttrList.Ilwalid_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjGetAttrList() for invalid syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncObjGetAttrList().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to invalid LwSciSyncObj.
* - syncAttrList set to pointer of LwSciSyncAttrList.}
*
* @testbehavior{- LwSciSyncObjGetAttrList() panics.
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
* @verify{18844656}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncAttrList:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjGetAttrList
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj
<<lwscisync_object_core.LwSciSyncObjGetAttrList.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncObjRef

-- Test Case: TC_001.LwSciSyncObjRef.Successfully_RefCountSyncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjRef
TEST.NEW
TEST.NAME:TC_001.LwSciSyncObjRef.Successfully_RefCountSyncObj
TEST.BASIS_PATH:3 of 3
TEST.NOTES:
/**
* @testname{TC_001.LwSciSyncObjRef.Successfully_RefCountSyncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjRef() for successful increment of reference count on the input LwSciSyncObj}
*
* @testpurpose{Unit testing of LwSciSyncObjRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{All the stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncObjRef() returns LwSciError_Success.
* - LwSciCommon functionality is called to increment ref count on input syncObj.
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
* @testcase{18854157}
*
* @verify{18844653}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonIncrAllRefCounts.return:LwSciError_Success
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjRef.return:LwSciError_Success
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonIncrAllRefCounts
  lwscisync_object_core.c.LwSciSyncObjRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonIncrAllRefCounts.ref
{{ <<uut_prototype_stubs.LwSciCommonIncrAllRefCounts.ref>> == ( <<lwscisync_object_core.LwSciSyncObjRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjRef.syncObj
<<lwscisync_object_core.LwSciSyncObjRef.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncObjRef.NULL_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjRef
TEST.NEW
TEST.NAME:TC_002.LwSciSyncObjRef.NULL_syncObj
TEST.NOTES:
/**
* @testname{TC_002.LwSciSyncObjRef.NULL_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjRef() for NULL syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncObjRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to null.}
*
* @testbehavior{- LwSciSyncObjRef() returns LwSciError_BadParameter.
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
* @testcase{18854160}
*
* @verify{18844653}
*/

TEST.END_NOTES:
TEST.VALUE:lwscisync_object_core.LwSciSyncObjRef.syncObj:<<null>>
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjRef.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjRef
  lwscisync_object_core.c.LwSciSyncObjRef
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncObjRef.Fail_DuetoOverflowOfReferences
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjRef
TEST.NEW
TEST.NAME:TC_003.LwSciSyncObjRef.Fail_DuetoOverflowOfReferences
TEST.NOTES:
/**
* @testname{TC_003.LwSciSyncObjRef.Fail_DuetoOverflowOfReferences}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjRef() if total number of references to syncObj are IN32_MAX and caller tries to take one more reference using this API.}
*
* @testpurpose{Unit testing of LwSciSyncObjRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{The following stub function(s) are simulated to return failure as per respective SWUD:
* LwSciCommonIncrAllRefCounts()
* All other stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- syncObj set to valid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncObjRef() returns LwSciError_IlwalidState.
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
* @testcase{18854163}
*
* @verify{18844653}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonIncrAllRefCounts.return:LwSciError_IlwalidState
TEST.EXPECTED:lwscisync_object_core.LwSciSyncObjRef.return:LwSciError_IlwalidState
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonIncrAllRefCounts
  lwscisync_object_core.c.LwSciSyncObjRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonIncrAllRefCounts.ref
{{ <<uut_prototype_stubs.LwSciCommonIncrAllRefCounts.ref>> == ( <<lwscisync_object_core.LwSciSyncObjRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj.coreObj[0].header
<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0].header = ( LwSciSyncCoreGenerateObjHeader(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0]));

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjRef.syncObj
<<lwscisync_object_core.LwSciSyncObjRef.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncObjRef.Ilwalid_syncObj
TEST.UNIT:lwscisync_object_core
TEST.SUBPROGRAM:LwSciSyncObjRef
TEST.NEW
TEST.NAME:TC_004.LwSciSyncObjRef.Ilwalid_syncObj
TEST.NOTES:
/**
* @testname{TC_004.LwSciSyncObjRef.Ilwalid_syncObj}
*
* @verifyFunction{This test case verifies functionality of API LwSciSyncObjRef() for invalid syncObj.}
*
* @testpurpose{Unit testing of LwSciSyncObjRef().}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{N/A}
*
* @testinput{- syncObj set to invalid LwSciSyncObj.}
*
* @testbehavior{- LwSciSyncObjRef() panics.
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
* @verify{18844653}
*/
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj[0].header:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj:<<malloc 1>>
TEST.FLOW
  lwscisync_object_core.c.LwSciSyncObjRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
*<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> = <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>>->objPtr;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( <<lwscisync_object_core.LwSciSyncObjRef.syncObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj.syncObj[0].refObj.objPtr
<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>[0].refObj.objPtr = &<<USER_GLOBALS_VCAST.<<GLOBAL>>.coreObj>>[0];
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_object_core.LwSciSyncObjRef.syncObj
<<lwscisync_object_core.LwSciSyncObjRef.syncObj>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.syncObj>>);
TEST.END_VALUE_USER_CODE:
TEST.END
