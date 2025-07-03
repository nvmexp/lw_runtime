-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCIBUF_MODULE
-- Unit(s) Under Test: lwscibuf_module
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

-- Unit: lwscibuf_module

-- Subprogram: LwSciBufCheckVersionCompatibility

-- Test Case: TC_001.LwSciBufCheckVersionCompatibility.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_001.LwSciBufCheckVersionCompatibility.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufCheckVersionCompatibility.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufCheckVersionCompatibility() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- majorVer is LwSciBufMajorVersion.
 * - LwSciBufCheckPlatformVersionCompatibility() '*platformCompatibility' to true.
 * - LwSciBufCheckPlatformVersionCompatibility() returns 'LwSciError_Success'.}
 *
 * @testinput{- majorVer is build major version
 * - minorVer is build minor version
 * - isCompatible points to a valid memory address.}
 *
 * @testbehavior{- LwSciBufCheckPlatformVersionCompatibility() is called to check platform version compatibility.
 * - returns 'LwSciError_Success' and '*isCompatible' as true.}
 *
 * @testcase{18857511}
 *
 * @verify{18842814}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.majorVer:MACRO=LwSciBufMajorVersion
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.minorVer:1
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_module.LwSciBufCheckVersionCompatibility.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
  uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
*<<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
{{ <<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible
{{ *<<lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible>> == ( true ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufCheckVersionCompatibility.majorVer_is_not_LwSciBufMajorVersion
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_002.LwSciBufCheckVersionCompatibility.majorVer_is_not_LwSciBufMajorVersion
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufCheckVersionCompatibility.majorVer_is_not_LwSciBufMajorVersion}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufCheckVersionCompatibility() majorVer is not LwSciBufMajorVersion.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- majorVer is not LwSciBufMajorVersion.
 * - LwSciBufCheckPlatformVersionCompatibility() '*platformCompatibility' to true.
 * - LwSciBufCheckPlatformVersionCompatibility() returns 'LwSciError_Success'.}
 *
 * @testinput{- majorVer is build major version
 * - minorVer is build minor version
 * - isCompatible points to a valid memory address.}
 *
 * @testbehavior{- LwSciBufCheckPlatformVersionCompatibility() is called to check platform version compatibility.
 * - returns 'LwSciError_Success' and '*isCompatible' as false.}
 *
 * @testcase{18857514}
 *
 * @verify{18842814}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.minorVer:1
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_module.LwSciBufCheckVersionCompatibility.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
  uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
*<<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
{{ <<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufCheckVersionCompatibility.majorVer
<<lwscibuf_module.LwSciBufCheckVersionCompatibility.majorVer>> = ( LwSciBufMajorVersion+1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible
{{ *<<lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible>> == ( false ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufCheckVersionCompatibility.platformCompatibility_is_false
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_003.LwSciBufCheckVersionCompatibility.platformCompatibility_is_false
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufCheckVersionCompatibility.platformCompatibility_is_false}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufCheckVersionCompatibility() platformCompatibility is false.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- majorVer is LwSciBufMajorVersion.
 * - LwSciBufCheckPlatformVersionCompatibility() '*platformCompatibility' to false.
 * - LwSciBufCheckPlatformVersionCompatibility() returns 'LwSciError_Success'.}
 *
 * @testinput{- majorVer is build major version
 * - minorVer is build minor version
 * - isCompatible points to a valid memory address.}
 *
 * @testbehavior{- LwSciBufCheckPlatformVersionCompatibility() is called to check platform version compatibility.
 * - returns 'LwSciError_Success' and '*isCompatible' as false.}
 *
 * @testcase{18857517}
 *
 * @verify{18842814}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.majorVer:MACRO=LwSciBufMajorVersion
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.minorVer:1
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_module.LwSciBufCheckVersionCompatibility.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
  uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
*<<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> = ( false );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility
{{ <<uut_prototype_stubs.LwSciBufCheckPlatformVersionCompatibility.platformCompatibility>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible
{{ *<<lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible>> == ( false ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufCheckVersionCompatibility.isCompatible_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_004.LwSciBufCheckVersionCompatibility.isCompatible_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufCheckVersionCompatibility.isCompatible_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufCheckVersionCompatibility() isCompatible is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- majorVer is LwSciBufMajorVersion.
 * - LwSciBufCheckPlatformVersionCompatibility() '*platformCompatibility' to true.
 * - LwSciBufCheckPlatformVersionCompatibility() returns 'LwSciError_Success'.}
 *
 * @testinput{- majorVer is build major version
 * - minorVer is build minor version
 * - isCompatible is null}
 *
 * @testbehavior{- LwSciBufCheckPlatformVersionCompatibility() is called to check platform version compatibility.
 * - returns 'LwSciError_Success' and '*isCompatible' as true.}
 *
 * @testcase{18857520}
 *
 * @verify{18842814}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.majorVer:MACRO=LwSciBufMajorVersion
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.minorVer:1
TEST.VALUE:lwscibuf_module.LwSciBufCheckVersionCompatibility.isCompatible:<<null>>
TEST.EXPECTED:lwscibuf_module.LwSciBufCheckVersionCompatibility.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
  lwscibuf_module.c.LwSciBufCheckVersionCompatibility
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufModuleClose

-- Test Case: TC_001.LwSciBufModuleClose.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleClose
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleClose.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleClose.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleClose() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_moduleObj.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to the address '0x9999999999999999'.
 * - 'test_moduleObj.iFaceOpenContext[1]' points to NULL.
 * - 'test_moduleObj.dev' points to the address '0x8888888888888888'.
 *
 * /** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.}
 *
 * @testbehavior{/** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * /** *** LwSciBufModuleCleanupObj *****************************************************************/
 * - LwSciBufAllocIfaceClose() is called to close the opaque open context of the given LwSciBufAllocIfaceType.
 * - LwSciBufDevClose() is called to free LwSciBufDev and deinitialize LwSciBufAllGpuContext.
 * /*************************************************************************************************/}
 *
 * @testcase{18857526}
 *
 * @verify{18842811}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[1]:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwSciBufAllocIfaceClose.allocType:LwSciBufAllocIfaceType_SysMem
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceClose
  uut_prototype_stubs.LwSciBufDevClose
  lwscibuf_module.c.LwSciBufModuleClose
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufAllocIfaceType) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufDevClose.dev
{{ <<uut_prototype_stubs.LwSciBufDevClose.dev>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceClose.context
{{ <<uut_prototype_stubs.LwSciBufAllocIfaceClose.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleClose.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleClose.dev_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleClose
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleClose.dev_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleClose.dev_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleClose() when LwSciBufDev object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_moduleObj.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to the address '0x9999999999999999'.
 * - 'test_moduleObj.iFaceOpenContext[1]' points to NULL.
 * - 'test_moduleObj.dev' points to the address '0xFFFFFFFFFFFFFFFF'.
 *
 * /** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/
 *
 * /** *** LwSciBufModuleCleanupObj *****************************************************************/
 * - LwSciBufDevClose() panics
 * /*************************************************************************************************/}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.}
 *
 * @testbehavior{/** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * /** *** LwSciBufModuleCleanupObj *****************************************************************/
 * - LwSciBufAllocIfaceClose() is called to close the opaque open context of the given LwSciBufAllocIfaceType.
 * - LwSciBufDevClose() is called to free LwSciBufDev and deinitialize LwSciBufAllGpuContext.
 * /*************************************************************************************************/
 * - LwSciBufModuleClose() panics.}
 *
 * @testcase{18857529}
 *
 * @verify{18842811}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[1]:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwSciBufAllocIfaceClose.allocType:LwSciBufAllocIfaceType_SysMem
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceClose
  uut_prototype_stubs.LwSciBufDevClose
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufAllocIfaceType) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufDevClose.dev
{{ <<uut_prototype_stubs.LwSciBufDevClose.dev>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceClose.context
{{ <<uut_prototype_stubs.LwSciBufAllocIfaceClose.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleClose.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleClose.moduleObj_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleClose
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleClose.moduleObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleClose.moduleObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleClose() when the module object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value '0x123'.
 *
 * /** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.}
 *
 * @testbehavior{/** *** LwSciBufModuleClose **********************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 * - LwSciBufModuleClose() panics.}
 *
 * @testcase{18857532}
 *
 * @verify{18842811}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:0x123
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleClose.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleClose.moduleObj_ref_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleClose
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleClose.moduleObj_ref_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleClose.moduleObj_ref_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleClose() when the module object reference is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciCommonGetObjFromRef() panics.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to NULL (invalid module reference).}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleClose() panics.}
 *
 * @testcase{18857535}
 *
 * @verify{18842811}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module[0].refHeader.objPtr:<<null>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleClose
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleClose.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleClose.module_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleClose
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleClose.module_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleClose.module_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleClose() when 'module' points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'module' points to NULL}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857538}
 *
 * @verify{18842811}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleClose.module:<<null>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleClose
  lwscibuf_module.c.LwSciBufModuleClose
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufModuleDupRef

-- Test Case: TC_001.LwSciBufModuleDupRef.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleDupRef.normal_operation
TEST.IMPORT_FAILURES:
(E) Errors from previous script import(s)
    >>> (E) @LINE: 614 TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.newModule.newModule[0].newModule[0][0].refHeader.objPtr
    >>>     >>> Expected a field name from the record type LwSciBufModuleRec
    >>>     >>> Read:     newModule
TEST.END_IMPORT_FAILURES:
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleDupRef.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleDupRef ********************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/
 *
 * - LwSciCommonDuplicateRef() points '*newRef' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonDuplicateRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'oldModule' points to a valid memory address.
 * - 'oldModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'newModule' points to a valid memory address.}
 *
 * @testbehavior{/** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleDupRef ********************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - LwSciCommonDuplicateRef() is called to create new LwSciRef structure pointed by newRef associated to the same LwSciObj as the input LwSciRef.
 * - returns 'LwSciError_Success' and '*newModule' with new LwSciBufModule.}
 *
 * @testcase{18857541}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleDupRef.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscibuf_module.c.LwSciBufModuleDupRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef.newRef[0]
<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>>[0] = ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader  );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.oldModule.oldModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.newModule.newModule[0]
{{ <<lwscibuf_module.LwSciBufModuleDupRef.newModule>>[0] == ( <<lwscibuf_module.LwSciBufModuleDupRef.oldModule>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleDupRef.failed_to_duplicate_module
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleDupRef.failed_to_duplicate_module
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleDupRef.failed_to_duplicate_module}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() when failed to duplicate module.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleDupRef ********************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/
 *
 * - LwSciCommonDuplicateRef() points '*newRef' to null.
 * - LwSciCommonDuplicateRef() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testinput{- 'oldModule' points to a valid memory address.
 * - 'oldModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'newModule' points to a valid memory address.}
 *
 * @testbehavior{/** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleDupRef ********************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - LwSciCommonDuplicateRef() is called to create new LwSciRef structur pointed by newRef associated to the same LwSciObj as the input LwSciRef.
 * - returns 'LwSciError_InsufficientMemory'.}
 *
 * @testcase{18857544}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonDuplicateRef.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleDupRef.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonDuplicateRef
  lwscibuf_module.c.LwSciBufModuleDupRef
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef.newRef[0]
<<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.oldRef>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonDuplicateRef.newRef
{{ <<uut_prototype_stubs.LwSciCommonDuplicateRef.newRef>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.oldModule.oldModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleDupRef.oldModuleObj_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleDupRef.oldModuleObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleDupRef.oldModuleObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() when the oldModule object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC+1'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'oldModule' points to a valid memory address.
 * - 'oldModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleValidate() panics.}
 *
 * @testcase{18857547}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule[0]:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.magic = ( LW_SCI_BUF_MODULE_MAGIC+1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0xFFFFFFFFFFFFFFFF  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.oldModule.oldModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleDupRef.oldModuleObj_ref_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleDupRef.oldModuleObj_ref_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleDupRef.oldModuleObj_ref_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() when the oldModule object reference is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC+1'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'oldModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'oldModule' points to a valid memory address.
 * - 'oldModule->refHeader.objPtr' points to null.
 * - 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleValidate() panics.}
 *
 * @testcase{18857550}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule[0].refHeader.objPtr:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule[0]:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleDupRef.oldModule_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleDupRef.oldModule_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleDupRef.oldModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() when 'oldModule' is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'oldModule' points to null.
 * - 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857553}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule[0]:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleDupRef.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  lwscibuf_module.c.LwSciBufModuleDupRef
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciBufModuleDupRef.newModule_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleDupRef
TEST.NEW
TEST.NAME:TC_006.LwSciBufModuleDupRef.newModule_is_null
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufModuleDupRef.newModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleDupRef() when 'newModule' is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.}
 *
 * @testinput{- 'oldModule' points to a valid memory address.
 * - 'oldModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'newModule' points to null.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857556}
 *
 * @verify{18842823}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.oldModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleDupRef.newModule:<<null>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleDupRef.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleDupRef
  lwscibuf_module.c.LwSciBufModuleDupRef
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleDupRef.oldModule.oldModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleDupRef.oldModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufModuleGetAllocIfaceOpenContext

-- Test Case: TC_001.LwSciBufModuleGetAllocIfaceOpenContext.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetAllocIfaceOpenContext
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleGetAllocIfaceOpenContext.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleGetAllocIfaceOpenContext.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetAllocIfaceOpenContext() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_objModule.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to a valid memory address.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleGetAllocIfaceOpenContext ***************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'allocType' is SysMem
 * - 'openContext' points to a valid memory address.}
 *
 * @testbehavior{/** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleGetAllocIfaceOpenContext ***************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18857559}
 *
 * @verify{18842832}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x5555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] = ( 0xAAAA );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext
{{ *<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleGetAllocIfaceOpenContext.iFaceOpenContext_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetAllocIfaceOpenContext
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleGetAllocIfaceOpenContext.iFaceOpenContext_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleGetAllocIfaceOpenContext.iFaceOpenContext_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetAllocIfaceOpenContext() iFaceOpenContext is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_objModule.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to null.
 *
 * /** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** LwSciBufModuleGetAllocIfaceOpenContext ***************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'allocType' is SysMem
 * - 'openContext' points to a valid memory address.}
 *
 * @testbehavior{/** *** LwSciBufModuleValidate *******************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** LwSciBufModuleGetAllocIfaceOpenContext ***************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - returns 'LwSciError_ResourceError'.}
 *
 * @testcase{18857562}
 *
 * @verify{18842832}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.return:LwSciError_ResourceError
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] = ( 0xAAAA );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext
{{ *<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleGetAllocIfaceOpenContext.moduleObj_ref_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetAllocIfaceOpenContext
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleGetAllocIfaceOpenContext.moduleObj_ref_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleGetAllocIfaceOpenContext.moduleObj_ref_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetAllocIfaceOpenContext() when the module object reference is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciCommonGetObjFromRef() panics.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to NULL (invalid module reference).
 * - 'allocType' is SysMem
 * - 'openContext' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleGetAllocIfaceOpenContext() panics.}
 *
 * @testcase{18857565}
 *
 * @verify{18842832}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module[0].refHeader.objPtr:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleGetAllocIfaceOpenContext.openContext_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetAllocIfaceOpenContext
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleGetAllocIfaceOpenContext.openContext_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleGetAllocIfaceOpenContext.openContext_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetAllocIfaceOpenContext() when the openContext is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_objModule.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to a valid memory address.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'allocType' is SysMem
 * - 'openContext' points to null.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857571}
 *
 * @verify{18842832}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext:<<null>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x5555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] = ( 0xAAAA );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufModuleGetAllocIfaceOpenContext.allocType_is_Max
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetAllocIfaceOpenContext
TEST.NEW
TEST.NAME:TC_006.LwSciBufModuleGetAllocIfaceOpenContext.allocType_is_Max
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufModuleGetAllocIfaceOpenContext.allocType_is_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetAllocIfaceOpenContext() when the allocType is Max.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test_objModule.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]' points to a valid memory address.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'allocType' is Max
 * - 'openContext' points to a valid memory address.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857574}
 *
 * @verify{18842832}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.openContext:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
  lwscibuf_module.c.LwSciBufModuleGetAllocIfaceOpenContext
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] = ( 0x5555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext.iFaceOpenContext[1]
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] = ( 0xAAAA );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetAllocIfaceOpenContext.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufModuleGetDevHandle

-- Test Case: TC_001.LwSciBufModuleGetDevHandle.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetDevHandle
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleGetDevHandle.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleGetDevHandle.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetDevHandle() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'dev' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18857577}
 *
 * @verify{18842820}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetDevHandle.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0x55555555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetDevHandle.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev
{{ <<lwscibuf_module.LwSciBufModuleGetDevHandle.dev>>[0]  == ( 0x55555555 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleGetDevHandle.moduleObj_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetDevHandle
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleGetDevHandle.moduleObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleGetDevHandle.moduleObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetDevHandle() when the module object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC+1'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'dev' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleGetDevHandle panics.}
 *
 * @testcase{18857580}
 *
 * @verify{18842820}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.magic = ( LW_SCI_BUF_MODULE_MAGIC+1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0x55555555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetDevHandle.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleGetDevHandle.moduleObj_ref_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetDevHandle
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleGetDevHandle.moduleObj_ref_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleGetDevHandle.moduleObj_ref_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetDevHandle() when the module object reference is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciCommonGetObjFromRef() panics.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'dev' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleGetDevHandle panics.}
 *
 * @testcase{18857583}
 *
 * @verify{18842820}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module[0].refHeader.objPtr:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleGetDevHandle.moduleObj_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetDevHandle
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleGetDevHandle.moduleObj_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleGetDevHandle.moduleObj_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetDevHandle() when the module object is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to null.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'dev' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleGetDevHandle panics.}
 *
 * @testcase{18857586}
 *
 * @verify{18842820}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev = ( 0x55555555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleGetDevHandle.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleGetDevHandle.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleGetDevHandle.module_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleGetDevHandle
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleGetDevHandle.module_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleGetDevHandle.module_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleGetDevHandle() when 'module' points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'module' points to null
 * - 'dev' points to a valid memory address.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857589}
 *
 * @verify{18842820}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.module:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleGetDevHandle.dev:<<malloc 1>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleGetDevHandle.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
  lwscibuf_module.c.LwSciBufModuleGetDevHandle
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufModuleIsEqual

-- Test Case: TC_001.LwSciBufModuleIsEqual.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleIsEqual.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleIsEqual.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * /** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'firstModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'secondModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{/** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - returns 'LwSciError_Success' and '*isEqual' is true.}
 *
 * @testcase{18857592}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:false
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleIsEqual.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleIsEqual
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int count1=0;
count1++;

if(1==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr );
else if(2==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count=0;
count++;

if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual
{{ *<<lwscibuf_module.LwSciBufModuleIsEqual.isEqual>> == ( true ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleIsEqual.firstModule_object_is_not_secondModule_object
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleIsEqual.firstModule_object_is_not_secondModule_object
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleIsEqual.firstModule_object_is_not_secondModule_object}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when firstModule object is not secondModule object.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test1_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * /** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'firstModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'secondModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test1_objModule.objHeader'.
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{/** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - returns 'LwSciError_Success' and '*isEqual' is false.}
 *
 * @testcase{18857595}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:true
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleIsEqual.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleIsEqual
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int count1=0;
count1++;

if(1==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr );
else if(2==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count=0;
count++;

if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual
{{ *<<lwscibuf_module.LwSciBufModuleIsEqual.isEqual>> == ( false ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleIsEqual.secondModule_object_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleIsEqual.secondModule_object_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleIsEqual.secondModule_object_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when the secondModule object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 * - 'test1_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC+1'.
 *
 * /** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'firstModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'secondModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test1_objModule.objHeader'.
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{/** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - LwSciBufModuleIsEqual() panics.}
 *
 * @testcase{18857598}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:true
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int count1=0;
count1++;

if(1==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr );
else if(2==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count=0;
count++;

if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj.magic
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj>>.magic = ( LW_SCI_BUF_MODULE_MAGIC+1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleIsEqual.firstModule_object_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleIsEqual.firstModule_object_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleIsEqual.firstModule_object_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when the firstModule object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC+1'.
 * - 'test1_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * /** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'firstModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'secondModule->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.
 * /*************************************************************************************************/}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test1_objModule.objHeader'.
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{/** *** firstModule ******************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 *
 * /** *** secondModule ****************************************************************************/
 * - LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * /*************************************************************************************************/
 *
 * - LwSciBufModuleIsEqual() panics.}
 *
 * @testcase{18857601}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:true
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
static int count1=0;
count1++;

if(1==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr );
else if(2==count1)
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
static int count=0;
count++;

if(1==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader ) }}
else if(2==count)
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.magic = ( LW_SCI_BUF_MODULE_MAGIC+1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test1_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleIsEqual.firstModule_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleIsEqual.firstModule_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleIsEqual.firstModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when firstModule is null.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.}
 *
 * @testinput{- 'firstModule' points to null
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter' and '*isEqual' is false.}
 *
 * @testcase{18857604}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:false
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleIsEqual.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  lwscibuf_module.c.LwSciBufModuleIsEqual
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual
{{ *<<lwscibuf_module.LwSciBufModuleIsEqual.isEqual>> == ( false ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufModuleIsEqual.secondModule_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_006.LwSciBufModuleIsEqual.secondModule_is_null
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufModuleIsEqual.secondModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when secondModule is null.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to null
 * - 'isEqual' points to a valid memory address.}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter' and '*isEqual' is false.}
 *
 * @testcase{18857607}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual[0]:false
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleIsEqual.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  lwscibuf_module.c.LwSciBufModuleIsEqual
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = (  &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual
{{ *<<lwscibuf_module.LwSciBufModuleIsEqual.isEqual>> == ( false ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufModuleIsEqual.isEqual_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleIsEqual
TEST.NEW
TEST.NAME:TC_007.LwSciBufModuleIsEqual.isEqual_is_null
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufModuleIsEqual.isEqual_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleIsEqual() when isEqual is null.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.}
 *
 * @testinput{- 'firstModule' points to a valid memory address.
 * - 'firstModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'secondModule' points to a valid memory address.
 * - 'secondModule->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.
 * - 'isEqual' points to null}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857610}
 *
 * @verify{18842826}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleIsEqual.isEqual:<<null>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleIsEqual.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleIsEqual
  lwscibuf_module.c.LwSciBufModuleIsEqual
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.firstModule.firstModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.firstModule>>[0].refHeader.objPtr = (  &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleIsEqual.secondModule.secondModule[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleIsEqual.secondModule>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufModuleOpen

-- Test Case: TC_001.LwSciBufModuleOpen.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleOpen
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleOpen.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleOpen.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleOpen() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- test_moduleRef.refHeader.objPtr points to VECTORCAST_INT1.
 * - LwSciCommonAllocObjWithRef() points '*objPtr' to 'test_objModule.objHeader'.
 * - LwSciCommonAllocObjWithRef() points '*refPtr' to 'test_moduleRef.refHeader'.
 * - LwSciCommonAllocObjWithRef() returns 'LwSciError_Success'.
 *
 * - LwSciBufDevOpen() points '*newDev' to 0x5555.
 * - LwSciBufDevOpen() returns 'LwSciError_Success'.
 *
 * - LwSciBufAllocIfaceOpen() points '*context' to 0x123.
 * - LwSciBufAllocIfaceOpen() returns 'LwSciError_Success'.
 *
 * - LwSciBufAllocIfaceOpen() points '*context' to 0x456.
 * - LwSciBufAllocIfaceOpen() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonAllocObjWithRef() is called to initialize LwSciRef and LwSciObj structure.
 * - LwSciBufDevOpen() is called to allocate a new LwSciBufDev.
 * - LwSciBufAllocIfaceOpen() is called to create allocation interface corresponding to LwSciBufAllocIfaceType(SysMem).
 * - LwSciBufAllocIfaceOpen() is called to create allocation interface corresponding to LwSciBufAllocIfaceType(1).
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18857613}
 *
 * @verify{18842808}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef.refHeader.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciBufDevOpen.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleOpen.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem,1
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciBufDevOpen
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceOpen
  lwscibuf_module.c.LwSciBufModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
*<<uut_prototype_stubs.LwSciBufDevOpen.newDev>> = ( 0x5555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.context
static int count = 0;
count++;

if(1==count)
*<<uut_prototype_stubs.LwSciBufAllocIfaceOpen.context>> = ( 0x123 );
else if(2==count)
*<<uut_prototype_stubs.LwSciBufAllocIfaceOpen.context>> = ( 0x456 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufAllocIfaceType) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
{{ <<uut_prototype_stubs.LwSciBufDevOpen.newDev>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciBufModuleObjPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciBufModuleRefPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.devHandle
{{ <<uut_prototype_stubs.LwSciBufAllocIfaceOpen.devHandle>> == ( 0x5555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]
{{ <<lwscibuf_module.LwSciBufModuleOpen.newModule>>[0] == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev == ( 0x5555 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] == ( 0x123 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[1]
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] == ( 0x456 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleOpen.LwSciBufAllocIfaceType_is_not_supported
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleOpen
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleOpen.LwSciBufAllocIfaceType_is_not_supported
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleOpen.LwSciBufAllocIfaceType_is_not_supported}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleOpen() when LwSciBufAllocIfaceType is not supported.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- test_moduleRef.refHeader.objPtr points to VECTORCAST_INT1.
 * - LwSciCommonAllocObjWithRef() points '*objPtr' to 'test_objModule.objHeader'.
 * - LwSciCommonAllocObjWithRef() points '*refPtr' to 'test_moduleRef.refHeader'.
 * - LwSciCommonAllocObjWithRef() returns 'LwSciError_Success'.
 *
 * - LwSciBufDevOpen() points '*newDev' to 0x5555.
 * - LwSciBufDevOpen() returns 'LwSciError_Success'.
 *
 * - LwSciBufAllocIfaceOpen() points '*context' to 0x123.
 * - LwSciBufAllocIfaceOpen() returns 'LwSciError_Success'.
 *
 * - LwSciBufAllocIfaceOpen() points '*context' to 0x456.
 * - LwSciBufAllocIfaceOpen() returns 'LwSciError_NotSupported'.}
 *
 * @testinput{- 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonAllocObjWithRef() is called to initialize LwSciRef and LwSciObj structure.
 * - LwSciBufDevOpen() is called to allocate a new LwSciBufDev.
 * - LwSciBufAllocIfaceOpen() is called to create allocation interface corresponding to LwSciBufAllocIfaceType(SysMem).
 * - LwSciBufAllocIfaceOpen() is called to create allocation interface corresponding to LwSciBufAllocIfaceType(1).
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18857616}
 *
 * @verify{18842808}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef.refHeader.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciBufDevOpen.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.return:LwSciError_Success,LwSciError_NotSupported
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleOpen.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem,1
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciBufDevOpen
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufAllocIfaceOpen
  lwscibuf_module.c.LwSciBufModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
*<<uut_prototype_stubs.LwSciBufDevOpen.newDev>> = ( 0x5555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.context
static int count = 0;
count++;

if(1==count)
*<<uut_prototype_stubs.LwSciBufAllocIfaceOpen.context>> = ( 0x123 );
else if(2==count)
*<<uut_prototype_stubs.LwSciBufAllocIfaceOpen.context>> = ( 0x456 );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufAllocIfaceType) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
{{ <<uut_prototype_stubs.LwSciBufDevOpen.newDev>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciBufModuleObjPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciBufModuleRefPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufAllocIfaceOpen.devHandle
{{ <<uut_prototype_stubs.LwSciBufAllocIfaceOpen.devHandle>> == ( 0x5555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]
{{ <<lwscibuf_module.LwSciBufModuleOpen.newModule>>[0] == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev == ( 0x5555 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem]
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[LwSciBufAllocIfaceType_SysMem] == ( 0x123 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.iFaceOpenContext[1]
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.iFaceOpenContext[1] == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleOpen.failed_to_allocate_new_LwSciBufDev
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleOpen
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleOpen.failed_to_allocate_new_LwSciBufDev
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleOpen.failed_to_allocate_new_LwSciBufDev}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleOpen() when failed to allocate new LwSciBufDev.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- test_moduleRef.refHeader.objPtr points to VECTORCAST_INT1.
 * - LwSciCommonAllocObjWithRef() points '*objPtr' to 'test_objModule.objHeader'.
 * - LwSciCommonAllocObjWithRef() points '*refPtr' to 'test_moduleRef.refHeader'.
 * - LwSciCommonAllocObjWithRef() returns 'LwSciError_Success'.
 *
 * - LwSciBufDevOpen() points '*newDev' to null.
 * - LwSciBufDevOpen() returns 'LwSciError_ResourceError'.}
 *
 * @testinput{- 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonAllocObjWithRef() is called to initialize LwSciRef and LwSciObj structure.
 * - LwSciBufDevOpen() is called to allocate a new LwSciBufDev.
 * - LwSciCommonFreeObjAndRef() is called to deallocate object and reference.
 * - returns 'LwSciError_ResourceError'.}
 *
 * @testcase{18857619}
 *
 * @verify{18842808}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef.refHeader.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciBufDevOpen.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleOpen.return:LwSciError_ResourceError
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  uut_prototype_stubs.LwSciBufDevOpen
  uut_prototype_stubs.LwSciCommonFreeObjAndRef
  lwscibuf_module.c.LwSciBufModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr.refPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
*<<uut_prototype_stubs.LwSciBufDevOpen.newDev>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciBufModuleObjPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciBufModuleRefPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.ref>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.objCleanupCallback>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFreeObjAndRef.refCleanupCallback
{{ <<uut_prototype_stubs.LwSciCommonFreeObjAndRef.refCleanupCallback>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufDevOpen.newDev
{{ <<uut_prototype_stubs.LwSciBufDevOpen.newDev>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleOpen.failed_to_allocate_LwSciRef_and_LwSciObj
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleOpen
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleOpen.failed_to_allocate_LwSciRef_and_LwSciObj
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleOpen.failed_to_allocate_LwSciRef_and_LwSciObj}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleOpen() when failed to allocate LwSciRef and LwSciObj.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- test_moduleRef.refHeader.objPtr points to VECTORCAST_INT1.
 * - LwSciCommonAllocObjWithRef() points '*objPtr' to 'test_objModule.objHeader'.
 * - LwSciCommonAllocObjWithRef() points '*refPtr' to 'test_moduleRef.refHeader'.
 * - LwSciCommonAllocObjWithRef() returns 'LwSciError_ResourceError'.}
 *
 * @testinput{- 'newModule' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonAllocObjWithRef() is called to initialize LwSciRef and LwSciObj structure.
 * - returns 'LwSciError_ResourceError'.}
 *
 * @testcase{18857622}
 *
 * @verify{18842808}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef.refHeader.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule[0]:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleOpen.return:LwSciError_ResourceError
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleOpen
  uut_prototype_stubs.LwSciCommonAllocObjWithRef
  lwscibuf_module.c.LwSciBufModuleOpen
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr.refPtr[0]
<<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>>[0] = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef>>.refHeader );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objSize>> == ( sizeof(LwSciBufModuleObjPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refSize>> == ( sizeof(LwSciBufModuleRefPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr
{{ <<uut_prototype_stubs.LwSciCommonAllocObjWithRef.refPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufModuleOpen.newModule_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleOpen
TEST.NEW
TEST.NAME:TC_005.LwSciBufModuleOpen.newModule_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufModuleOpen.newModule_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleOpen() when 'newModule' is null.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- test_moduleRef.refHeader.objPtr points to VECTORCAST_INT1.}
 *
 * @testinput{- 'newModule' points to null}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857625}
 *
 * @verify{18842808}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleRef.refHeader.objPtr:VECTORCAST_INT1
TEST.VALUE:lwscibuf_module.LwSciBufModuleOpen.newModule:<<null>>
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleOpen.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleOpen
  lwscibuf_module.c.LwSciBufModuleOpen
TEST.END_FLOW
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.dev
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.dev == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Subprogram: LwSciBufModuleValidate

-- Test Case: TC_001.LwSciBufModuleValidate.normal_operation
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleValidate
TEST.NEW
TEST.NAME:TC_001.LwSciBufModuleValidate.normal_operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufModuleValidate.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleValidate() for normal operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value 'LW_SCI_BUF_MODULE_MAGIC'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - returns 'LwSciError_Success'.}
 *
 * @testcase{18857628}
 *
 * @verify{18842829}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:MACRO=LW_SCI_BUF_MODULE_MAGIC
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleValidate.return:LwSciError_Success
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  lwscibuf_module.c.LwSciBufModuleValidate
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleValidate.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufModuleValidate.moduleObj_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleValidate
TEST.NEW
TEST.NAME:TC_002.LwSciBufModuleValidate.moduleObj_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufModuleValidate.moduleObj_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleValidate() when the module object is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_objModule.magic' is set to the value '0x123'.
 *
 * - LwSciCommonGetObjFromRef() points '*objPtr' to 'module->refHeader.objPtr'.
 * - LwSciCommonGetObjFromRef() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to the address of 'test_objModule.objHeader'.}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleValidate() panics.}
 *
 * @testcase{18857631}
 *
 * @verify{18842829}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj.magic:0x123
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.module:<<malloc 1>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr.objPtr[0]
<<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>>[0] = ( <<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader.objPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_module.LwSciBufModuleValidate.module.module[0].refHeader.objPtr
<<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader.objPtr = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_moduleObj>>.objHeader );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufModuleValidate.moduleObj_ref_is_ilwalid
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleValidate
TEST.NEW
TEST.NAME:TC_003.LwSciBufModuleValidate.moduleObj_ref_is_ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufModuleValidate.moduleObj_ref_is_ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleValidate() when the module object reference is invalid.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- LwSciCommonGetObjFromRef() panics.}
 *
 * @testinput{- 'module' points to a valid memory address.
 * - 'module->refHeader.objPtr' points to NULL (invalid module reference).}
 *
 * @testbehavior{- LwSciCommonGetObjFromRef() is called to retrieve LwSciObj object associated with the input LwSciRef object.
 * - LwSciBufModuleValidate() panics.}
 *
 * @testcase{18857634}
 *
 * @verify{18842829}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.module:<<malloc 1>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.module[0].refHeader.objPtr:<<null>>
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleValidate
  uut_prototype_stubs.LwSciCommonGetObjFromRef
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.ref
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.ref>> == ( &<<lwscibuf_module.LwSciBufModuleValidate.module>>[0].refHeader ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr
{{ <<uut_prototype_stubs.LwSciCommonGetObjFromRef.objPtr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufModuleValidate.module_is_null
TEST.UNIT:lwscibuf_module
TEST.SUBPROGRAM:LwSciBufModuleValidate
TEST.NEW
TEST.NAME:TC_004.LwSciBufModuleValidate.module_is_null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufModuleValidate.module_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufModuleValidate() when 'module' points to NULL.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{- 'module' points to points to NULL}
 *
 * @testbehavior{- returns 'LwSciError_BadParameter'.}
 *
 * @testcase{18857637}
 *
 * @verify{18842829}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.module:<<null>>
TEST.VALUE:lwscibuf_module.LwSciBufModuleValidate.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_module.LwSciBufModuleValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_module.c.LwSciBufModuleValidate
  lwscibuf_module.c.LwSciBufModuleValidate
TEST.END_FLOW
TEST.END


