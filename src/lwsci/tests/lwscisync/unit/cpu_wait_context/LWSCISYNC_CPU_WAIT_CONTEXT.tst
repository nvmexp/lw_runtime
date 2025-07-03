-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCISYNC_CPU_WAIT_CONTEXT
-- Unit(s) Under Test: lwscisync_cpu_wait_context
-- 
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Subprogram: LwSciSyncCoreCpuWaitContextGetBackEnd

-- Test Case: TC_001.LwSciSyncCoreCpuWaitContextGetBackEnd.return_waitcontext
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextGetBackEnd
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreCpuWaitContextGetBackEnd.return_waitcontext
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreCpuWaitContextGetBackEnd.return_waitcontext}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextGetBackEnd() for success use-case
 * when backEnd is returned correctly from input context}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context[0].waitContextBackEnd points to the adddress '0X8888888888888888'}
 *
 * @testbehavior{LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address 0X8888888888888888}
 *
 * @testcase{18853956}
 *
 * @verify{18844452}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.context:<<malloc 1>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetBackEnd
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetBackEnd
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.context.context[0].waitContextBackEnd
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.context>>[0].waitContextBackEnd = ( 0X8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0X5555555555555555 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.return
{{ <<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> == ( 0X8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreCpuWaitContextGetBackEnd.context_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextGetBackEnd
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreCpuWaitContextGetBackEnd.context_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreCpuWaitContextGetBackEnd.context_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextGetBackEnd() for panic use-case when context is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18853959}
 *
 * @verify{18844452}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetBackEnd.context:<<null>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreCpuWaitContextGetModule

-- Test Case: TC_001.LwSciSyncCoreCpuWaitContextGetModule.return_module
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextGetModule
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreCpuWaitContextGetModule.return_module
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreCpuWaitContextGetModule.return_module}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextGetModule() for success use-case
 * when module is returned correctly from input context.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context[0].module points to the adddress '0X555555555555555555'}
 *
 * @testbehavior{LwSciSyncCoreCpuWaitContextGetModule() returns the address '0X555555555555555555'}
 *
 * @testcase{18853962}
 *
 * @verify{18844449}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.context:<<malloc 1>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetModule
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetModule
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.context>>[0].module = ( 0X555555555555555555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.return
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.return>> = ( 0X2222222222222222 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.return
{{ <<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.return>> == ( 0X555555555555555555 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreCpuWaitContextGetModule.context_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextGetModule
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreCpuWaitContextGetModule.context_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreCpuWaitContextGetModule.context_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextGetModule() for panic use-case when context is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18853965}
 *
 * @verify{18844449}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextGetModule.context:<<null>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextGetModule
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreCpuWaitContextValidate

-- Test Case: TC_001.LwSciSyncCoreCpuWaitContextValidate.context_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextValidate
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreCpuWaitContextValidate.context_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreCpuWaitContextValidate.context_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextValidate() for failure use-case when context is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context is set to NULL}
 *
 * @testbehavior{returned LwSciError_BadParameter}
 *
 * @testcase{18853968}
 *
 * @verify{18844446}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context:<<null>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreCpuWaitContextValidate.context_Ilwalid
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextValidate
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreCpuWaitContextValidate.context_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreCpuWaitContextValidate.context_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextValidate() for panic use-case when context is Invalid}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context set to Invalid}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18853971}
 *
 * @verify{18844446}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context[0].header:0xFFFFFFFFFFFFFFFF
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreCpuWaitContextValidate.Failure_due_to_module_is_null
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextValidate
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreCpuWaitContextValidate.Failure_due_to_module_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreCpuWaitContextValidate.Failure_due_to_module_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextValidate() for failure use-case
 * where LwSciSyncCoreModuleValidate() returns error when module points to NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() returns LwSciError_Badparameter}
 *
 * @testinput{context set to valid memory
 * context[0].header set to a predefined value
 * context[0].module set to NULL
 *
 * @testbehavior{returned LwSciError_BadParameter}
 *
 * @testcase{18853974}
 *
 * @verify{18844446}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].header
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].header = ( (((uint64_t)(&<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0]) & (0xFFFF00000000FFFFULL)) | 0x123456780000ULL) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].module = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreCpuWaitContextValidate.rmWaitCtxBackEnd_Ilwalid
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextValidate
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreCpuWaitContextValidate.rmWaitCtxBackEnd_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreCpuWaitContextValidate.rmWaitCtxBackEnd_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextValidate() for failure use-case
 * where LwSciSyncCoreRmWaitCtxBackEndValidate() returns error}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() returns LwSciError_Success
 * LwSciSyncCoreRmWaitCtxBackEndValidate() returns LwSciError_BadParameter}
 *
 * @testinput{context set to valid memory
 * context[0].header set to a predefined value
 * context[0].module set to the address '0X5555555555555555')
 * context[0].waitContextBackEnd set to NULL}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18853977}
 *
 * @verify{18844446}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate.return:LwSciError_BadParameter
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0X5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].header
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].header = ( (((uint64_t)(&<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0]) & (0xFFFF00000000FFFFULL)) | 0x123456780000ULL) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].module = ( 0X5555555555555555 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].waitContextBackEnd
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].waitContextBackEnd = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreCpuWaitContextValidate.normal_operation
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCoreCpuWaitContextValidate
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreCpuWaitContextValidate.normal_operation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreCpuWaitContextValidate.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreCpuWaitContextValidate() for success use-case 
 * when the successful validation of input context}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context is assigned to valid memory
 * context[0].header set to a predefined value
 * Context[0].module set to the address '0X5050505050505050'
 * context[0].waitContextBackEnd set to the address '0X2525252525252525'}
 *
 * @testbehavior{LwSciError_Success returned
 * LwSciSyncCoreModuleValidate() receives correct arguments
 * LwSciSyncCoreRmWaitCtxBackEndValidate() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18853980}
 *
 * @verify{18844446}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCoreCpuWaitContextValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndValidate.waitContextBackEnd>> == ( 0X2525252525252525 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0X5050505050505050 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].header
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].header = ( (((uint64_t)(&<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0]) & (0xFFFF00000000FFFFULL)) | 0x123456780000ULL) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].module = ( 0X5050505050505050 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context.context[0].waitContextBackEnd
<<lwscisync_cpu_wait_context.LwSciSyncCoreCpuWaitContextValidate.context>>[0].waitContextBackEnd = ( 0X2525252525252525 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCpuWaitContextAlloc

-- Test Case: TC_001.LwSciSyncCpuWaitContextAlloc.module_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCpuWaitContextAlloc.module_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCpuWaitContextAlloc.module_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for failure use-case when module is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() return to LwSciError_BadParameter}
 *
 * @testinput{module set to NULL
 * newContext set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18853983}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCpuWaitContextAlloc.newContext_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCpuWaitContextAlloc.newContext_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCpuWaitContextAlloc.newContext_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for failure use-case when newContext is NULL.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() return to LwSciError_Success}
 *
 * @testinput{module points to the adddress '0X5555555555555555')
 * newContext set to NULL}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct aruguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18853986}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<null>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0X5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( 0X5555555555555555 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCpuWaitContextAlloc.module_Ilwalid
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCpuWaitContextAlloc.module_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCpuWaitContextAlloc.module_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for panic use-case
 * where LwSciSyncCpuWaitContextAlloc() panics when module is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() panics}
 *
 * @testinput{module set to value representing invalid LwSciSyncModule (0xFFFFFFFFFFFFFFFF)
 * newContext set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18853989}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCpuWaitContextAlloc.LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCpuWaitContextAlloc.LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCpuWaitContextAlloc.LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for failure use-case when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL}
 *
 * @testinput{Module set to a valid memory
 * newContext set to a valid memory}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18853992}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCpuWaitContextRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCpuWaitContextAlloc.Not_enough_system_resources
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCpuWaitContextAlloc.Not_enough_system_resources
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCpuWaitContextAlloc.Not_enough_system_resources}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for failure use-case
 * where LwSciSyncCoreRmWaitCtxBackEndAlloc() returns error when not enough system resources.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciSyncCoreModuleGetRmBackEnd() returns to LwSciError_Success
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() returns to LwSciError_ResourceError}
 *
 * @testinput{module points to the adddress '0X5555555555555555')
 * newContext set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct arugument.
 * LwSciCommonCalloc() receives correct arguments
 * LwSciSyncCoreModuleGetRmBackEnd() receives correct arugument.
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() receives correct argument
 * LwSciSyncCoreRmWaitCtxBackEndFree() receives correct argument.
 * LwSciCommonFree() receives correct argument
 * returned LwSciError_ResourceError}
 *
 * @testcase{18853995}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_ResourceError
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>> ));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>> == ( <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0X5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0X5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( 0X5555555555555555 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCpuWaitContextAlloc.failed_to_associate_module_with_newContext
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCpuWaitContextAlloc.failed_to_associate_module_with_newContext
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCpuWaitContextAlloc.failed_to_associate_module_with_newContext}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for failure use-case
 * where LwSciSyncCoreModuleDup() returns error when failed to associate module with newContext}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciSyncCoreModuleGetRmBackEnd() returns to LwSciError_Success
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() returns to LwSciError_Success
 * LwSciSyncCoreModuleDup() returns to LwSciError_IlwalidState}
 *
 * @testinput{module points to the adddress '0xFFFFFFFF'
 * newContext set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct aruguments.
 * LwSciCommonCalloc() receives correct arguments
 * LwSciSyncCoreModuleGetRmBackEnd() receives correct aruguments.
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() receives correct arguments
 * LwSciSyncCoreModuleDup() receives correct arguments
 * LwSciSyncCoreRmWaitCtxBackEndFree() receives correct arguments.
 * LwSciCommonFree() receives correct arguments
 * returned LwSciError_IlwalidState}
 *
 * @testcase{18853998}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_IlwalidState
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_IlwalidState
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>> ));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCpuWaitContextRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>> == ( <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleDup.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( 0xFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCpuWaitContextAlloc.normal_operation
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextAlloc
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCpuWaitContextAlloc.normal_operation
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCpuWaitContextAlloc.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextAlloc() for success use-case 
 * when successful allocation of a new LwSciSyncCpuWaitContext}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciSyncCoreModuleGetRmBackEnd() returns to LwSciError_Success
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() returns to LwSciError_Success
 * LwSciSyncCoreModuleDup() returns to LwSciError_Success}
 *
 * @testinput{module points to the adddress '0xFFFFFFFF'
 * newContext set to valid memory}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct aruguments.
 * LwSciCommonCalloc() receives correct arguments
 * LwSciSyncCoreModuleGetRmBackEnd() receives correct aruguments.
 * LwSciSyncCoreRmWaitCtxBackEndAlloc() receives correct arguments
 * LwSciSyncCoreModuleDup() receives correct arguments
 * Newcontext set to valid memory
 * returned LwSciError_Success}
 *
 * @testcase{18854001}
 *
 * @verify{18844437}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext:<<malloc 1>>
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleDup.return:LwSciError_Success
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc
  uut_prototype_stubs.LwSciSyncCoreModuleDup
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>> ));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd.waitContextBackEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.waitContextBackEnd>>[0] = ( 0x1515151515151515 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.dupModule.dupModule[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleDup.dupModule>>[0] = ( 0x5555555555555555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x2525252525252525 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCpuWaitContextRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndAlloc.rmBackEnd>> == ( <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleDup.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleDup.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0xFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.module>> = ( 0xFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext.newContext[0][0].header
{{ <<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext>>[0][0].header == ( (((uint64_t)(<<uut_prototype_stubs.LwSciCommonCalloc.return>>) & (0xFFFF00000000FFFFULL)) | 0x123456780000ULL) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext.newContext[0][0].module
{{ <<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext>>[0][0].module == ( 0x5555555555555555 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext.newContext[0][0].waitContextBackEnd
{{ <<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextAlloc.newContext>>[0][0].waitContextBackEnd == ( 0x1515151515151515 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCpuWaitContextFree

-- Test Case: TC_001.LwSciSyncCpuWaitContextFree.context_NULL
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCpuWaitContextFree.context_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCpuWaitContextFree.context_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextFree() for failure use-case when the input context is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{Context set to NULL}
 *
 * @testbehavior{no operation}
 *
 * @testcase{18854004}
 *
 * @verify{18844440}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context:<<null>>
TEST.EXPECTED:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context:<<null>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCpuWaitContextFree.Failure_due_to_module_is_null
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCpuWaitContextFree.Failure_due_to_module_is_null
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCpuWaitContextFree.Failure_due_to_module_is_null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextFree() for failure use-case
 * where LwSciSyncCoreModuleValidate() fails when module points to NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() returns to LwSciError_BadParameter}
 *
 * @testinput{Context[0].module set to NULL}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct arguments}
 *
 * @testcase{18854007}
 *
 * @verify{18844440}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context>>[0].module = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCpuWaitContextFree.LwSciSyncModule_Ilwalid
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextFree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCpuWaitContextFree.LwSciSyncModule_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCpuWaitContextFree.LwSciSyncModule_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextFree() for panic use-case
 * where LwSciSyncCpuWaitContextFree() panics}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() panics}
 *
 * @testinput{Context[0].module set to value representing invalid LwSciSyncModule (0xFFFFFFFFFFFFFFFF)}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct argument
 * LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18854010}
 *
 * @verify{18844440}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context:<<malloc 1>>
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context>>[0].module = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCpuWaitContextFree.normal_operation
TEST.UNIT:lwscisync_cpu_wait_context
TEST.SUBPROGRAM:LwSciSyncCpuWaitContextFree
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCpuWaitContextFree.normal_operation
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCpuWaitContextFree.normal_operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCpuWaitContextFree() for success use-case
 * when the successful freeing of input context.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciSyncCoreModuleValidate() returns to LwSciError_Success}
 *
 * @testinput{Context[0].module set to the address '0X2525252525252525'
 * Context[0].waitContextBackEnd set to the address '0X3030303030303030'}
 *
 * @testbehavior{LwSciSyncCoreModuleValidate() receives correct argument
 * LwSciSyncCoreRmWaitCtxBackEndFree() receives correct argument
 * LwSciSyncModuleClose() receives correct argument
 * LwSciCommonFree() receives correct argument}
 *
 * @testcase{18854013}
 *
 * @verify{18844440}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
  uut_prototype_stubs.LwSciSyncCoreModuleValidate
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree
  uut_prototype_stubs.LwSciSyncModuleClose
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_cpu_wait_context.c.LwSciSyncCpuWaitContextFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context>>) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncModuleClose.module
{{ <<uut_prototype_stubs.LwSciSyncModuleClose.module>> == ( 0X2525252525252525 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxBackEndFree.waitContextBackEnd>> == ( 0X3030303030303030 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleValidate.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleValidate.module>> == ( 0X2525252525252525 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context.context[0].module
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context>>[0].module = ( 0X2525252525252525 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context.context[0].waitContextBackEnd
<<lwscisync_cpu_wait_context.LwSciSyncCpuWaitContextFree.context>>[0].waitContextBackEnd = ( 0X3030303030303030 );
TEST.END_VALUE_USER_CODE:
TEST.END
