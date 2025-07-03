-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
--
-- Environment    : LWSCISYNC_HEADER_CORE
-- Unit(s) Under Test: lwscisync_core
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscisync_core

-- Subprogram: LwSciSyncCoreGetLibVersion

-- Test Case: TC_001.LwSciSyncCoreGetLibVersion.Normal_Operation
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCoreGetLibVersion
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreGetLibVersion.Normal_Operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreGetLibVersion.Normal_Operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreGetLibVersion() - Normal Operation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{None.}
 *
 * @testbehavior{Returns the bit-shifted library version.}
 *
 * @testcase{18853827}
 *
 * @verify{18844413}
 */
TEST.END_NOTES:
TEST.FLOW
  lwscisync_core.h.LwSciSyncCoreGetLibVersion
  lwscisync_core.h.LwSciSyncCoreGetLibVersion
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscisync_core.LwSciSyncCoreGetLibVersion.return
{{ <<lwscisync_core.LwSciSyncCoreGetLibVersion.return>> == ( ((uint64_t)LwSciSyncMajorVersion << 32U) | (uint64_t)LwSciSyncMinorVersion ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreValidateIpcEndpoint

-- Test Case: TC_001.LwSciSyncCoreValidateIpcEndpoint.Normal_Operation
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCoreValidateIpcEndpoint
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreValidateIpcEndpoint.Normal_Operation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreValidateIpcEndpoint.Normal_Operation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreValidateIpcEndpoint() - Normal Operation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreValidateIpcEndpoint() returns 'LwSciError_Success'(if handle is a valid LwSciIpcEndpoint).}
 *
 * @testinput{handle driven as 1.}
 *
 * @testbehavior{Returns LwSciError_Success bealwase of handle is a valid LwSciIpcEndpoint}
 *
 * @testcase{18853830}
 *
 * @verify{18844416}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCoreValidateIpcEndpoint.handle:1
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_Success
TEST.EXPECTED:lwscisync_core.LwSciSyncCoreValidateIpcEndpoint.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:1
TEST.FLOW
  lwscisync_core.h.LwSciSyncCoreValidateIpcEndpoint
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_core.h.LwSciSyncCoreValidateIpcEndpoint
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreValidateIpcEndpoint.handle_is_not_a_valid_LwSciIpcEndpoint
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCoreValidateIpcEndpoint
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreValidateIpcEndpoint.handle_is_not_a_valid_LwSciIpcEndpoint
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreValidateIpcEndpoint.handle_is_not_a_valid_LwSciIpcEndpoint}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreValidateIpcEndpoint() - handle is not a valid LwSciIpcEndpoint.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciSyncCoreValidateIpcEndpoint() returns 'LwSciError_BadParameter'(if handle is not a valid LwSciIpcEndpoint).}
 *
 * @testinput{handle driven as 85899345937.}
 *
 * @testbehavior{Returns LwSciError_BadParameter  bealwase of handle is not a valid LwSciIpcEndpoint.}
 *
 * @testcase{18853833}
 *
 * @verify{18844416}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCoreValidateIpcEndpoint.handle:85899345937
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info[0].nframes:2
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.info[0].frame_size:3
TEST.VALUE:uut_prototype_stubs.LwSciIpcGetEndpointInfo.return:LwSciError_BadParameter
TEST.EXPECTED:lwscisync_core.LwSciSyncCoreValidateIpcEndpoint.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciIpcGetEndpointInfo.handle:85899345937
TEST.FLOW
  lwscisync_core.h.LwSciSyncCoreValidateIpcEndpoint
  uut_prototype_stubs.LwSciIpcGetEndpointInfo
  lwscisync_core.h.LwSciSyncCoreValidateIpcEndpoint
TEST.END_FLOW
TEST.END
