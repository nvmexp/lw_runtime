-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
-- 
-- Environment    : LWSCISYNC_CORE
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

-- Subprogram: LwSciSyncCheckVersionCompatibility

-- Test Case: TC_001.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MinorVersion
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MinorVersion
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MinorVersion}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCheckVersionCompatibility() - Success use case when Minor version is not compatible and sets the output variable to false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'majorVer' is driven as 'LwSciSyncMajorVersion'.
 *  - 'minorVer' is driven as 'LwSciSyncMinorVersion+1'.
 *  - 'iscompatible' set to valid memory.}
 *
 * @testbehavior{- 'isCompatible' returns 'false' because Minor version is not compatible.
 *  - Returns LwSciError_Success.}
 *
 * @testcase{18853836}
 *
 * @verify{18844407}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.majorVer:MACRO=LwSciSyncMajorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:true
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:false
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Success
TEST.VALUE_USER_CODE:lwscisync_core.LwSciSyncCheckVersionCompatibility.minorVer
<<lwscisync_core.LwSciSyncCheckVersionCompatibility.minorVer>> = ( LwSciSyncMinorVersion+1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCheckVersionCompatibility.Success_Compatible_Version
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCheckVersionCompatibility.Success_Compatible_Version
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCheckVersionCompatibility.Success_Compatible_Version}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCheckVersionCompatibility() - Success use case when Major and minor version is compatible and sets the output variable to true.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'majorVer' is driven as 'LwSciSyncMajorVersion'.
 *  - 'minorVer' is driven as 'LwSciSyncMinorVersion'.
 *  - 'iscompatible' set to valid memory.}
 *
 * @testbehavior{- 'isCompatible' returns 'true' when Major and minor version is compatible.
 *  - Returns LwSciError_Success.}
 *
 * @testcase{18853839}
 *
 * @verify{18844407}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.majorVer:MACRO=LwSciSyncMajorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.minorVer:MACRO=LwSciSyncMinorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:false
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:true
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Success
TEST.END

-- Test Case: TC_003.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MajorVersion
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MajorVersion
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCheckVersionCompatibility.Success_Incompatible_MajorVersion}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCheckVersionCompatibility() - Major version is not compatible and sets the output variable to false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'majorVer' is driven as 'LwSciSyncMajorVersion+1'.
 *  - 'minorVer' is driven as 'LwSciSyncMinorVersion'.
 *  - 'iscompatible' set to valid memory.}
 *
 * @testbehavior{- 'isCompatible' returns 'false' because Major version is not compatible.
 *  - Returns LwSciError_Success.}
 *
 * @testcase{18853842}
 *
 * @verify{18844407}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.minorVer:MACRO=LwSciSyncMinorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible:<<malloc 1>>
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:true
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible[0]:false
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Success
TEST.VALUE_USER_CODE:lwscisync_core.LwSciSyncCheckVersionCompatibility.majorVer
<<lwscisync_core.LwSciSyncCheckVersionCompatibility.majorVer>> = ( LwSciSyncMajorVersion+1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCheckVersionCompatibility.Failure_due_to_NULL_iscompatible
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCheckVersionCompatibility
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCheckVersionCompatibility.Failure_due_to_NULL_iscompatible
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCheckVersionCompatibility.Failure_due_to_NULL_iscompatible}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCheckVersionCompatibility() - Failure due to iscompatible is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'majorVer' is driven as 'LwSciSyncMajorVersion'.
 *  - 'minorVer' is driven as 'LwSciSyncMinorVersion'.
 *  - 'iscompatible' set to NULL.}
 *
 * @testbehavior{- Returns the invalid arguments as LwSciError_BadParameter.}
 *
 * @testcase{18853845}
 *
 * @verify{18844407}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.majorVer:MACRO=LwSciSyncMajorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.minorVer:MACRO=LwSciSyncMinorVersion
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.isCompatible:<<null>>
TEST.VALUE:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_core.LwSciSyncCheckVersionCompatibility.return:LwSciError_BadParameter
TEST.END

-- Subprogram: LwSciSyncCorePermLEq

-- Test Case: TC_001.LwSciSyncCorePermLEq.Success_case1_permA_Is_smaller_than_permB
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLEq
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePermLEq.Success_case1_permA_Is_smaller_than_permB
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePermLEq.Success_case1_permA_Is_smaller_than_permB}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCorePermLEq() - Success when permA is smaller than permB}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'permA' is driven as 'LwSciSyncAccessPerm_SignalOnly'.
 *  - 'permB' is driven as 'LwSciSyncAccessPerm_WaitSignal'.}
 *
 * @testbehavior{- Returns true.}
 *
 * @testcase{22060328}
 *
 * @verify{21423605}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.return:false
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLEq.return:true
TEST.END

-- Test Case: TC_002.LwSciSyncCorePermLEq.Success_case2_permA_Is_equal_to_permB
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLEq
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePermLEq.Success_case2_permA_Is_equal_to_permB
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePermLEq.Success_case2_permA_Is_equal_to_permB}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCorePermLEq() - Success when permA is equal to permB}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'permA' is driven as 'LwSciSyncAccessPerm_WaitSignal'.
 *  - 'permB' is driven as 'LwSciSyncAccessPerm_WaitSignal'.}
 *
 * @testbehavior{- Returns true.}
 *
 * @testcase{22060331}
 *
 * @verify{21423605}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.return:false
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLEq.return:true
TEST.END

-- Test Case: TC_003.LwSciSyncCorePermLEq.Failure_due_to_permA_greater_than_permB
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLEq
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCorePermLEq.Failure_due_to_permA_greater_than_permB
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCorePermLEq.Failure_due_to_permA_greater_than_permB}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCorePermLEq() - Failure due to permA is greater than permB}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'permA' is driven as 'LwSciSyncAccessPerm_WaitSignal'.
 *  - 'permB' is driven as 'LwSciSyncAccessPerm_SignalOnly'.}
 *
 * @testbehavior{- Returns false.}
 *
 * @testcase{22060334}
 *
 * @verify{21423605}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLEq.return:false
TEST.END

-- Subprogram: LwSciSyncCorePermLessThan

-- Test Case: TC_001.LwSciSyncCorePermLessThan.false
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLessThan
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePermLessThan.false
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePermLessThan.false}
 *
 * @verifyFunction{Normal scenario, result is false.}
 *
 * @casederiv{Analysis of Requirements.
 *
 * Generation and Analysis of Equivalence Classes.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- permA set to LwSciSyncAccessPerm_WaitSignal.
 *  - permB set to LwSciSyncAccessPerm_SignalOnly.}
 *
 * @testbehavior{Function should return false.}
 *
 * @testcase{18852804}
 *
 * @verify{18844290}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLessThan.return:false
TEST.FLOW
  lwscisync_core.c.LwSciSyncCorePermLessThan
  lwscisync_core.c.LwSciSyncCorePermLessThan
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCorePermLessThan.true
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLessThan
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePermLessThan.true
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePermLessThan.true}
 *
 * @verifyFunction{Normal scenario, result is true.}
 *
 * @casederiv{Analysis of Requirements.
 *
 * Generation and Analysis of Equivalence Classes.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- permA set to LwSciSyncAccessPerm_SignalOnly.
 *  - permB set to LwSciSyncAccessPerm_WaitSignal.}
 *
 * @testbehavior{Function should return true.}
 *
 * @testcase{18852807}
 *
 * @verify{18844290}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLessThan.return:true
TEST.FLOW
  lwscisync_core.c.LwSciSyncCorePermLessThan
  lwscisync_core.c.LwSciSyncCorePermLessThan
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCorePermLessThan.false_due_to_permA_Is_equal_to_permB
TEST.UNIT:lwscisync_core
TEST.SUBPROGRAM:LwSciSyncCorePermLessThan
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCorePermLessThan.false_due_to_permA_Is_equal_to_permB
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCorePermLessThan.Failure_due_to_permBA_Is_equal_to_permB}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCorePermLessThan() - Failure due to permA is equal to permB}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- None.}
 *
 * @testinput{- 'permA' is driven as 'LwSciSyncAccessPerm_WaitSignal'.
 *  - 'permB' is driven as 'LwSciSyncAccessPerm_WaitSignal'.}
 *
 * @testbehavior{- Returns false.}
 *
 * @testcase{22060338}
 *
 * @verify{18844290}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permA:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.permB:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_core.LwSciSyncCorePermLessThan.return:true
TEST.EXPECTED:lwscisync_core.LwSciSyncCorePermLessThan.return:false
TEST.END
