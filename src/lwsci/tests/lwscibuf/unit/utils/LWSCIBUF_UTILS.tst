-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCIBUF_UTILS
-- Unit(s) Under Test: lwscibuf_utils
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

-- Subprogram: LwColorDataTypeToLwSciBufDataType

-- Test Case: TC_001.LwColorDataTypeToLwSciBufDataType.NormalOperation
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwColorDataTypeToLwSciBufDataType
TEST.NEW
TEST.NAME:TC_001.LwColorDataTypeToLwSciBufDataType.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwColorDataTypeToLwSciBufDataType.NormalOperation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwColorDataTypeToLwSciBufDataType()-Normal operation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.colorDataType driven as LwColorDataType_Signed.
 * 2.channelCount driven as 3.
 * 3.colorBPP driven as 12.}
 *
 * @testbehavior{1.returns LwSciDataType_Uint4.}
 *
 * @testcase{18858414}
 *
 * @verify{18842988}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.colorDataType:LwColorDataType_Signed
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.channelCount:3
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.colorBPP:12
TEST.EXPECTED:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.return:LwSciDataType_Int4
TEST.FLOW
  lwscibuf_utils.c.LwColorDataTypeToLwSciBufDataType
  lwscibuf_utils.c.LwColorDataTypeToLwSciBufDataType
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwColorDataTypeToLwSciBufDataType.powOf2BPCIsGreaterThan5.
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwColorDataTypeToLwSciBufDataType
TEST.NEW
TEST.NAME:TC_002.LwColorDataTypeToLwSciBufDataType.powOf2BPCIsGreaterThan5.
TEST.NOTES:
/**
 * @testname{TC_002.LwColorDataTypeToLwSciBufDataType.powOf2BPCIsGreaterThan5.}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwColorDataTypeToLwSciBufDataType()-powOf2BPC is greater than 5.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.colorDataType driven as LwColorDataType_Signed.
 * 2.channelCount driven as 1.
 * 3.colorBPP driven as 64.}
 *
 * @testbehavior{1.returns LwSciDataType_UpperBound.}
 *
 * @testcase{18858417}
 *
 * @verify{18842988}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.colorDataType:LwColorDataType_Signed
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.channelCount:1
TEST.VALUE:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.colorBPP:64
TEST.EXPECTED:lwscibuf_utils.LwColorDataTypeToLwSciBufDataType.return:LwSciDataType_UpperBound
TEST.FLOW
  lwscibuf_utils.c.LwColorDataTypeToLwSciBufDataType
  lwscibuf_utils.c.LwColorDataTypeToLwSciBufDataType
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufAliglwalue32

-- Test Case: TC_001.LwSciBufAliglwalue32.Successful_32BitAlignment
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue32
TEST.NEW
TEST.NAME:TC_001.LwSciBufAliglwalue32.Successful_32BitAlignment
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAliglwalue32.Successful_32BitAlignment}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue32() successfully aligns 32-bit value to given 32-bit alignment.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue32().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to pointer.}
 *
 * @testbehavior{- LwSciBufAliglwalue32() returns LwSciError_Success.
 * - alignedValue pointing to 32-bit alignment value.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.value:1
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignment:2
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignedValue:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue32.alignedValue[0]:2
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue32.return:LwSciError_Success
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue32
  lwscibuf_utils.c.LwSciBufAliglwalue32
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufAliglwalue32.ArithmeticOverflow
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue32
TEST.NEW
TEST.NAME:TC_002.LwSciBufAliglwalue32.ArithmeticOverflow
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAliglwalue32.ArithmeticOverflow}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue32() when arithmetic overflow oclwrs.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue32().}
 *
 * @casederiv{Analysis of Requirements.
 * - Boundary value analysis.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to pointer.}
 *
 * @testbehavior{- LwSciBufAliglwalue32() returns LwSciError_Overflow.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.value:<<MAX>>
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignment:<<MAX>>
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignedValue:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue32.return:LwSciError_Overflow
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue32
  lwscibuf_utils.c.LwSciBufAliglwalue32
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufAliglwalue32.NULL_alignedValue
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue32
TEST.NEW
TEST.NAME:TC_003.LwSciBufAliglwalue32.NULL_alignedValue
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAliglwalue32.NULL_alignedValue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue32() for NULL alignedValue.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue32().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to NULL.}
 *
 * @testbehavior{- LwSciBufAliglwalue32() panics.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.value:1
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignment:2
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue32.alignedValue:<<null>>
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue32
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufAliglwalue64

-- Test Case: TC_001.LwSciBufAliglwalue64.Successful_64BitAlignment
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue64
TEST.NEW
TEST.NAME:TC_001.LwSciBufAliglwalue64.Successful_64BitAlignment
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAliglwalue64.Successful_64BitAlignment}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue64() successfully aligns 64-bit value to given 64-bit alignment.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue64().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to pointer.}
 *
 * @testbehavior{- LwSciBufAliglwalue64() returns LwSciError_Success.
 * - alignedValue pointing to 64-bit alignment value.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.value:1
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignment:2
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignedValue:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue64.alignedValue[0]:2
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue64.return:LwSciError_Success
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue64
  lwscibuf_utils.c.LwSciBufAliglwalue64
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufAliglwalue64.ArithmeticOverflow
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue64
TEST.NEW
TEST.NAME:TC_002.LwSciBufAliglwalue64.ArithmeticOverflow
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAliglwalue64.Successful_64BitAlignment}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue64() when arithmetic overflow oclwrs.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue64().}
 *
 * @casederiv{Analysis of Requirements.
 * - Boundary value analysis.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to pointer.}
 *
 * @testbehavior{- LwSciBufAliglwalue64() returns LwSciError_Overflow.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.value:<<MAX>>
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignment:<<MAX>>
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignedValue:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufAliglwalue64.return:LwSciError_Overflow
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue64
  lwscibuf_utils.c.LwSciBufAliglwalue64
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufAliglwalue64.NULL_alignedValue
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufAliglwalue64
TEST.NEW
TEST.NAME:TC_003.LwSciBufAliglwalue64.NULL_alignedValue
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAliglwalue64.NULL_alignedValue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAliglwalue64() for NULL alignedValue.}
 *
 * @testpurpose{Unit testing of LwSciBufAliglwalue64().}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{- value set to non-zero value.
 * - alignment set to non-zero value.
 * - alignedValue set to NULL.}
 *
 * @testbehavior{- LwSciBufAliglwalue64() panics.
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
 * @verify{}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.value:1
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignment:2
TEST.VALUE:lwscibuf_utils.LwSciBufAliglwalue64.alignedValue:<<null>>
TEST.FLOW
  lwscibuf_utils.c.LwSciBufAliglwalue64
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufIsMaxValue

-- Test Case: TC_001.LwSciBufIsMaxValue.cmp1cmp2AreSameSoReturnsIsbiggerIsTrue
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_001.LwSciBufIsMaxValue.cmp1cmp2AreSameSoReturnsIsbiggerIsTrue
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIsMaxValue.cmp1cmp2AreSameSoReturnsIsbiggerIsTrue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufIsMaxValue()- cmp1 and cmp2 both are same so,returns *isBigger is TRUE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 4.
 * 2.src2 driven as 4.
 * 3.len driven as 1.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.isBigger[0] pointing to valid one.
 * 2.returns LwSciError_Success.}
 *
 * @testcase{18858420}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.isBigger[0]:true
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_Success
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIsMaxValue.cmp1IsLessThancmp2SoReturnsIsbiggerIsFlase
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_002.LwSciBufIsMaxValue.cmp1IsLessThancmp2SoReturnsIsbiggerIsFlase
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIsMaxValue.cmp1IsLessThancmp2SoReturnsIsbiggerIsFlase}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufIsMaxValue()- cmp1 is less than cmp2 so,returns *isBigger is FALSE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 1.
 * 2.src2 driven as 2.
 * 3.len driven as 1.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.isBigger[0] pointing to valid one.
 * 2.returns LwSciError_Success.}
 *
 * @testcase{18858423}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.isBigger[0]:false
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_Success
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 1;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 2;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIsMaxValue.cmp1IsGreaterThancmp2SoReturnsIsbiggerIsTrue
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_003.LwSciBufIsMaxValue.cmp1IsGreaterThancmp2SoReturnsIsbiggerIsTrue
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIsMaxValue.cmp1IsGreaterThancmp2SoReturnsIsbiggerIsTrue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufIsMaxValue()- cmp1 is greater than cmp2 so,returns *isBigger is TRUE.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 3.
 * 2.src2 driven as 2.
 * 3.len driven as 1.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.isBigger[0] pointing to valid one.
 * 2.returns LwSciError_Success.}
 *
 * @testcase{18858426}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:2
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.isBigger[0]:true
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_Success
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 3;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 2;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIsMaxValue.Src1Src2IsbiggerIsNULL
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_004.LwSciBufIsMaxValue.Src1Src2IsbiggerIsNULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIsMaxValue.Src1Src2IsbiggerIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufIsMaxValue()- src1,src2,len,isBigger is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 set to NULL.
 * 2.src2 set to NULL.
 * 4.isBigger set to NULL.}
 *
 * @testbehavior{1.returns LwSciError_BadParameter.}
 *
 * @testcase{18858429}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.src1:<<null>>
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.src2:<<null>>
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<null>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciBufIsMaxValue.src1IsNULL
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_005.LwSciBufIsMaxValue.src1IsNULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIsMaxValue.src1IsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufIsMaxValue() - src1 is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 set to NULL.
 * 2.src2 driven as 4.
 * 3.len driven as 1.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.returns LwSciError_BadParameter.}
 *
 * @testcase{18858432}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.src1:<<null>>
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIsMaxValue.src2IsNULL
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_006.LwSciBufIsMaxValue.src2IsNULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIsMaxValue.src2IsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API
 * src2 set to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 4.
 * 2.src2 set to NULL.
 * 3.len driven as 1.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.returns LwSciError_BadParameter.}
 *
 * @testcase{18858435}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.src2:<<null>>
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIsMaxValue.IsbiggerIsNULL
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_007.LwSciBufIsMaxValue.IsbiggerIsNULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIsMaxValue.IsbiggerIsNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API
 * IsBigger set to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 4.
 * 2.src2 driven as 4.
 * 3.len driven as 1.
 * 4.isBigger set to NULL.}
 *
 * @testbehavior{1.returns LwSciError_BadParameter.}
 *
 * @testcase{18858438}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:1
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<null>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIsMaxValue.len_is_0
TEST.UNIT:lwscibuf_utils
TEST.SUBPROGRAM:LwSciBufIsMaxValue
TEST.NEW
TEST.NAME:TC_008.LwSciBufIsMaxValue.len_is_0
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIsMaxValue.len_is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API
 * len set to 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{1.src1 driven as 4.
 * 2.src2 driven as 4.
 * 3.len set to 0.
 * 4.isBigger set to valid memory.}
 *
 * @testbehavior{1.returns LwSciError_BadParameter.}
 *
 * @testcase{18858441}
 *
 * @verify{18842991}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.len:0
TEST.VALUE:lwscibuf_utils.LwSciBufIsMaxValue.isBigger:<<malloc 1>>
TEST.EXPECTED:lwscibuf_utils.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_utils.c.LwSciBufIsMaxValue
  lwscibuf_utils.c.LwSciBufIsMaxValue
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src1
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src1>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_utils.LwSciBufIsMaxValue.src2
static uint8_t val = 4;

<<lwscibuf_utils.LwSciBufIsMaxValue.src2>> = (&val);
TEST.END_VALUE_USER_CODE:
TEST.END

