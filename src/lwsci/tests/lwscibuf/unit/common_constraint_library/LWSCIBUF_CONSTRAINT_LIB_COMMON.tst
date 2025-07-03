-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCIBUF_CONSTRAINT_LIB_COMMON
-- Unit(s) Under Test: lwscibuf_constraint_lib_common
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

-- Unit: lwscibuf_constraint_lib_common

-- Subprogram: LwSciBufGetConstraints

-- Test Case: TC_001.LwSciBufGetConstraints.Panic_due_to_bufType_set_to_LwSciBufType_MaxValid
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufGetConstraints.Panic_due_to_bufType_set_to_LwSciBufType_MaxValid
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufGetConstraints.Panic_due_to_bufType_set_to_LwSciBufType_MaxValid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() BufType set to 'LwSciBufType_MaxValid'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Buffer type set to 'LwSciBufType_MaxValid'.
 * Chip id set to 25.
 * Array containing engine details is set to NULL.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855327}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:5
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_MaxValid
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT1
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufGetConstraints.Success_due_to_bufType_set_to_LwSciBufType_RawBuffer
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufGetConstraints.Success_due_to_bufType_set_to_LwSciBufType_RawBuffer
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufGetConstraints.Success_due_to_bufType_set_to_LwSciBufType_RawBuffer}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() BufType set to 'LwSciBufType_RawBuffer'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Buffer type set to 'LwSciBufType_RawBuffer'.
 * Chip id set to 25.
 * Array containing engine details is set to NULL.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array set to NULL.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855330}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:5
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_RawBuffer
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].startAddrAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].pitchAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].heightAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].sizeAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobSize:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockX:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockY:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockZ:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_one
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_one
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_one}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() engineCount set to 1}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 25.
 * Array containing engine details is initialized.
 * Number of engines set to 1.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array is initialized.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855333}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray[0].rmModuleID:107
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].startAddrAlign:128
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].pitchAlign:128
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].heightAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].sizeAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobSize:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockX:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockY:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockZ:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
static int i =0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufImageConstraints) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i =0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_Zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufGetConstraints.Success_due_to_engineCount_set_to_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() engineCount set to 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 25.
 * Array containing engine details is initialized.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array is initialized.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855336}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].startAddrAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].pitchAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].heightAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].sizeAlign:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobSize:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockX:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockY:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints[0].log2GobsperBlockZ:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufGetConstraints.Failure_due_to_chipId_Not_found
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_007.LwSciBufGetConstraints.Failure_due_to_chipId_Not_found
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufGetConstraints.Failure_due_to_chipId_Not_found}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() chipId not found}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 1.
 * Array containing engine details is initialized.
 * Number of engines set to 1.
 * Data set to invalid LwSciBufAttrValImageLayoutType
 * Reconciled output Hardware constraints array is initialized.}
 *
 * @testbehavior{Returns LwSciError_NotSupported}
 *
 * @testcase{18855345}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_NotSupported
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufGetConstraints.Failure_due_to_NotSupported_LayOutType
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_008.LwSciBufGetConstraints.Failure_due_to_NotSupported_LayOutType
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufGetConstraints.Failure_due_to_NotSupported_LayOutType}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() Not Supported Pitch and Block Linear Type.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.
 * LwSciBufGetT194ImageConstraints() returns LwSciError_Success}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 1.
 * Array containing engine details is initialized.
 * Number of engines set to 1.
 * Data set to invalid LwSciBufAttrValImageLayoutType
 * Reconciled output Hardware constraints array is initialized.}
 *
 * @testbehavior{Returns LwSciError_NotSupported if Image Layout is none of BlockLinear and PitchLinear Type}
 *
 * @testcase{18855348}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:6
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray[0].rmModuleID:MACRO=LwSciBufHwEngName_Display
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufGetT194ImageConstraints.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_NotSupported
TEST.EXPECTED:uut_prototype_stubs.LwSciBufGetT194ImageConstraints.engine.rmModuleID:4
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciBufGetT194ImageConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
static int i =0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufImageConstraints) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i =0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufGetConstraints.Success_due_to_rmModuleId_set_to_four
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_009.LwSciBufGetConstraints.Success_due_to_rmModuleId_set_to_four
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufGetConstraints.Success_due_to_rmModuleId_set_to_four}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() engineArray[0].rmModuleId set to 4}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 25.
 * Array containing engine details array is initialized with rmModuleId 4.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array is initialized.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855351}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray[0].rmModuleID:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufImageConstraints) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufGetConstraints.Panic_due_to_imageConstraints_is_Null
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_010.LwSciBufGetConstraints.Panic_due_to_imageConstraints_is_Null
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufGetConstraints.Panic_due_to_imageConstraints_is_Null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() BufType set to 'LwSciBufType_Image'
 * and image constraints set to NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 25.
 * Array containing engine details is set to NULL.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set.
 * Reconciled output Hardware constraints array set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855354}
 *
 * @verify{18842763}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:5
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:VECTORCAST_INT2
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciBufGetConstraints.NULL_data
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetConstraints
TEST.NEW
TEST.NAME:TC_011.LwSciBufGetConstraints.NULL_data
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufGetConstraints.NULL_data}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetConstraints() for null data when BufType set to 'LwSciBufType_Image'}
 *
 * @testpurpose{Unit testing of LwSciBufGetConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Buffer type set to 'LwSciBufType_Image'.
 * Chip id set to 25.
 * Array containing engine details is set to NULL.
 * Number of engines set to 0.
 * Data whose constraints we need to get is set to NULL.
 * Reconciled output Hardware constraints array set to allocated memory.}
 *
 * @testbehavior{- LwSciBufGetConstraints() panics
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
 * @testcase{22060557}
 *
 * @verify{18842763}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:5
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.bufType:LwSciBufType_Image
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineArray:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.engineCount:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.constraints:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.data:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId
<<lwscibuf_constraint_lib_common.LwSciBufGetConstraints.chipId>> = ( LWRM_T194_ID );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufGetDefaultImageConstraints

-- Test Case: TC_001.LwSciBufGetDefaultImageConstraints.notNull_constraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetDefaultImageConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufGetDefaultImageConstraints.notNull_constraints
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufGetDefaultImageConstraints.notNull_constraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetDefaultImageConstraints() for constraints not null.}
 *
 * @testpurpose{Unit testing of LwSciBufGetDefaultImageConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{constraints set to valid memory.}
 *
 * @testbehavior{- Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
 * - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22060560}
 *
 * @verify{19808982}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetDefaultImageConstraints.constraints:<<malloc 1>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetDefaultImageConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufGetDefaultImageConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufGetDefaultImageConstraints.Null_constraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufGetDefaultImageConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufGetDefaultImageConstraints.Null_constraints
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufGetDefaultImageConstraints.Null_constraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufGetDefaultImageConstraints() for null constraints.}
 *
 * @testpurpose{Unit testing of LwSciBufGetDefaultImageConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{constraints set to valid memory.}
 *
 * @testbehavior{- LwSciBufGetDefaultImageConstraints() Panics
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
 * @testcase{22060563}
 *
 * @verify{19808982}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufGetDefaultImageConstraints.constraints:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufGetDefaultImageConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufHwEngCreateIdWithInstance

-- Test Case: TC_001.LwSciBufHwEngCreateIdWithInstance.Failure_due_to_engName_is_LwSciBufHwEngName_Num
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_001.LwSciBufHwEngCreateIdWithInstance.Failure_due_to_engName_is_LwSciBufHwEngName_Num
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufHwEngCreateIdWithInstance.Failure_due_to_engName_is_LwSciBufHwEngName_Num}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() when engName is LwSciBufHwEngName_Num}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Num'(InValid).
 * instance of the engine defined by engName enum set to 1.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855357}
 *
 * @verify{18842751}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Num
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Min_value
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_002.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Min_value
TEST.BASIS_PATH:3 of 3 (partial)
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Min_value}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() when instance set to 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Isp'.
 * instance of the engine defined by engName enum set to 0.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855360}
 *
 * @verify{18842751}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Isp
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId.engId[0]
{{ <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId>>[0] == (int64_t) ( ((uint64_t) ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName>> ) & LW_SCI_BUF_ENG_NAME_BIT_MASK) << LW_SCI_BUF_ENG_NAME_BIT_START | ( ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance>> )  & LW_SCI_BUF_ENG_INSTANCE_BIT_MASK) << LW_SCI_BUF_ENG_INSTANCE_BIT_START ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Mid_value
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_003.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Mid_value
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Mid_value}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() instance set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Csi'.
 * instance of the engine defined by engName enum set to 1.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855363}
 *
 * @verify{18842751}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Csi
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId.engId[0]
{{ <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId>>[0] == (int64_t) ( ((uint64_t) ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName>> ) & LW_SCI_BUF_ENG_NAME_BIT_MASK) << LW_SCI_BUF_ENG_NAME_BIT_START | ( ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance>> )  & LW_SCI_BUF_ENG_INSTANCE_BIT_MASK) << LW_SCI_BUF_ENG_INSTANCE_BIT_START ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Max_value
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_004.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Max_value
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufHwEngCreateIdWithInstance.Success_due_to_instance_set_to_Max_value}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() instance set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_PVA'.
 * instance of the engine defined by engName enum set to UINT32_MAX.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855366}
 *
 * @verify{18842751}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_PVA
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance
<<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance>> = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId.engId[0]
{{ <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId>>[0] == (int64_t) ( ((uint64_t) ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName>> ) & LW_SCI_BUF_ENG_NAME_BIT_MASK) << LW_SCI_BUF_ENG_NAME_BIT_START | ( ( <<lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance>> )  & LW_SCI_BUF_ENG_INSTANCE_BIT_MASK) << LW_SCI_BUF_ENG_INSTANCE_BIT_START ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufHwEngCreateIdWithInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_005.LwSciBufHwEngCreateIdWithInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufHwEngCreateIdWithInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() success case}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Display'.
 * instance of the engine defined by engName enum set to 0.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855369}
 *
 * @verify{18842751}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Display
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId[0]:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciBufHwEngCreateIdWithInstance.Null_engId
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_008.LwSciBufHwEngCreateIdWithInstance.Null_engId
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufHwEngCreateIdWithInstance.Null_engId}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() when engId is NULL}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Vi'.
 * instance of the engine defined by engName enum set to 1.
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithInstance() returns LwSciError_BadParameter
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
 * @testcase{22060566}
 *
 * @verify{18842751}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Vi
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciBufHwEngCreateIdWithInstance.Ilwalid_engName
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithInstance
TEST.NEW
TEST.NAME:TC_009.LwSciBufHwEngCreateIdWithInstance.Ilwalid_engName
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufHwEngCreateIdWithInstance.Ilwalid_engName}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithInstance() for engName LwSciBufHwEngName_Ilwalid }
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{enum defining HW engine name is initialised to 'LwSciBufHwEngName_Ilwalid'(InValid).
 * instance of the engine defined by engName enum set to 1.
 * engine ID set to allocated memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithInstance() returns LwSciError_BadParameter
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
 * @testcase{22060569}
 *
 * @verify{18842751}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engName:LwSciBufHwEngName_Ilwalid
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.instance:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithInstance
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufHwEngCreateIdWithoutInstance

-- Test Case: TC_001.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engId_set_to_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_001.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engId_set_to_NULL
TEST.BASIS_PATH:1 of 3
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engId_set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() engId set to NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_Vi'.
 * engine ID set to NULL.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855378}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_Vi
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufHwEngCreateIdWithoutInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_002.LwSciBufHwEngCreateIdWithoutInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufHwEngCreateIdWithoutInstance.Success_due_to_engineName_set_to_LwSciBufHwEngName_Display}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() Success case}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_Display'.
 * engine ID set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855381}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_Display
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId[0]:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Num
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_005.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Num
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Num}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engineName is InValid.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_Num'(InValid).
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_BadParameter
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
 * @testcase{22060571}
 *
 * @verify{18842748}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_Num
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Ilwalid
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_006.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufHwEngCreateIdWithoutInstance.Failure_due_to_engName_set_to_LwSciBufHwEngName_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engineName is InValid.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_Ilwalid'(InValid).
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_BadParameter
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
 * @testcase{22060576}
 *
 * @verify{18842748}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_Ilwalid
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_Vic
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_007.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_Vic
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_Vic}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engName is 'LwSciBufHwEngName_Vic'.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_Vic'.
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_Success.
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
 * @testcase{22060577}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_Vic
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId[0]:106
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_MSENC
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_008.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_MSENC
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufHwEngCreateIdWithoutInstance.Success_engineName_set_to_LwSciBufHwEngName_LWDEC}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engName is 'LwSciBufHwEngName_MSENC'.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_MSENC'.
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_Success.
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
 * @testcase{22060582}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_MSENC
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId[0]:109
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWDEC
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_009.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWDEC
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWDEC}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engName is 'LwSciBufHwEngName_LWDEC'.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_LWDEC'.
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_Success.
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
 * @testcase{22060584}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_LWDEC
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId[0]:117
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Test Case: TC_010.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWJPG
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngCreateIdWithoutInstance
TEST.NEW
TEST.NAME:TC_010.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWJPG
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufHwEngCreateIdWithoutInstance.Successful_engineName_set_to_LwSciBufHwEngName_LWJPG}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngCreateIdWithoutInstance() when engName is 'LwSciBufHwEngName_LWJPG'.}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngCreateIdWithoutInstance().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engName enum defining HW engine name is initialised to 'LwSciBufHwEngName_LWJPG'.
 * engine ID set to valid memory.}
 *
 * @testbehavior{- LwSciBufHwEngCreateIdWithoutInstance() returns LwSciError_Success.
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
 * @testcase{22060588}
 *
 * @verify{18842748}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engName:LwSciBufHwEngName_LWJPG
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.engId[0]:118
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngCreateIdWithoutInstance.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngCreateIdWithoutInstance
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufHwEngGetInstanceFromId

-- Test Case: TC_001.LwSciBufHwEngGetInstanceFromId.Failure_due_to_instance_is_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetInstanceFromId
TEST.NEW
TEST.NAME:TC_001.LwSciBufHwEngGetInstanceFromId.Failure_due_to_instance_is_NULL
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufHwEngGetInstanceFromId.Failure_due_to_instance_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetInstanceFromId() when engine instance is NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to 1.
 * engine instance set to NULL.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855390}
 *
 * @verify{18842757}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.engId:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engId_set_to_less_than_Zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetInstanceFromId
TEST.NEW
TEST.NAME:TC_002.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engId_set_to_less_than_Zero
TEST.BASIS_PATH:2 of 4
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engId_set_to_less_than_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetInstanceFromId() when engine Id set to less than 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary values}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to -1.
 * engine instance set to allocated memory.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855393}
 *
 * @verify{18842757}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.engId:-1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engid_is_Zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetInstanceFromId
TEST.NEW
TEST.NAME:TC_003.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engid_is_Zero
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufHwEngGetInstanceFromId.Failure_due_to_engid_is_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetInstanceFromId() Retrieves hardware engine instance from LwSciBuf hardware engine ID successfully.}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{engine ID set to 0.
 * engine instance set to allocated memory.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855399}
 *
 * @verify{18842757}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.engId:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufHwEngGetInstanceFromId.Success_due_to_engId_is_valid_nominalvalue
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetInstanceFromId
TEST.NEW
TEST.NAME:TC_004.LwSciBufHwEngGetInstanceFromId.Success_due_to_engId_is_valid_nominalvalue
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufHwEngGetInstanceFromId.Success_due_to_engId_is_valid_nominalvalue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetInstanceFromId() when engId set to valid nominal value}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{engine ID set to a random valid value 122
 * engine instance set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{22060795}
 *
 * @verify{18842757}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.engId:122
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance[0]:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufHwEngGetInstanceFromId.Ilwalid_engId
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetInstanceFromId
TEST.NEW
TEST.NAME:TC_005.LwSciBufHwEngGetInstanceFromId.Ilwalid_engId
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufHwEngGetInstanceFromId.Ilwalid_engId}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetInstanceFromId() when engId set to inValid nominal value}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngGetInstanceFromId().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to invalid random value -255.
 * engine instance set to allocated memory.}
 *
 * @testbehavior{- LwSciBufHwEngGetInstanceFromId() returns LwSciError_BadParameter.
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
 * @testcase{22060591}
 *
 * @verify{18842757}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.engId:-255
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.instance:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetInstanceFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetInstanceFromId
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufHwEngGetNameFromId

-- Test Case: TC_001.LwSciBufHwEngGetNameFromId.Failure_due_to_engName_set_to_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetNameFromId
TEST.NEW
TEST.NAME:TC_001.LwSciBufHwEngGetNameFromId.Failure_due_to_engName_set_to_NULL
TEST.BASIS_PATH:1 of 4
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufHwEngGetNameFromId.Failure_due_to_engName_set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetNameFromId() when engName set to NULL.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to 1.
 * engine name set to NULL.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855402}
 *
 * @verify{18842754}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_less_than_Zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetNameFromId
TEST.NEW
TEST.NAME:TC_002.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_less_than_Zero
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_less_than_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetNameFromId() when engine Id set to less than 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary values}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to -1.
 * engine name set to allocated memory}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855405}
 *
 * @verify{18842754}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId:-1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_set_to_Zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetNameFromId
TEST.NEW
TEST.NAME:TC_003.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufHwEngGetNameFromId.Failure_due_to_engId_set_to_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetNameFromId() when engine Id set to 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{engine ID set to 0.
 * engine name set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_BadParameter}
 *
 * @testcase{18855408}
 *
 * @verify{18842754}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufHwEngGetNameFromId.Success_due_to_engId_is_valid_nominalvalue
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetNameFromId
TEST.NEW
TEST.NAME:TC_004.LwSciBufHwEngGetNameFromId.Success_due_to_engId_is_valid_nominalvalue
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufHwEngGetNameFromId.Success_due_to_engId_is_valid_nominalvalue}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetNameFromId() when engId set to valid nominal value}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonMemcpyS() copies memory of specified size in bytes from source to destination.}
 *
 * @testinput{engine ID set to a random valid value 250
 * engine name set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success}
 *
 * @testcase{18855411}
 *
 * @verify{18842754}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId:11
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName[0]:LwSciBufHwEngName_Isp
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciBufHwEngName) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufHwEngGetNameFromId.Max_engId
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufHwEngGetNameFromId
TEST.NEW
TEST.NAME:TC_005.LwSciBufHwEngGetNameFromId.Max_engId
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufHwEngGetNameFromId.Max_engId}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufHwEngGetNameFromId() when engId set to max value}
 *
 * @testpurpose{Unit testing of LwSciBufHwEngGetNameFromId().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{engine ID set to value INT64_MAX.
 * engine name set to allocated memory}
 *
 * @testbehavior{- LwSciBufHwEngGetNameFromId() returns LwSciError_BadParameter.
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
 * @testcase{22060594}
 *
 * @verify{18842754}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engName:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
  lwscibuf_constraint_lib_common.c.LwSciBufHwEngGetNameFromId
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId
<<lwscibuf_constraint_lib_common.LwSciBufHwEngGetNameFromId.engId>> = ( INT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufReconcileFloatPyramidConstraints

-- Test Case: TC_001.LwSciBufReconcileFloatPyramidConstraints.Panics_if_dest_is_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileFloatPyramidConstraints.Panics_if_dest_is_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileFloatPyramidConstraints.Panics_if_dest_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it panics if dest input parameter is NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonPanic() calls exit() with EXIT_FAILURE error code to simulate panic.}
 *
 * @testinput{dest set to null.
 * src set to allocated memory and the value at memory is set to 0.3.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() Panics}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061581}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var4:0.3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var4>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufReconcileFloatPyramidConstraints.Panics_if_src_is_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileFloatPyramidConstraints.Panics_if_src_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileFloatPyramidConstraints.Panics_if_src_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it panics if src input parameter is NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonPanic() calls exit() with EXIT_FAILURE error code to simulate panic.}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.2.}
 * src set to null.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() Panics}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061583}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var3>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_greater_than_destScaleFactor
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_greater_than_destScaleFactor
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_greater_than_destScaleFactor}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it succeeds when srcScaleFactor value is set greater than dstScaleFactor.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.3
 * src set to allocated memory and the value at memory is set to 0.4.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() returns LwSciError_Success.
 * - Memory value pointed to by dest is not updated and holds the value 0.3}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061586}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.3
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var4:0.4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var3>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var4>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_less_than_destScaleFactor
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_less_than_destScaleFactor
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileFloatPyramidConstraints.Successful_srcScaleFactor_set_less_than_destScaleFactor}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it succeeds when srcScaleFactor value is set less than dstScaleFactor.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.4
 * src set to allocated memory and the value at memory is set to 0.3.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() returns LwSciError_Success.
 * - Memory value pointed to by dest is updated to value 0.3.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061589}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.4
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var4:0.3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var3>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var4>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_zero_and_srcScaleFactor_set_non-zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_005.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_zero_and_srcScaleFactor_set_non-zero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_zero_and_srcScaleFactor_set_non-zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it succeeds when destScaleFactor value is set to zero and srcScaleFactor value is set to non-zero.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.0
 * src set to allocated memory and the value at memory is set to 0.1.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() returns LwSciError_Success.
 * - Memory value pointed to by dest is updated to value 0.1.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061592}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.0
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var4:0.1
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var3>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var4>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_non-zero_and_srcScaleFactor_set_zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileFloatPyramidConstraints
TEST.NEW
TEST.NAME:TC_006.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_non-zero_and_srcScaleFactor_set_zero
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufReconcileFloatPyramidConstraints.Successful_destScaleFactor_set_non-zero_and_srcScaleFactor_set_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileFloatPyramidConstraints() such that it succeeds when destScaleFactor value is set to non-zero and srcScaleFactor value is set to zero.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileFloatPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.1
 * src set to allocated memory and the value at memory is set to 0.0.}
 *
 * @testbehavior{- LwSciBufReconcileFloatPyramidConstraints() returns LwSciError_Success.
 * - Memory value pointed to by dest is not updated and holds the value 0.1}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061595}
 *
 * @verify{22062154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.1
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var4:0.0
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var3:0.1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileFloatPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var3>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileFloatPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var4>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufReconcileImageBLConstraints

-- Test Case: TC_001.LwSciBufReconcileImageBLConstraints.Panic_due_to_srcConstraints_is_Null
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImageBLConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileImageBLConstraints.Panic_due_to_srcConstraints_is_Null
TEST.BASIS_PATH:1 of 6
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileImageBLConstraints.Panic_due_to_srcConstraints_is_Null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImageBLConstraints() when source constraints set to null.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Blocklinear Image constraints set to allocated memory.
 * Source Blocklinear Image constraints set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855414}
 *
 * @verify{18842775}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufReconcileImageBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImageBLConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileImageBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.BASIS_PATH:2 of 6
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileImageBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImageBLConstraints() Destination constraints set to values greater than
 * source constraints.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Blocklinear Image constraints array to reconcile is initialised.
 * Source Blocklinear Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855417}
 *
 * @verify{18842775}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobSize:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockX:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockY:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockZ:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobSize:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockX:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockY:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockZ:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobSize:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockX:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockY:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockZ:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufReconcileImageBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImageBLConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileImageBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.BASIS_PATH:3 of 6
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileImageBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImageBLConstraints() Source constraints set to values greater than
 * Destination constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Blocklinear Image constraints array to reconcile is initialised.
 * Source Blocklinear Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855420}
 *
 * @verify{18842775}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobSize:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockX:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockY:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockZ:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobSize:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockX:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockY:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src[0].log2GobsperBlockZ:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobSize:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockX:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockY:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest[0].log2GobsperBlockZ:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciBufReconcileImageBLConstraints.Null_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImageBLConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileImageBLConstraints.Null_dstConstraints
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileImageBLConstraints.Null_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImageBLConstraints() when destination pointer is null.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileImageBLConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{'dest'- Destination Blocklinear Image constraints set to NULL
 * 'src'- Source Blocklinear Image constraints set to allocated memory}
 *
 * @testbehavior{- LwSciBufReconcileImageBLConstraints() Panics
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
 * @testcase{22060614}
 *
 * @verify{18842775}
 */


TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.dest:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImageBLConstraints.src:<<malloc 1>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImageBLConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufReconcileImgCommonConstraints

-- Test Case: TC_001.LwSciBufReconcileImgCommonConstraints.Panic_due_to_srcConstraints_is_Null
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImgCommonConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileImgCommonConstraints.Panic_due_to_srcConstraints_is_Null
TEST.BASIS_PATH:1 of 6
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileImgCommonConstraints.Panic_due_to_srcConstraints_is_Null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImgCommonConstraints() source constraints set to null.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Common Image constraints to reconcile set to allocated memory.
 * Source Common Image constraints to reconcile set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855423}
 *
 * @verify{18842772}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufReconcileImgCommonConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImgCommonConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileImgCommonConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.BASIS_PATH:3 of 6
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileImgCommonConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImgCommonConstraints() Destination constraints set to values
 * greater than source constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Common Image constraints array to reconcile is initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855426}
 *
 * @verify{18842772}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].startAddrAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].pitchAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].heightAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].sizeAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].startAddrAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].pitchAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].heightAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].sizeAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].startAddrAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].pitchAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].heightAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].sizeAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufReconcileImgCommonConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImgCommonConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileImgCommonConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.BASIS_PATH:4 of 6
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileImgCommonConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImgCommonConstraints() source constraints set values greater than
 * Destination constraints.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination Common Image constraints array to reconcile is initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855429}
 *
 * @verify{18842772}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].startAddrAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].pitchAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].heightAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].sizeAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].startAddrAlign:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].pitchAlign:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].heightAlign:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src[0].sizeAlign:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].startAddrAlign:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].pitchAlign:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].heightAlign:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest[0].sizeAlign:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciBufReconcileImgCommonConstraints.Null_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileImgCommonConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileImgCommonConstraints.Null_dstConstraints
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileImgCommonConstraints.Null_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileImgCommonConstraints() when destination pointer is null.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileImgCommonConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{'dest'- Destination Common Image constraints to reconcile set to NULL.
 * 'src'- Source Common Image constraints to reconcile set to allocated memory.}
 *
 * @testbehavior{- LwSciBufReconcileImgCommonConstraints() Panics
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
 * @testcase{22060618}
 *
 * @verify{18842772}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.dest:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileImgCommonConstraints.src:<<malloc 1>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileImgCommonConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufReconcileIntPyramidConstraints

-- Test Case: TC_001.LwSciBufReconcileIntPyramidConstraints.Panics_if_dest_is_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileIntPyramidConstraints.Panics_if_dest_is_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileIntPyramidConstraints.Panics_if_dest_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it panics if dest input parameter is NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonPanic() calls exit() with EXIT_FAILURE error code to simulate panic.}
 *
 * @testinput{dest set to null.
 * src set to allocated memory and the value at memory is set to 2.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() Panics}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061599}
 *
 * @verify{22062151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var1:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufReconcileIntPyramidConstraints.Panics_if_src_is_NULL
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileIntPyramidConstraints.Panics_if_src_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileIntPyramidConstraints.Panics_if_src_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it panics if src input parameter is NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciCommonPanic() calls exit() with EXIT_FAILURE error code to simulate panic.}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 6.}
 * src set to null.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() Panics}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061602}
 *
 * @verify{22062151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:6
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var2>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_less_than_destLevelCount
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_less_than_destLevelCount
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_less_than_destLevelCount}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it succeeds when srclevelCount value is set less than destLevelCount.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 4
 * src set to allocated memory and the value at memory is set to 3.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() returns LwSciError_Success.
 * - Memory value pointed to by dest is updated to value 3.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061606}
 *
 * @verify{22062151}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var1:3
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var2>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_greater_than_destLevelCount
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_greater_than_destLevelCount
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileIntPyramidConstraints.Successful_srclevelCount_set_greater_than_destLevelCount}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it succeeds when srclevelCount value is set greater than destLevelCount}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 3
 * src set to allocated memory and the value at memory is set to 4.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() returns LwSciError_Success
 * - Memory value pointed to by dest is not updated and holds the value 3.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061608}
 *
 * @verify{22062151}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var1:4
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var2>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_non-zero_and_srclevelCount_set_zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_005.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_non-zero_and_srclevelCount_set_zero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_non-zero_and_srclevelCount_set_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it succeeds when destLevelCount value is set to non-zero and srclevelCount value is set to zero.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 1
 * src set to allocated memory and the value at memory is set to 0.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() returns LwSciError_Success
 * - Memory value pointed to by dest is not updated and holds the value 1.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061612}
 *
 * @verify{22062151}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var1:0
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var2>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_zero_and_srclevelCount_set_non-zero
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileIntPyramidConstraints
TEST.NEW
TEST.NAME:TC_006.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_zero_and_srclevelCount_set_non-zero
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufReconcileIntPyramidConstraints.Successful_destLevelCount_set_zero_and_srclevelCount_set_non-zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileIntPyramidConstraints() such that it succeeds when destLevelCount value is set to zero and srclevelCount value is set to non-zero.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileIntPyramidConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{dest set to allocated memory and the value at memory is set to 0.
 * src set to allocated memory and the value at memory is set to 1.}
 *
 * @testbehavior{- LwSciBufReconcileIntPyramidConstraints() returns LwSciError_Success
 * - Memory value pointed to by dest is updated to value 1.}
 *
 * @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
 * - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
 * - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
 * - Execute the test and check all assertions pass.}
 *
 * @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
 *
 * @testcase{22061614}
 *
 * @verify{22062151}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var1:1
TEST.VALUE:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:0
TEST.EXPECTED:lwscibuf_constraint_lib_common.<<GLOBAL>>.var2:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileIntPyramidConstraints
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.dest>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var2>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src
<<lwscibuf_constraint_lib_common.LwSciBufReconcileIntPyramidConstraints.src>> = ( &<<lwscibuf_constraint_lib_common.<<GLOBAL>>.var1>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufReconcileOutputBLConstraints

-- Test Case: TC_001.LwSciBufReconcileOutputBLConstraints.Panic_due_to_srcConstraints_is_Null
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputBLConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileOutputBLConstraints.Panic_due_to_srcConstraints_is_Null
TEST.BASIS_PATH:1 of 6
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileOutputBLConstraints.Panic_due_to_srcConstraints_is_Null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputBLConstraints() when source constraints set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints to reconcile set to allocated memory.
 * Source Common Image constraints to reconcile set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855432}
 *
 * @verify{18842769}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufReconcileOutputBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputBLConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileOutputBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.BASIS_PATH:2 of 6
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileOutputBLConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputBLConstraints() Destination constraints set to values
 * greater than source constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints array to reconcile is initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855435}
 *
 * @verify{18842769}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].startAddrAlign:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].pitchAlign:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].heightAlign:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].sizeAlign:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobSize:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockX:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockY:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockZ:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobSize:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockX:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockY:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockZ:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].startAddrAlign:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].pitchAlign:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].heightAlign:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].sizeAlign:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobSize:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockX:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockY:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockZ:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufReconcileOutputBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputBLConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileOutputBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints
TEST.BASIS_PATH:3 of 6
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileOutputBLConstraints.Success_due_to_srcConstraints_set_greater_than_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputBLConstraints() source constraints set to values greater than
 * destination constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints array to reconcile is initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and source variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855438}
 *
 * @verify{18842769}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].startAddrAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].pitchAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].heightAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].sizeAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobSize:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockX:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockY:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockZ:0
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobSize:4
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockX:5
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockY:6
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src[0].log2GobsperBlockZ:7
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].startAddrAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].pitchAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].heightAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].sizeAlign:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobSize:4
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockX:5
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockY:6
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest[0].log2GobsperBlockZ:7
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciBufReconcileOutputBLConstraints.Null_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputBLConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileOutputBLConstraints.Null_dstConstraints
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileOutputBLConstraints.Null_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputBLConstraints() when Destination constraints set to NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileOutputBLConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{'dest'- Destination Common Image constraints to reconcile set to NULL.
 * 'src'- Source Common Image constraints to reconcile set to allocated memory.}
 *
 * @testbehavior{- LwSciBufReconcileOutputBLConstraints() Panics
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
 * @testcase{22060638}
 *
 * @verify{18842769}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.dest:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputBLConstraints.src:<<malloc 1>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputBLConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufReconcileOutputImgConstraints

-- Test Case: TC_001.LwSciBufReconcileOutputImgConstraints.Panic_due_to_srcConstraints_is_Null
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputImgConstraints
TEST.NEW
TEST.NAME:TC_001.LwSciBufReconcileOutputImgConstraints.Panic_due_to_srcConstraints_is_Null
TEST.BASIS_PATH:1 of 6
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufReconcileOutputImgConstraints.Panic_due_to_srcConstraints_is_Null}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputImgConstraints() -  when source constraints set to null.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints to reconcile set to allocated memory.
 * Source Common Image constraints to reconcile set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18855441}
 *
 * @verify{18842766}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src:<<null>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufReconcileOutputImgConstraints.Success_due_to_srcConstraints_set_gretaer_than_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputImgConstraints
TEST.NEW
TEST.NAME:TC_002.LwSciBufReconcileOutputImgConstraints.Success_due_to_srcConstraints_set_gretaer_than_dstConstraints
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufReconcileOutputImgConstraints.Success_due_to_srcConstraints_set_gretaer_than_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputImgConstraints() source constraints set to values greater than
 * than destination constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints array to reconcile are initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855444}
 *
 * @verify{18842766}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].startAddrAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].pitchAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].heightAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].sizeAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobSize:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockX:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockY:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockZ:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].startAddrAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].pitchAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].heightAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].sizeAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].startAddrAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].pitchAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].heightAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].sizeAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobSize:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockX:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockY:1
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockZ:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciBufReconcileOutputImgConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputImgConstraints
TEST.NEW
TEST.NAME:TC_003.LwSciBufReconcileOutputImgConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints
TEST.BASIS_PATH:2 of 6
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufReconcileOutputImgConstraints.Success_due_to_dstConstraints_set_greater_than_srcConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputImgConstraints() - Destination constraints set to values
 * greater than source constraints}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{Destination HW constraints array to reconcile is initialised.
 * Source Common Image constraints array to reconcile is initialised.}
 *
 * @testbehavior{Returns LwSciError_Success and destination variables which is max of each variable between source and destination constraints.}
 *
 * @testcase{18855447}
 *
 * @verify{18842766}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].startAddrAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].pitchAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].heightAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].sizeAlign:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobSize:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockX:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockY:2
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockZ:3
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src:<<malloc 1>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].startAddrAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].pitchAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].heightAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src[0].sizeAlign:1
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].startAddrAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].pitchAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].heightAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].sizeAlign:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobSize:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockX:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockY:2
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest[0].log2GobsperBlockZ:3
TEST.EXPECTED:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.return:LwSciError_Success
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciBufReconcileOutputImgConstraints.Null_dstConstraints
TEST.UNIT:lwscibuf_constraint_lib_common
TEST.SUBPROGRAM:LwSciBufReconcileOutputImgConstraints
TEST.NEW
TEST.NAME:TC_004.LwSciBufReconcileOutputImgConstraints.Null_dstConstraints
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufReconcileOutputImgConstraints.Null_dstConstraints}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufReconcileOutputImgConstraints() when Destination constraints set to NULL.}
 *
 * @testpurpose{Unit testing of LwSciBufReconcileOutputImgConstraints().}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None}
 *
 * @testinput{'dest'- Destination Common Image constraints to reconcile set to NULL.
 * 'src'- Source Common Image constraints to reconcile set to allocated memory.}
 *
 * @testbehavior{- LwSciBufReconcileOutputImgConstraints() Panics
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
 * @testcase{22060641}
 *
 * @verify{18842766}
 */



TEST.END_NOTES:
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.dest:<<null>>
TEST.VALUE:lwscibuf_constraint_lib_common.LwSciBufReconcileOutputImgConstraints.src:<<malloc 1>>
TEST.FLOW
  lwscibuf_constraint_lib_common.c.LwSciBufReconcileOutputImgConstraints
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END
