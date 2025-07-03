// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Control/tests/TestPrinting.h>

// -----------------------------------------------------------------------------
class TestRtPrintfNewAndLegacy : public TestPrintingNoKnobs, public ::testing::WithParamInterface<const char*>
{
};

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintSimpleString )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print" );
    const std::string result = launch( 1 );
    EXPECT_EQ( "printf successfully exelwted!\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintConstant )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_const" );
    const std::string result = launch( 1 );
    EXPECT_EQ( "int: 7\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintConstant64 )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_const64" );
    const std::string result = launch( 1 );
    EXPECT_EQ( "long long int: 100200300400500\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintMultiplePrintfStatements )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_multiplePrintfStatements" );
    const std::string result = launch( 1 );
    EXPECT_EQ(
        "char: A\nint: -10\nunsigned int: 20\nlong int: -30\nlong unsigned int: 40\nlong long int: -50\nlong long "
        "unsigned int: 60\nfloat: 42.000000\ndouble: 42.000000\n",
        result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintMultiplePrintfParameters )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_multiplePrintfParameters" );
    const std::string result = launch( 1 );
    EXPECT_EQ(
        "char: A\nint: -10\nunsigned int: 20\nlong int: -30\nlong unsigned int: 40\nlong long int: -50\nlong long "
        "unsigned int: 60\nfloat: 42.000000\ndouble: 42.000000\n",
        result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintInputFromBuffer )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_inputFromBuffer" );
    const std::string result = launch( 1 );
    EXPECT_EQ( "input from buffer: 13\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintLaunchIndex )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_launchindex" );
    const std::string result = launch( 3 );
    EXPECT_EQ( "launch index: 0 0 0\nlaunch index: 1 0 0\nlaunch index: 2 0 0\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintWithRestrictedLaunchIndex )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_launchindex" );
    m_context->setPrintLaunchIndex( 1, 0, 0 );
    const std::string result = launch( 3 );
    EXPECT_EQ( "launch index: 1 0 0\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintExceptionDetails )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_exception_details" );
    const std::string result = launch( 1 );
    EXPECT_EQ(
        "Caught RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS\n  launch index   : 1, 2, 3\n  buffer address : 0x64\n  "
        "dimensionality : 3\n  size           : 200x200x200\n  element size   : 8\n  accessed index : 4, 5, 6\n",
        result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintInConditional )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_conditional" );
    const std::string result = launch( 1 );
    EXPECT_EQ( "You should see this.\n", result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintInLoop )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_loop" );
    const std::string result = launch( 1 );
    EXPECT_EQ(
        "iteration 0\niteration 1\niteration 2\niteration 3\niteration 4\niteration 5\niteration 6\niteration "
        "7\niteration 8\niteration 9\niteration 10\niteration 11\niteration 12\n",
        result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintInLoopWithPrintActiveHoistedOut )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_loop_with_final_rtPrintf" );
    const std::string result = launch( 1 );
    EXPECT_EQ(
        "iteration 0\niteration 1\niteration 2\niteration 3\niteration 4\niteration 5\niteration 6\niteration "
        "7\niteration 8\niteration 9\niteration 10\niteration 11\niteration 12\nDone!\n",
        result );
}

////////////////////////////////////////////////////////////////////////////////

TEST_P( TestRtPrintfNewAndLegacy, TestPrintingDisabled )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_loop" );
    m_context->setPrintEnabled( false );
    const std::string result = launch( 10 );
    EXPECT_EQ( "", result );
}

////////////////////////////////////////////////////////////////////////////////

// This test is expected to recompile for the second launch because megakernel
// removes all the printing code if printing is disabled entirely.
TEST_P( TestRtPrintfNewAndLegacy, TestPrintingDisabledBetweenLaunches )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_launchindex" );
    const std::string result = launch( 3 );
    EXPECT_EQ( "launch index: 0 0 0\nlaunch index: 1 0 0\nlaunch index: 2 0 0\n", result );
    m_context->setPrintEnabled( false );
    const std::string result2 = launch( 3 );
    EXPECT_EQ( "", result2 );
}

////////////////////////////////////////////////////////////////////////////////

// This test is expected not to recompile for the second launch.
TEST_P( TestRtPrintfNewAndLegacy, TestPrintIndexChangedBetweenLaunches )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_launchindex" );
    const std::string result = launch( 3 );
    EXPECT_EQ( "launch index: 0 0 0\nlaunch index: 1 0 0\nlaunch index: 2 0 0\n", result );
    m_context->setPrintLaunchIndex( 1, 0, 0 );
    const std::string result2 = launch( 3 );
    EXPECT_EQ( "launch index: 1 0 0\n", result2 );
}

////////////////////////////////////////////////////////////////////////////////
TEST_P( TestRtPrintfNewAndLegacy, TestPrintAddress )
{
    const char* ptxFileName = GetParam();
    setupProgram( ptxFileName, "print_address" );
    const std::string result   = launch( 1 );
    bool              notEmpty = !result.empty();
    EXPECT_TRUE( notEmpty );
}

// -----------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P( NewRtPrintf, TestRtPrintfNewAndLegacy, ::testing::Values( "rtPrintf.lw" ) );
INSTANTIATE_TEST_SUITE_P( LegacyRtPrintf, TestRtPrintfNewAndLegacy, ::testing::Values( "rtPrintf_legacy.lw" ) );

// The following tests are here to verify that we can handle the management of legacy ptx code constructs.
// -----------------------------------------------------------------------------
class TestRtPrintfLegacyPTX : public TestPrintingNoKnobs
{
};

struct PTXModule
{
    const char* description;
    const char* functionName;
    const char* code;
};

// clang-format off
#define PTX_MODULE( functionName, ... )\
{ "", functionName, #__VA_ARGS__ }

PTXModule printPtx = PTX_MODULE( "print",
.version 4.2
.target sm_20
.address_size 64
.global .align 1 .b8 $str[31] = {112, 114, 105, 110, 116, 102, 32, 115, 117, 99, 99, 101, 115, 115, 102, 117, 108, 108, 121, 32, 101, 120, 101, 99, 117, 116, 101, 100, 33, 10, 0};

.visible .entry _Z5printv(

)
{
  .reg .pred  %p<2>;
  .reg .s32   %r<4>;
  .reg .s64   %rd<3>;

  call (%r1), _rt_print_active, ();
  setp.eq.s32 %p1, %r1, 0;
  @%p1 bra  BB0_2;

  mov.u64   %rd2, $str;
  cvta.global.u64   %rd1, %rd2;
  // %r3 contains the size of the format string (plus alignement) but it is ignored by the C14n pass.
  mov.u32   %r3, 36;
  call (%r2), _rt_print_start_64, (%rd1, %r3);

BB0_2:
  ret;
}

);

////////////////////////////////////////////////////////////////////////////////

// This test is a copy of the TestPrintsimpleString, the difference is that the
// program is generated from a ptx string rather than a lwca program.
TEST_F ( TestRtPrintfLegacyPTX, TestPrintSimpleStringFromPTX ) {
  setupProgramFromPTXString( printPtx.code, printPtx.functionName);
  const std::string result = launch( 1 );
  EXPECT_EQ( "printf successfully exelwted!\n", result );
}

////////////////////////////////////////////////////////////////////////////////

PTXModule printWithStrlenPtx = PTX_MODULE( "print_with_strlen",
.version 4.2
.target sm_20
.address_size 64
.global .align 1 .b8 $str[48] = {112,114,105,110,116,102,32,119,105,116,104,32,115,116,114,108,101,110,32,108,111,111,112,32,115,117,99,99,101,115,115,102,117,108,108,121,32,101,120,101,99,117,116,101,100,33,10,0};


.visible .entry _Z17print_with_strlen(

)
{
  .reg .pred  %p<3>;
  .reg .s32   %r<16>;
  .reg .s64   %rd<5>;

  call (%r1), _rt_print_active, ();
  setp.eq.s32 %p1, %r1, 0;
  @%p1 bra  exitingBlock;

  // Loop that computes the size of the string.
  // This loop is not usually generated by newer versions of lwcc.
  mov.u64   %rd1, $str;
  $strlenBlock:
  add.u64   %rd1, %rd1, 1;
  ld.const.s8   %r5, [%rd1+0];
  mov.u32   %r6, 0;
  setp.ne.s32   %p2, %r5, %r6;
  @%p2 bra  $strlenBlock;

  // printBlock:
  mov.u64   %rd2, $str;
  cvt.s32.u64   %r8, %rd2;
  mov.u64   %rd4, $str;
  cvt.s32.u64   %r9, %rd4;
  sub.s32   %r10, %r8, %r9;
  add.s32   %r11, %r10, 8;
  and.b32   %r12, %r11, -4;
  mov.u32   %r14, %r12;
  call (%r15), _rt_print_start_64, (%rd4, %r14);
  bra   exitingBlock;

exitingBlock:
  ret;
}

);

// This tests verify that we can manage the presence of legacy strlen loop in the
// code that generates the printf.
TEST_F ( TestRtPrintfLegacyPTX, TestPrintsimpleStringWithStrlenLoopFromPTX ) {
  setupProgramFromPTXString( printWithStrlenPtx.code, printWithStrlenPtx.functionName);
  const std::string result = launch( 1 );
  EXPECT_EQ( "printf with strlen loop successfully exelwted!\n", result );
}

////////////////////////////////////////////////////////////////////////////////

PTXModule printWithStrlenAndArgumentPtx = PTX_MODULE( "print_with_strlen_and_argument",
.version 4.2
.target sm_20
.address_size 64
.global .align 1 .b8 $str1[9] = {105, 110, 116, 58, 32, 37, 100, 10, 0};

.visible .entry _Z30print_with_strlen_and_argument(

)
{
  .reg .pred  %p<3>;
  .reg .s32   %r<18>;
  .reg .s64   %rd<5>;

  call (%r1), _rt_print_active, ();
  setp.eq.s32 %p1, %r1, 0;
  @%p1 bra  exitingBlock;

  mov.u64   %rd1, $str1;
  $strlenBlock:
  add.u64   %rd1, %rd1, 1;
  ld.const.s8   %r5, [%rd1+0];
  mov.u32   %r6, 0;
  setp.ne.s32   %p2, %r5, %r6;
  @%p2 bra  $strlenBlock;

  // printBlock:
  mov.u64   %rd2, $str1;
  cvt.s32.u64   %r8, %rd2;
  mov.u64   %rd4, $str1;
  cvt.s32.u64   %r9, %rd4;
  sub.s32   %r10, %r8, %r9;
  add.s32   %r11, %r10, 8;
  and.b32   %r12, %r11, -4;
  mov.u32   %r14, %r12;
  call (%r15), _rt_print_start_64, (%rd4, %r14);
  setp.eq.s32 %p2, %r15, 0;
  @%p2 bra  exitingBlock;

  mov.u32   %r16, 0;
  call (), _rt_print_write32, (%r16, %r3);
  add.s32   %r8, %r3, 4;
  mov.u32   %r7, 7;
  call (), _rt_print_write32, (%r7, %r8);

exitingBlock:
  ret;
}

);

TEST_F ( TestRtPrintfLegacyPTX, TestPrintsimpleStringWithStrlenLoopAndArgumentFromPTX ) {
  setupProgramFromPTXString( printWithStrlenAndArgumentPtx.code, printWithStrlenAndArgumentPtx.functionName);
  const std::string result = launch( 1 );
  EXPECT_EQ( "int: 7\n", result );
}

////////////////////////////////////////////////////////////////////////////////

// Generated from:
// rtPrintf( "int: %d\n", (long int) 3);
// With lwcc --ptx --optimize 0 

PTXModule printArgLoopPtx = PTX_MODULE( "print_int_with_arg_loop",
.version 4.2
.target sm_20
.address_size 64
.global .align 1 .b8 $str[9] = {105, 110, 116, 58, 32, 37, 100, 10, 0};

.visible .entry _Z23print_int_with_arg_loopv(
)
{
  .local .align 8 .b8   __local_depot0[8];
  .reg .b64   %SP;
  .reg .b64   %SPL;
  .reg .pred  %p<5>;
  .reg .s32   %r<31>;
  .reg .s64   %rd<12>;

  mov.u64   %rd11, __local_depot0;
  cvta.local.u64  %SP, %rd11;
  call (%r11), _rt_print_active, ();
  setp.eq.s32 %p1, %r11, 0;
  @%p1 bra  BB0_6;

  mov.u32   %r14, 8;
  mov.u32   %r15, 4;
  max.u32   %r16, %r15, %r14;
  add.s32   %r13, %r16, 20;
  mov.u64   %rd5, $str;
  cvta.global.u64   %rd4, %rd5;
  call (%r12), _rt_print_start_64, (%rd4, %r13);
  setp.eq.s32 %p2, %r12, 0;
  @%p2 bra  BB0_6;

  add.u64   %rd6, %SP, 0;
  cvta.to.local.u64   %rd10, %rd6;
  mov.u64   %rd8, 3;
  st.local.u64  [%rd10], %rd8;
  mov.u32   %r17, 0;
  call (), _rt_print_write32, (%r17, %r12);
  max.s32   %r2, %r15, %r14;
  setp.lt.s32 %p3, %r2, 4;
  @%p3 bra  BB0_6;

  shr.s32   %r23, %r2, 31;
  shr.u32   %r24, %r23, 30;
  add.s32   %r25, %r2, %r24;
  shr.s32   %r3, %r25, 2;
  add.s32   %r28, %r12, 4;
  mov.u32   %r30, 1;
  mov.u32   %r29, 3;
  bra.uni   BB0_4;

BB0_5:
  add.s64   %rd10, %rd2, 4;
  ld.local.u32  %r29, [%rd2+4];
  add.s32   %r30, %r30, 1;
  add.s32   %r28, %r28, 4;

BB0_4:
  mov.u32   %r6, %r29;
  mov.u64   %rd2, %rd10;
  call (), _rt_print_write32, (%r6, %r28);
  setp.ge.s32 %p4, %r30, %r3;
  @%p4 bra  BB0_6;
  bra.uni   BB0_5;

BB0_6:
  ret;
}
);

TEST_F ( TestRtPrintfLegacyPTX, TestPrintIntConstWithArgLoopFromPTX ) {
  setupProgramFromPTXString( printArgLoopPtx.code, printArgLoopPtx.functionName);
  const std::string result = launch( 1 );
  EXPECT_EQ( "int: 3\n", result );
}

////////////////////////////////////////////////////////////////////////////////

// %r3 contains the size of the format string (plus alignement) but it is ignored by the C14n pass.
PTXModule printPtxWithFormatStringInConst = PTX_MODULE( "ptxWithFormatStringInConst",
.version 4.2
.target sm_20
.address_size 64
.const .align 1 .b8 $str[31] = {112, 114, 105, 110, 116, 102, 32, 115, 117, 99, 99, 101, 115, 115, 102, 117, 108, 108, 121, 32, 101, 120, 101, 99, 117, 116, 101, 100, 33, 10, 0};

.visible .entry _Z26ptxWithFormatStringInConstv(

)
{
  .reg .pred  %p<2>;
  .reg .s32   %r<4>;
  .reg .s64   %rd<3>;

  call (%r1), _rt_print_active, ();
  setp.eq.s32 %p1, %r1, 0;
  @%p1 bra  BB0_2;

  mov.u64   %rd2, $str;
  cvta.global.u64   %rd1, %rd2;
  mov.u32   %r3, 36;
  call (%r2), _rt_print_start_64, (%rd1, %r3);

BB0_2:
  ret;
}

);

TEST_F (TestRtPrintfLegacyPTX, TestPrintFromConstMemoryFromPTX ) {
  setupProgramFromPTXString( printPtxWithFormatStringInConst.code, printPtxWithFormatStringInConst.functionName );
  const std::string result = launch(1);
  EXPECT_EQ( "printf successfully exelwted!\n", result );
}

////////////////////////////////////////////////////////////////////////////////

PTXModule printPtxWithLongLongInLoop = PTX_MODULE( "ptxWithLongLongInLoop",
.version 4.2
.target sm_20
.address_size 64
.const .align 1 .b8 $str[18] = {108,111,110,103,32,108,111,110,103,32,105,110,116,58,32,37,100,0};

.visible .entry _Z21ptxWithLongLongInLoop(

)
{
  .local .align 8 .b8   __local_depot[8];
  .local .align 8 .b8   __local_depot2[4];
  .reg .b64   %SP;
  .reg .b64   %local;
  .reg .pred  %p<5>;
  .reg .u32   %r<20>;
  .reg .u64   %rd<20>;

  mov.u64   %SP, __local_depot;
  mov.u64   %local, __local_depot2;
  mov.u32   %r17,  8;
  st.local.u32 [%local], %r17;
  call (%r1), _rt_print_active, ();
  setp.eq.u32 %p1, %r1, 0;
  @%p1 bra  BB3;

  mov.u32   %r14, 8;
  mov.u32   %r18, 4;
  mov.u64   %rd2, $str;
  cvta.global.u64   %rd1, %rd2;
  mov.u32   %r3, 20;
  call (%r2), _rt_print_start_64, (%rd1, %r3);
  setp.eq.s32 %p2, %r2, 0;
  @%p2 bra  BB3;

BB1:
  mov.u32  %r10, 10;
  cvt.u64.u32  %rd4, %r10;
  st.local.u64  [%SP], %rd4;
  mov.u32   %r4, 1;
  mov.u32   %r5, 1;
  call (), _rt_print_write32, (%r4, %r5);
  mov.u32   %r6, 0;
  max.s32   %r2, %r18, %r14;
  setp.lt.s32 %p3, %r2, 4;
  @%p3 bra  BB3;

BB2:
  cvt.u64.u32   %rd4, %r6;
  add.u64       %rd5,  %SP, %rd4; 
  ld.local.u32  %r8, [%rd5];
  mov.u32 %r15, 0;
  call (), _rt_print_write32, (%r8, %r15);
  add.u32   %r6, %r6, 4;
  ld.local.u32 %r9, [%local];
  setp.lt.u32   %p4, %r6, %r9;
  @%p4 bra   BB2;

BB3:
  ret;

}

);
// clang-format on

// This unit test tests the case in which a call to _print_write32(1) controls a single instance to _print_write32.
// This happens when the ptx compiler cannot peel the printing loop (in lumiscaphe for example).
TEST_F( TestRtPrintfLegacyPTX, TestPrintLongLongInLoopFromPTX )
{
    setupProgramFromPTXString( printPtxWithLongLongInLoop.code, printPtxWithLongLongInLoop.functionName );
    const std::string result = launch( 1 );
    EXPECT_EQ( "long long int: 10", result );
}

// -----------------------------------------------------------------------------
class TestSetPrintBufferSizeExceptions : public TestPrintingNoKnobs
{
};

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetPrintBufferSizeExceptions, CanSetBufferSize )
{
    setupProgram( "rtPrintf.lw", "print" );
    EXPECT_NO_THROW( m_context->setPrintBufferSize( 100 ) );
    launch( 1 );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetPrintBufferSizeExceptions, SetBufferSizeAfterFirstLaunchCausesException )
{
    setupProgram( "rtPrintf.lw", "print" );
    m_context->setPrintBufferSize( 100 );
    launch( 1 );
    EXPECT_ANY_THROW( m_context->setPrintBufferSize( 200 ) );
}
