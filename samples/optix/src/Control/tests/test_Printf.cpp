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

#include <lwca.h>


class TestPrintf : public TestPrintingNoKnobs, public ::testing::WithParamInterface<const char*>
{
  public:
    std::string getOutputString( std::string lwdaFile, std::string functionName );
    std::string getOutputStringFromPTX( std::string ptxString, std::string functionName );

    // TEMPORARY: See lwbugs/3521134
    void launchNoCapture( std::string lwdaFile, std::string functionName );
};

std::string TestPrintf::getOutputString( std::string lwdaFile, std::string functionName )
{
    setupProgram( lwdaFile.c_str(), functionName.c_str() );
    m_context["value"]->setFloat( 3.141592f );
    return launch( 1 );
}

std::string TestPrintf::getOutputStringFromPTX( std::string ptxString, std::string functionName )
{
    setupProgramFromPTXString( ptxString.c_str(), functionName.c_str() );
    m_context["value"]->setFloat( 3.141592f );
    return launch( 1 );
}

// TEMPORARY: Launch the kernel without capturing any output. Device prints
// should appear in the robot log. See lwbugs/3521134
void TestPrintf::launchNoCapture( std::string lwdaFile, std::string functionName )
{
    setupProgram( lwdaFile.c_str(), functionName.c_str() );
    m_context["value"]->setFloat( 3.141592f );
    TestPrintingNoKnobs::launchNoCapture( 1 );
}

TEST_F( TestPrintf, PrintfNoCapture )
{
    TestPrintf::launchNoCapture( "printf.lw", "rg" );
}

TEST_F( TestPrintf, PrintfWorks )
{
    std::string result = getOutputString( "printf.lw", "rg" );
    int         major  = LWDA_VERSION / 1000;
    int         middle = ( LWDA_VERSION % 1000 ) / 10;
    int         minor  = LWDA_VERSION % 10;
    EXPECT_EQ( "[0]: Testing from LWCA version " + std::to_string( major ) + "." + std::to_string( middle ) + "."
                   + std::to_string( minor ) + "\n[0]: \tvalue = 3.141592\n",
               result );
}

TEST_F( TestPrintf, PrintfStructWorks )
{
    std::string result = getOutputString( "printf.lw", "struct_test" );
    int         major  = LWDA_VERSION / 1000;
    int         middle = ( LWDA_VERSION % 1000 ) / 10;
    int         minor  = LWDA_VERSION % 10;
    EXPECT_EQ( "[0]: Testing from LWCA version " + std::to_string( major ) + "." + std::to_string( middle ) + "."
                   + std::to_string( minor )
                   + "\n[0]: \tts0: test_0\tts1: 1, test_1\tts2: test_2, 2\tts3: 0.5, 3, test_3\n",
               result );
}

struct PTXModule
{
    const char* description;
    const char* functionName;
    const char* code;
};

// clang-format off
#define PTX_MODULE( functionName, ... )\
{ "", functionName, #__VA_ARGS__ }

PTXModule printfLwda40 = PTX_MODULE( "rg",
  .version 2.3
  .target sm_20
  .address_size 64
  // compiled with C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/bin/../open64/lib//be.exe
  // lwopencc 4.0 built on 2011-05-13

  .extern .func (.param .s32 __lwdaretf_printf) printf (.param .u64 __lwdaparmf1_printf)

  .extern .func (.param .s32 __lwdaretf_vprintf) vprintf (.param .u64 __lwdaparmf1_vprintf, .param .u64 __lwdaparmf2_vprintf)

  .visible .func _ZN5optix16rt_undefined_useEi (.param .s32 __lwdaparmf1__ZN5optix16rt_undefined_useEi)

  .visible .func _ZN5optix18rt_undefined_use64Ey (.param .u64 __lwdaparmf1__ZN5optix18rt_undefined_use64Ey)

  //-----------------------------------------------------------
  // Compiling C:/Users/jbigler/AppData/Local/Temp/tmpxft_001072b4_00000000-11_printf.cpp3.i (C:/Users/jbigler/AppData/Local/Temp/ccBI#.a77988)
  //-----------------------------------------------------------

  //-----------------------------------------------------------
  // Options:
  //-----------------------------------------------------------
  //  Target:ptx, ISA:sm_20, Endian:little, Pointer Size:64
  //  -O3 (Optimization level)
  //  -g0 (Debug level)
  //  -m2 (Report advisories)
  //-----------------------------------------------------------

  .file 1 "C:/Users/jbigler/AppData/Local/Temp/tmpxft_001072b4_00000000-10_printf.lwdafe2.gpu"
  .file 2 "c:\code\rtsdk\rel3.0\include\internal\optix_defines.h"
  .file 3 "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin/../../VC/\INCLUDE\crtdefs.h"
  .file 4 "C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/include\crt/device_runtime.h"
  .file 5 "C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/include\host_defines.h"
  .file 6 "C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/include\builtin_types.h"
  .file 7 "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\device_types.h"
  .file 8 "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\driver_types.h"
  .file 9 "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\surface_types.h"
  .file 10  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\texture_types.h"
  .file 11  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\vector_types.h"
  .file 12  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\builtin_types.h"
  .file 13  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\host_defines.h"
  .file 14  "C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/include\device_launch_parameters.h"
  .file 15  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\crt\storage_class.h"
  .file 16  "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin/../../VC/\INCLUDE\time.h"
  .file 17  "C:/code/rtsdk/rel3.0/tests/Unit/printf/printf.lw"
  .file 18  "c:\code\rtsdk\rel3.0\include\internal/optix_internal.h"
  .file 19  "c:\code\rtsdk\rel3.0\include\optix_device.h"
  .file 20  "C:/Program Files/LWPU GPU Computing Toolkit/LWCA/v4.0/include\common_functions.h"
  .file 21  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\math_functions.h"
  .file 22  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\math_constants.h"
  .file 23  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\device_functions.h"
  .file 24  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\sm_11_atomic_functions.h"
  .file 25  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\sm_12_atomic_functions.h"
  .file 26  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\sm_13_double_functions.h"
  .file 27  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\sm_20_atomic_functions.h"
  .file 28  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\sm_20_intrinsics.h"
  .file 29  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\surface_functions.h"
  .file 30  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\texture_fetch_functions.h"
  .file 31  "c:\program files\lwpu gpu computing toolkit\lwca\v4.0\include\math_functions_dbl_ptx3.h"

  .global .f32 value;
  .global .align 4 .b8 launch_index[4];
  .global .align 1 .b8 __constant879[42] = {0x5b,0x25,0x75,0x5d,0x3a,0x20,0x54,0x65,0x73,0x74,0x69,0x6e,0x67,0x20,0x66,0x72,0x6f,0x6d,0x20,0x43,0x55,0x44,0x41,0x20,0x76,0x65,0x72,0x73,0x69,0x6f,0x6e,0x20,0x25,0x64,0x2e,0x25,0x64,0x2e,0x25,0x64,0xa,0x0};
  .global .align 1 .b8 __constant881[19] = {0x5b,0x25,0x75,0x5d,0x3a,0x20,0x9,0x76,0x61,0x6c,0x75,0x65,0x20,0x3d,0x20,0x25,0x66,0xa,0x0};

  .entry _Z2rgv
  {
  .reg .u32 %r<7>;
  .reg .u64 %rd<6>;
  .reg .f32 %f<3>;
  .reg .f64 %fd<3>;
  .param .u64 __lwdaparma1_vprintf;
  .param .u64 __lwdaparma2_vprintf;
  .local .align 8 .b8 __lwda___lwda__temp__valist_array_38_160[16];
  .loc  17  10  0
$LDWbegin__Z2rgv:
  .loc  17  12  0
  ld.global.u32   %r1, [launch_index+0];
  st.local.u32  [__lwda___lwda__temp__valist_array_38_160+0], %r1;
  mov.s32   %r2, 4;
  st.local.s32  [__lwda___lwda__temp__valist_array_38_160+4], %r2;
  mov.s32   %r3, 0;
  st.local.s32  [__lwda___lwda__temp__valist_array_38_160+8], %r3;
  mov.s32   %r4, 0;
  st.local.s32  [__lwda___lwda__temp__valist_array_38_160+12], %r4;
  cvta.global.u64   %rd1, __constant879;
  st.param.u64  [__lwdaparma1_vprintf], %rd1;
  cvta.local.u64  %rd2, __lwda___lwda__temp__valist_array_38_160;
  st.param.u64  [__lwdaparma2_vprintf], %rd2;
  call.uni (_), vprintf, (__lwdaparma1_vprintf, __lwdaparma2_vprintf);
  .loc  17  13  0
  ld.global.u32   %r5, [launch_index+0];
  st.local.u32  [__lwda___lwda__temp__valist_array_38_160+0], %r5;
  ld.global.f32   %f1, [value];
  cvt.ftz.f64.f32   %fd1, %f1;
  st.local.f64  [__lwda___lwda__temp__valist_array_38_160+8], %fd1;
  cvta.global.u64   %rd3, __constant881;
  st.param.u64  [__lwdaparma1_vprintf], %rd3;
  cvta.local.u64  %rd4, __lwda___lwda__temp__valist_array_38_160;
  st.param.u64  [__lwdaparma2_vprintf], %rd4;
  call.uni (_), vprintf, (__lwdaparma1_vprintf, __lwdaparma2_vprintf);
  .loc  17  14  0
  exit;
$LDWend__Z2rgv:
  } // _Z2rgv

  .visible .func _ZN5optix16rt_undefined_useEi (.param .s32 __lwdaparmf1__ZN5optix16rt_undefined_useEi)
  {
  .loc  18  39  0
$LDWbegin__ZN5optix16rt_undefined_useEi:
  .loc  18  41  0
  ret;
$LDWend__ZN5optix16rt_undefined_useEi:
  } // _ZN5optix16rt_undefined_useEi

  .visible .func _ZN5optix18rt_undefined_use64Ey (.param .u64 __lwdaparmf1__ZN5optix18rt_undefined_use64Ey)
  {
  .loc  18  49  0
$LDWbegin__ZN5optix18rt_undefined_use64Ey:
  .loc  18  51  0
  ret;
$LDWend__ZN5optix18rt_undefined_use64Ey:
  } // _ZN5optix18rt_undefined_use64Ey
  .global .u64 _ZN21rti_internal_register20reg_bitness_detectorE;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail0E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail1E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail2E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail3E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail4E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail5E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail6E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail7E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail8E;
  .global .u64 _ZN21rti_internal_register24reg_exception_64_detail9E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail0E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail1E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail2E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail3E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail4E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail5E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail6E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail7E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail8E;
  .global .u32 _ZN21rti_internal_register21reg_exception_detail9E;
  .global .u32 _ZN21rti_internal_register14reg_rayIndex_xE;
  .global .u32 _ZN21rti_internal_register14reg_rayIndex_yE;
  .global .u32 _ZN21rti_internal_register14reg_rayIndex_zE;
  .global .align 4 .b8 _ZN21rti_internal_typeinfo5valueE[8] = {82,97,121,0,4,0,0,0};
  .global .align 4 .b8 _ZN21rti_internal_typeinfo12launch_indexE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename5valueE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .align 1 .b8 _ZN21rti_internal_typename12launch_indexE[6] = {0x75,0x69,0x6e,0x74,0x31,0x0};
  .global .align 1 .b8 _ZN21rti_internal_semantic5valueE[1] = {0x0};
  .global .align 1 .b8 _ZN21rti_internal_semantic12launch_indexE[14] = {0x72,0x74,0x4c,0x61,0x75,0x6e,0x63,0x68,0x49,0x6e,0x64,0x65,0x78,0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation5valueE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation12launch_indexE[1] = {0x0};
);

PTXModule printfLwda42 = PTX_MODULE( "rg",
//
// Generated by LWPU LWVM Compiler
// Compiler built on Sat Apr 07 16:53:32 2012 (1333839212)
// Lwca compilation tools, release 4.2, V0.2.1221
//

.version 3.0
.target sm_20
.address_size 64

  .file 1 "C:/Users/jbigler/AppData/Local/Temp/tmpxft_00103a74_00000000-11_printf.cpp3.i"
  .file 2 "C:/code/rtsdk/rel3.0/tests/Unit/printf/printf.lw"
.extern .func  (.param .b32 func_retval0) vprintf
(
  .param .b64 vprintf_param_0,
  .param .b64 vprintf_param_1
)
;
.global .align 4 .f32 value;
.global .align 4 .b8 launch_index[4];
.global .align 8 .u64 _ZN21rti_internal_register20reg_bitness_detectorE;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail0E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail1E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail2E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail3E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail4E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail5E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail6E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail7E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail8E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail0E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail1E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail2E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail3E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail4E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail5E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail6E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail7E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail8E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_xE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_yE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_zE;
.global .align 4 .b8 _ZN21rti_internal_typeinfo5valueE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 4 .b8 _ZN21rti_internal_typeinfo12launch_indexE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 1 .b8 _ZN21rti_internal_typename5valueE[6] = {102, 108, 111, 97, 116, 0};
.global .align 1 .b8 _ZN21rti_internal_typename12launch_indexE[6] = {117, 105, 110, 116, 49, 0};
.global .align 1 .b8 _ZN21rti_internal_semantic5valueE[1];
.global .align 1 .b8 _ZN21rti_internal_semantic12launch_indexE[14] = {114, 116, 76, 97, 117, 110, 99, 104, 73, 110, 100, 101, 120, 0};
.global .align 1 .b8 _ZN23rti_internal_annotation5valueE[1];
.global .align 1 .b8 _ZN23rti_internal_annotation12launch_indexE[1];
.global .align 1 .b8 $str[42] = {91, 37, 117, 93, 58, 32, 84, 101, 115, 116, 105, 110, 103, 32, 102, 114, 111, 109, 32, 67, 85, 68, 65, 32, 118, 101, 114, 115, 105, 111, 110, 32, 37, 100, 46, 37, 100, 46, 37, 100, 10, 0};
.global .align 1 .b8 $str1[19] = {91, 37, 117, 93, 58, 32, 9, 118, 97, 108, 117, 101, 32, 61, 32, 37, 102, 10, 0};

.entry _Z2rgv(

)
{
  .local .align 8 .b8   __local_depot0[16];
  .reg .b64   %SP;
  .reg .f32   %f<2>;
  .reg .f64   %fd<2>;
  .reg .s32   %r<8>;
  .reg .s64   %rl<5>;


  mov.u64   %SP, __local_depot0;
  add.u64   %rl1, %SP, 0;
  ldu.global.u32  %r1, [launch_index];
  .loc 2 12 1
  cvta.local.u64  %rl2, %rl1;
  st.local.u32  [%SP+0], %r1;
  mov.u32   %r2, 4;
  .loc 2 12 1
  st.local.u32  [%SP+4], %r2;
  mov.u32   %r3, 2;
  .loc 2 12 1
  st.local.u32  [%SP+8], %r3;
  mov.u32   %r4, 0;
  .loc 2 12 1
  st.local.u32  [%SP+12], %r4;
  mov.u64   %rl3, $str;
  // Callseq Start 0
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  .loc 2 12 1
  st.param.b64  [param0+0], %rl3;
  .param .b64 param1;
  st.param.b64  [param1+0], %rl2;
  .param .b32 retval0;
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r5, [retval0+0];
  }
  // Callseq End 0
  .loc 2 13 1
  ld.global.u32   %r6, [launch_index];
  ld.global.f32   %f1, [value];
  st.local.u32  [%SP+0], %r6;
  cvt.ftz.f64.f32   %fd1, %f1;
  st.local.f64  [%SP+8], %fd1;
  mov.u64   %rl4, $str1;
  // Callseq Start 1
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  .loc 2 13 1
  st.param.b64  [param0+0], %rl4;
  .param .b64 param1;
  st.param.b64  [param1+0], %rl2;
  .param .b32 retval0;
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r7, [retval0+0];
  }
  // Callseq End 1
  .loc 2 14 2
  ret;
}
);

PTXModule printfLwda50 = PTX_MODULE( "rg",
//
// Generated by LWPU LWVM Compiler
// Compiler built on Tue Aug 07 21:33:10 2012 (1344396790)
// Lwca compilation tools, release 5.0, V0.2.1221
//

.version 3.1
.target sm_20
.address_size 64

  .file 1 "C:/Users/jbigler/AppData/Local/Temp/tmpxft_00109d28_00000000-11_printf.cpp3.i"
  .file 2 "C:/code/rtsdk/rel3.0/tests/Unit/printf/printf.lw"
.extern .func  (.param .b32 func_retval0) vprintf
(
  .param .b64 vprintf_param_0,
  .param .b64 vprintf_param_1
)
;
.global .align 4 .f32 value;
.global .align 4 .b8 launch_index[4];
.global .align 8 .u64 _ZN21rti_internal_register20reg_bitness_detectorE;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail0E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail1E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail2E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail3E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail4E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail5E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail6E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail7E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail8E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail0E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail1E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail2E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail3E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail4E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail5E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail6E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail7E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail8E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_xE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_yE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_zE;
.global .align 4 .b8 _ZN21rti_internal_typeinfo5valueE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 4 .b8 _ZN21rti_internal_typeinfo12launch_indexE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 1 .b8 _ZN21rti_internal_typename5valueE[6] = {102, 108, 111, 97, 116, 0};
.global .align 1 .b8 _ZN21rti_internal_typename12launch_indexE[6] = {117, 105, 110, 116, 49, 0};
.global .align 1 .b8 _ZN21rti_internal_semantic5valueE[1];
.global .align 1 .b8 _ZN21rti_internal_semantic12launch_indexE[14] = {114, 116, 76, 97, 117, 110, 99, 104, 73, 110, 100, 101, 120, 0};
.global .align 1 .b8 _ZN23rti_internal_annotation5valueE[1];
.global .align 1 .b8 _ZN23rti_internal_annotation12launch_indexE[1];
.const .align 1 .b8 $str[42] = {91, 37, 117, 93, 58, 32, 84, 101, 115, 116, 105, 110, 103, 32, 102, 114, 111, 109, 32, 67, 85, 68, 65, 32, 118, 101, 114, 115, 105, 111, 110, 32, 37, 100, 46, 37, 100, 46, 37, 100, 10, 0};
.const .align 1 .b8 $str1[19] = {91, 37, 117, 93, 58, 32, 9, 118, 97, 108, 117, 101, 32, 61, 32, 37, 102, 10, 0};

.visible .entry _Z2rgv(

)
{
  .local .align 8 .b8   __local_depot0[16];
  .reg .b64   %SP;
  .reg .b64   %SPL;
  .reg .s32   %r<23>;
  .reg .f32   %f<2>;
  .reg .s64   %rd<5>;
  .reg .f64   %fd<2>;


  mov.u64   %SPL, __local_depot0;
  cvta.local.u64  %SP, %SPL;
  add.u64   %rd1, %SP, 0;
  .loc 2 12 1
  cvta.to.local.u64   %rd2, %rd1;
  ldu.global.u32  %r1, [launch_index];
  mov.u32   %r2, 5;
  .loc 2 12 1
  st.local.v2.u32   [%rd2], {%r1, %r2};
  mov.u32   %r4, 0;
  .loc 2 12 1
  st.local.v2.u32   [%rd2+8], {%r4, %r4};
  cvta.const.u64  %rd3, $str;
  // Callseq Start 0
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  st.param.b64  [param0+0], %rd3;
  .param .b64 param1;
  st.param.b64  [param1+0], %rd1;
  .param .b32 retval0;
  .loc 2 12 1
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r6, [retval0+0];
  }
  // Callseq End 0
  .loc 2 13 1
  ld.global.u32   %r7, [launch_index];
  ld.global.f32   %f1, [value];
  st.local.u32  [%rd2], %r7;
  cvt.ftz.f64.f32   %fd1, %f1;
  st.local.f64  [%rd2+8], %fd1;
  cvta.const.u64  %rd4, $str1;
  // Callseq Start 1
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  st.param.b64  [param0+0], %rd4;
  .param .b64 param1;
  st.param.b64  [param1+0], %rd1;
  .param .b32 retval0;
  .loc 2 13 1
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r10, [retval0+0];
  }
  // Callseq End 1
  .loc 2 14 2
  ret;
}
);

PTXModule printfLwda55 = PTX_MODULE( "rg",
//
// Generated by LWPU LWVM Compiler
// Compiler built on Wed Jul 10 13:41:20 2013 (1373485280)
// Lwca compilation tools, release 5.5, V5.5.0
//

.version 3.2
.target sm_20
.address_size 64

  .file 1 "C:/code/2rtsdk/rtmain/tests/Unit/printf/printf.lw", 1375309167, 411
.extern .func  (.param .b32 func_retval0) vprintf
(
  .param .b64 vprintf_param_0,
  .param .b64 vprintf_param_1
)
;
.global .align 4 .f32 value;
.global .align 4 .b8 launch_index[4];
.global .align 8 .u64 _ZN21rti_internal_register20reg_bitness_detectorE;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail0E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail1E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail2E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail3E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail4E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail5E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail6E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail7E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail8E;
.global .align 8 .u64 _ZN21rti_internal_register24reg_exception_64_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail0E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail1E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail2E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail3E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail4E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail5E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail6E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail7E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail8E;
.global .align 4 .u32 _ZN21rti_internal_register21reg_exception_detail9E;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_xE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_yE;
.global .align 4 .u32 _ZN21rti_internal_register14reg_rayIndex_zE;
.global .align 8 .b8 _ZTVSt14error_category[72];
.global .align 8 .b8 _ZTVSt23_Generic_error_category[72];
.global .align 8 .b8 _ZTVSt24_Iostream_error_category[72];
.global .align 8 .b8 _ZTVSt22_System_error_category[72];
.global .align 4 .b8 _ZN21rti_internal_typeinfo5valueE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 4 .b8 _ZN21rti_internal_typeinfo12launch_indexE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
.global .align 1 .b8 _ZN21rti_internal_typename5valueE[6] = {102, 108, 111, 97, 116, 0};
.global .align 1 .b8 _ZN21rti_internal_typename12launch_indexE[6] = {117, 105, 110, 116, 49, 0};
.global .align 1 .b8 _ZN21rti_internal_semantic5valueE[1];
.global .align 1 .b8 _ZN21rti_internal_semantic12launch_indexE[14] = {114, 116, 76, 97, 117, 110, 99, 104, 73, 110, 100, 101, 120, 0};
.global .align 1 .b8 _ZN23rti_internal_annotation5valueE[1];
.global .align 1 .b8 _ZN23rti_internal_annotation12launch_indexE[1];
.global .align 1 .b8 $str[42] = {91, 37, 117, 93, 58, 32, 84, 101, 115, 116, 105, 110, 103, 32, 102, 114, 111, 109, 32, 67, 85, 68, 65, 32, 118, 101, 114, 115, 105, 111, 110, 32, 37, 100, 46, 37, 100, 46, 37, 100, 10, 0};
.global .align 1 .b8 $str1[19] = {91, 37, 117, 93, 58, 32, 9, 118, 97, 108, 117, 101, 32, 61, 32, 37, 102, 10, 0};

.visible .entry _Z2rgv(

)
{
  .local .align 8 .b8   __local_depot0[16];
  .reg .b64   %SP;
  .reg .b64   %SPL;
  .reg .s32   %r<7>;
  .reg .f32   %f<2>;
  .reg .s64   %rd<5>;
  .reg .f64   %fd<2>;


  mov.u64   %SPL, __local_depot0;
  cvta.local.u64  %SP, %SPL;
  add.u64   %rd1, %SP, 0;
  .loc 1 12 1
  cvta.to.local.u64   %rd2, %rd1;
  ldu.global.u32  %r1, [launch_index];
  mov.u32   %r2, 5;
  .loc 1 12 1
  st.local.v2.u32   [%rd2], {%r1, %r2};
  mov.u32   %r3, 0;
  .loc 1 12 1
  st.local.v2.u32   [%rd2+8], {%r2, %r3};
  cvta.global.u64   %rd3, $str;
  // Callseq Start 0
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  st.param.b64  [param0+0], %rd3;
  .param .b64 param1;
  st.param.b64  [param1+0], %rd1;
  .param .b32 retval0;
  .loc 1 12 1
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r4, [retval0+0];
  }
  // Callseq End 0
  .loc 1 13 1
  ld.global.u32   %r5, [launch_index];
  ld.global.f32   %f1, [value];
  st.local.u32  [%rd2], %r5;
  cvt.ftz.f64.f32 %fd1, %f1;
  st.local.f64  [%rd2+8], %fd1;
  cvta.global.u64   %rd4, $str1;
  // Callseq Start 1
  {
  .reg .b32 temp_param_reg;
  .param .b64 param0;
  st.param.b64  [param0+0], %rd4;
  .param .b64 param1;
  st.param.b64  [param1+0], %rd1;
  .param .b32 retval0;
  .loc 1 13 1
  call.uni (retval0), 
  vprintf, 
  (
  param0, 
  param1
  );
  ld.param.b32  %r6, [retval0+0];
  }
  // Callseq End 1
  .loc 1 14 2
  ret;
}
);
// clang-format on

TEST_F( TestPrintf, PrintfLwda40Works )
{
    std::string result = getOutputStringFromPTX( printfLwda40.code, printfLwda40.functionName );
    EXPECT_EQ( "[0]: Testing from LWCA version 4.0.0\n[0]: \tvalue = 3.141592\n", result );
}

TEST_F( TestPrintf, PrintfLwda42Works )
{
    std::string result = getOutputStringFromPTX( printfLwda42.code, printfLwda42.functionName );
    EXPECT_EQ( "[0]: Testing from LWCA version 4.2.0\n[0]: \tvalue = 3.141592\n", result );
}

TEST_F( TestPrintf, PrintfLwda50Works )
{
    std::string result = getOutputStringFromPTX( printfLwda50.code, printfLwda50.functionName );
    EXPECT_EQ( "[0]: Testing from LWCA version 5.0.0\n[0]: \tvalue = 3.141592\n", result );
}

TEST_F( TestPrintf, PrintfLwda55Works )
{
    std::string result = getOutputStringFromPTX( printfLwda55.code, printfLwda55.functionName );
    EXPECT_EQ( "[0]: Testing from LWCA version 5.5.0\n[0]: \tvalue = 3.141592\n", result );
}
