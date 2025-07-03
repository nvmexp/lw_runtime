/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "cmdline.h"
#include <array>
#include <string>

using namespace lwn;

class LWNLowpMediumpMixedTest
{

    public:
        static const int total_tests = 32;
        static const int cellsX = 8;
        static const int cellsY = 6;

        struct Vertex {
        dt::vec2 position;
        };

        LWNTEST_CppMethods();
};

lwString LWNLowpMediumpMixedTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test for lowp and mediump precision floats used in mixed modes."
           "\nAny red square means a compilation failure."
           "\nAny color other than green means behavior has changed or shader compilation has failed."
           "\nGreen = good, Red = a failed test case."
           "\nThe resulting image should have 4 rows of 8 green squares.";
    return sb.str();
}

int LWNLowpMediumpMixedTest::isSupported() const
{
    GLSLCversion dllVersion = g_glslcLibraryHelper->GetVersion();
    if (dllVersion.gpuCodeVersionMajor < 1 || dllVersion.gpuCodeVersionMinor < 2) {
        return 0;
    }

#if defined(SPIRV_ENABLED)
    // We use a custom LWN-specific extension LW_desktop_lowp_mediump in this test.
    // This extension is not understood by the Khronos glslang GLSL->SPIR-V compiler,
    // so some tests relying on this feature might fail.
    if (useSpirv) {
        return 0;
    }
#endif

    return lwogCheckLWNAPIVersion(5, 0);
}

//The tests begin here...
void LWNLowpMediumpMixedTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    VertexShader vs(450);
    vs.addExtension(lwShaderExtension::LW_desktop_lowp_mediump);
    vs <<
        "layout(location=0) in vec3 position;\
        \nlayout(location=1) in vec3 color;\
        \nout vec3 ocolor;\
        \lwoid main()\
        {\n    gl_Position = vec4(position, 1.0);\
        \n  ocolor = color;\n}";


    // Input values to be used in tests
    float float_values1[] = { 1023.0f, 1023.0f, 1023.0f, 1023.0f, -1023.0f, -1023.0f, 257.0f, 257.0f, -257.0f, -257.0f, 16376.25f, 16376.25, 16376.25f, 16376.25,\
                              16376.25, 16376.25f, 64.0f, 64.0f, 0.5235988f, 1.0471976f, 0.7853982f, 1.0f, 0.5235988f, 1.0471976f, 0.7853982f, 1.0f, 2.22567f, 2.22567f,\
                              2.22567f, 2.22567f, 2.22567f, 2.22567f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float float_values2[] = { 513.0f, 513.0f, 513.0f, 513.0f, -513.0f, -513.0f, 7.0f, 7.0f, -7.0f, -7.0f, 2.25f, 2.25f, 2.25f, 2.25f,\
                              2.25f, 2.25f, 57.00625f, 57.00625f, 60.0f, 60.0f, 0.0f, 0.0f, 60.0f, 60.0f, 0.0f, 0.0f, 2.1567f, 2.1567f,\
                              2.1567f, 2.1567f, 2.1567f, 2.1567f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float float_values3[] = { 1536.0f, 1536.0f, 510.0f, 510.0f, -510.0f, -510.0f, 1799.0f, 1799.0f, 1799.0f, 1799.0f, 57.95f, 57.95f, 57.95f, 57.95f,\
                              57.95f, 57.95f, 6500.4f, 6500.4f, 45.0f, 45.0f, 0.0f, 0.0f, 60.0f, 60.0f, 0.0f, 0.0f, 57.69431f, 57.69431f,\
                              57.69431f, 57.69431f, 57.69431f, 57.69431f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float expectedResults[] = { 1536.0f, 1536.0f, 510.0f, 510.0f, -510.0f, -510.0f, 1799.0f, 1799.0f, 1799.0f, 1799.0f, 36904.5125f, 36904.5125f, 36904.5125f, 36904.5125f,\
                              36904.5125f, 36904.5125f, 131008.4f, 131008.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 57.69431f, 57.69431f,\
                              57.69431f, 57.69431f, 57.69431f, 2.1567f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Alternate values for expected results, when computations are carried
    // out or results are stored at FP16 precision.  On the Windows reference
    // implementation, some hardware supports lowp/mediump natively, while
    // most pre-Volta hardware (except GP100) does not.
    float altExpectedResults[] = { 1536.0f, 1536.0f, 510.0f, 510.0f, -510.0f, -510.0f, 1799.0f, 1799.0f, 1799.0f, 1799.0f, 36896.0f, 36896.0f, 36896.0f, 36896.0f,\
                                 36896.0f, 36896.0f, 131008.4f, 131008.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 57.69431f, 57.69431f,\
                                 57.69431f, 57.69431f, 57.69431f, 2.1567f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,\
                                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    std::array<std::string, total_tests> shader_content =  {
        {
        //Test 1: Add 2 positive low precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 + input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n} ",

        //Test 2: Add 2 positive medium precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 + input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 3: Subtract 2 positive low precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 - input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 4: Subtract 2 positive medium precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 - input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 5: Subtract 2 negative low precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 - input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 6: Subtract 2 negative medium precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 - input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 7: Multiply 2 positive low precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 * input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 8: Multiply 2 positive medium precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 * input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 9: Multiply 2 negative low precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 * input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 10: Multiply 2 negative medium precision floats and assign to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = input_value_1 * input_value_2;\
        \n   if ( result == expectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 11: Fused Multiply and Add with all low precision values assigned to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * input_value_2) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 12: Fused Multiply and Add with all medium precision values assigned to high precision float result
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * input_value_2) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 13: Fused Multiply and Add with all low precision values with immediates
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     lowp float expectedResult;\
        \n     lowp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * 2.25f) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 14: Fused Multiply and Add with all medium precision values with immediates
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * 2.25f) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 15: Fused Multiply and Add with low precision and high precision values
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * input_value_2) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 16: Fused Multiply and Add with medium precision and high precision values
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_1 * input_value_2) + input_value_3;\
        \n   if ( result == expectedResult || result == altExpectedResult )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 17: Fused Multiply and Add where high precision RHS is assigned to low precision LHS
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n lowp float lowp_float;\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_3 / input_value_1) - input_value_2;\
        \n   lowp_float = result;\
        \n   if ( lowp_float == 44.5625f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 18: Fused Multiply and Add where high precision RHS is assigned to a medium precision LHS
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n mediump float mediump_float;\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = (input_value_3 / input_value_1) - input_value_2;\
        \n   mediump_float = result;\
        \n   if ( mediump_float == 44.5625f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 19: low precision sine
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = sin(input_value_1);\
        \n   if ( abs(0.5f - result) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 20: low precision cosine
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = cos(input_value_1);\
        \n   if ( abs(0.5f - result) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 21: low precision tan
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = tan(input_value_1) + 3.0f;\
        \n   if ( abs(result - 4.0f) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 22: low precision exp
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     lowp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = exp(input_value_1);\
        \n   if ( abs(result - 2.71828182846f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 23: medium precision sine
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = sin(input_value_1);\
        \n   if ( abs(0.5f - result) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 24: medium precision cosine
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = cos(input_value_1);\
        \n   if ( abs(0.5f - result) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 25: medium precision tan
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = tan(input_value_1) + 3.0f;\
        \n   if ( abs(result - 4.0f) <= 0.0000001f )\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 26: medium precision exp
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = exp(input_value_1);\
        \n   if ( abs(result - 2.71828182846f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 27: clamp using low precision, medium precision, medium precision
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     mediump float input_value_2;\
        \n     mediump float input_value_3;\
        \n     mediump float expectedResult;\
        \n     mediump float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 28: clamp using low precision, medium precision, high precision
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     mediump float input_value_2;\
        \n     highp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 29: clamp using low precision, high precision, high precision
        "layout(binding = 0) uniform Block {\
        \n     lowp float input_value_1;\
        \n     highp float input_value_2;\
        \n     highp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 30: clamp using medium precision, medium precision, high precision
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     mediump float input_value_2;\
        \n     highp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 31: clamp using medium precision, highp, high precision
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     highp float input_value_2;\
        \n     highp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",

        //Test 32: clamp using medium precision, high precision, low precision
        "layout(binding = 0) uniform Block {\
        \n     mediump float input_value_1;\
        \n     highp float input_value_2;\
        \n     lowp float input_value_3;\
        \n     highp float expectedResult;\
        \n     highp float altExpectedResult;\
        \n  };\
        \n highp float result;\
        \n out vec4 fcolor;\
        \n void main()\
        \n {\
        \n   result = clamp(input_value_1, input_value_2, input_value_3);\
        \n   if ( abs(result - 2.22567f) <= 0.0000001f)\
        \n          //pass\
        \n          fcolor = vec4(0.0, 1.0, 0.0, 1.0);\
        \n   else\
        \n          //fail\
        \n          fcolor = vec4(1.0, 0.0, 0.0, 1.0);\
        \n}",
        }
    };

    int vboSize = 1024*1024;

    int vpWidth = lwrrentWindowWidth / cellsX;
    int vpHeight = lwrrentWindowHeight / cellsY;
    int row = 0;
    int col = 0;

    MemoryPoolAllocator allocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *vbo = allocator.allocBuffer(&bb, BufferAlignBits(BUFFER_ALIGN_VERTEX_BIT), vboSize);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);

    Vertex *vboMap = (Vertex *) vbo->Map();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    typedef struct {
                float input_value_1;
                float input_value_2;
                float input_value_3;
                float expectedResult;
                float altExpectedResult;
    } UniformBlock;

    static LWNsizeiptr coherentPoolSize = 0x100000UL; // 1MB pool size
    MemoryPoolAllocator coherent_allocator(device, NULL, coherentPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    UniformBlock blockData;

    int vnum = 0;

    //Run the tests
    for (int test_num = 0; test_num <= total_tests - 1; test_num++) {

        Program *pgm = device->CreateProgram();

        blockData.input_value_1 = float_values1[test_num];
        blockData.input_value_2 = float_values2[test_num];
        blockData.input_value_3 = float_values3[test_num];
        blockData.expectedResult = expectedResults[test_num];
        blockData.altExpectedResult = altExpectedResults[test_num];

        Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &blockData, sizeof(blockData), BUFFER_ALIGN_UNIFORM_BIT, false);
        BufferAddress uboAddr = ubo->GetAddress();
        queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(blockData));

        FragmentShader fs_x(450);
        fs_x.addExtension(lwShaderExtension::LW_desktop_lowp_mediump);
        fs_x <<
            shader_content[test_num].c_str();

        if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs_x)) {
            printf("\nTest %d : Include Shader ERROR:\n", test_num);
            printf("Infolog: %s\n", g_glslcHelper->GetInfoLog());

            // replace with known good shader
            FragmentShader fs_x(450);
            fs_x <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "void main() {\n"
                "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                "}\n";
            if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs_x)) {
                printf("Infolog: %s\n", g_glslcHelper->GetInfoLog());
            }
        }

        queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

        row = test_num / cellsX;
        col = test_num % cellsX;

        queueCB.SetViewportScissor(col * vpWidth + 2, row * vpHeight + 2, vpWidth - 4, vpHeight - 4);

        for (int v = 0; v < 4; ++v) {
            vboMap[vnum + v].position[0] = (v & 2) ? +1.0 : -1.0;
            vboMap[vnum + v].position[1] = (v & 1) ? +1.0 : -1.0;
        }

        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, vnum, 4);
        vnum += 4;

    }

    queueCB.submit();

    //Tests finished

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

}

OGTEST_CppTest(LWNLowpMediumpMixedTest, lwn_lowp_mediump_mixed, );
