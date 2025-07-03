/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define BUG_1700459_FIXED 0

// Sizes should be >= 2
#define SIZE_1 2
#define SIZE_2 3
#define SIZE_3 4

#define TEST_VALUE_1 9
#define TEST_VALUE_2 7

#define UBO_BINDING_0 0
#define UBO_BINDING_1 1
#define UBO_BINDING_2 2

using namespace lwn;

class LWNGlslTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNGlslTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Miscellaneous testing for various coverage of GLSL built-ins and extensions which might not "
          "be exercised in other tests.\n";
    return sb.str();
}

int LWNGlslTest::isSupported() const
{
    // The LW_shader_thread_group and ARB_shader_ballot built-in uniforms will fail up until
    // GLSLC package version 34.
    return (lwogCheckLWNGLSLCPackageVersion(34) &&
            lwogCheckLWNAPIVersion(53, 0));
}

// Utility function to get the SM count, warp size (number of threads per warp), and number of warps per SM
static void GetThreadParams(Device *device, int *smCount, int *warpSize, int *warpsPerSM)
{
    // Number of SMs (same as minimum scaling factor).
    device->GetInteger(lwn::DeviceInfo::SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM, smCount);

    // Number of warps per SM.  Derive this by taking recommended (warps-per-SM * num_sms) and dividing by the
    // minimum scale factor (num_sms) , which leaves warps-per-SM.
    int recommendScaleFactor = 0;
    device->GetInteger(lwn::DeviceInfo::SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED,
                       &recommendScaleFactor);
    *warpsPerSM = recommendScaleFactor / *smCount;

    // All lwrrently-supported hardware uses 32 threads per warp.
    *warpSize = 32;

}

// A test that covers testing built-in uniforms in the LW_shader_thread_group.
static void TestShaderThreadGroupUniforms(Device *device, QueueCommandBuffer *queueCB, Queue *queue)
{
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "}\n";

    // A fragment shader which checks that the HW parameters from LWN match the built-ins.
    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::LW_shader_thread_group);
    fs.addExtension(lwShaderExtension::ARB_shader_ballot);
    fs <<
        "layout(binding = 0) uniform block {\n"
        "  int expectedWarpSize;\n"
        "  int expectedWarpsPerSM;\n"
        "  int expectedSMCount;\n"
        "};\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  if (\n"
        "         (gl_WarpSizeLW      != (expectedWarpSize)) \n"
        "      || (gl_SubGroupSizeARB != expectedWarpSize) \n"
        "      || (gl_WarpsPerSMLW    != expectedWarpsPerSM) \n"
        "      || (gl_SMCountLW       != expectedSMCount)) {\n"

        // FAIL
        "    fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    } else {\n"
        // PASS
        "    fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        printf("Error compiling shaders:\n%s\n", g_glslcHelper->GetInfoLog());
        queueCB->ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };

    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };

    // Uniform block
    struct UniformBlock {
        int expectedWarpSize;
        int expectedWarpsPerSM;
        int expectedSMCount;
    };

    // Set up an allocator for our UBO and vertex data.
    LWNsizeiptr uboSize = 1024;
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData) + uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    BufferAlignBits uboAlignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT |
                                                   BUFFER_ALIGN_SHADER_STORAGE_BIT |
                                                   BUFFER_ALIGN_COPY_READ_BIT);
    Buffer *uboBuffer = allocator.allocBuffer(&bb, uboAlignBits, sizeof(UniformBlock));
    BufferAddress uboBufferAddr = uboBuffer->GetAddress();
    UniformBlock *uboBufferMap = (UniformBlock *) uboBuffer->Map();

    // Set the UBO parameters (number of SMs, warps per SM, threads per warp).
    GetThreadParams(device, &uboBufferMap->expectedSMCount, &uboBufferMap->expectedWarpSize, &uboBufferMap->expectedWarpsPerSM);

    queueCB->BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboBufferAddr, sizeof(UniformBlock));

    // Vertex attributes
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 6, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB->BindVertexArrayState(vertex);

    queueCB->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB->DrawArrays(DrawPrimitive::TRIANGLES, 0, 6);

    queueCB->submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    queue->Finish();
}

// Compute version to teset built-in thread uniforms
// Compute version of TestShaderThreadGroupUniforms.  We just write either green or red in the shader to
// a single entry vec4 SSBO and clear screen to that color.
static void TestShaderThreadGroupUniformsCompute(Device *device, QueueCommandBuffer *queueCB, Queue *queue)
{
    ComputeShader cs(450);
    cs.addExtension(lwShaderExtension::LW_shader_thread_group);
    cs.addExtension(lwShaderExtension::ARB_shader_ballot);
    cs.setCSGroupSize(1, 1);
    cs <<
        "layout(std430, binding = 0) buffer SSBO {\n"
        "  vec4 values[1];\n"
        "};\n"
        "layout(binding = 0) uniform block {\n"
        "  int expectedWarpSize;\n"
        "  int expectedWarpsPerSM;\n"
        "  int expectedSMCount;\n"
        "};\n"
        "void main() {\n"
        "  vec4 computedColor = vec4(0, 0, 0, 1);\n"
        "  if (\n"
        "         (gl_WarpSizeLW      != expectedWarpSize) \n"
        "      || (gl_SubGroupSizeARB != expectedWarpSize) \n"
        "      || (gl_WarpsPerSMLW    != expectedWarpsPerSM) \n"
        "      || (gl_SMCountLW       != expectedSMCount)) {\n"
        // FAIL
        "           computedColor = vec4(1, 0, 0, 1);\n"
        // PASS
        "  } else { computedColor = vec4(0, 1, 0, 1); }\n"
        "  values[0] = computedColor;\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, cs)) {
        printf("Error compiling shaders:\n%s\n", g_glslcHelper->GetInfoLog());
        queueCB->ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        return;
    }

    // Uniform block
    struct UniformBlock {
        int expectedWarpSize;
        int expectedWarpsPerSM;
        int expectedSMCount;
    };

    // This test allocates one large pool of "dataPoolSize" used for vertex attributes and the uniform block.
    LWNsizeiptr uboSize = 512 * 1024;
    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // UBO
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    BufferAlignBits uboAlignBits = BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT |
                                                   BUFFER_ALIGN_SHADER_STORAGE_BIT |
                                                   BUFFER_ALIGN_COPY_READ_BIT);
    Buffer *uboBuffer = allocator.allocBuffer(&bb, uboAlignBits, sizeof(UniformBlock));
    BufferAddress uboBufferAddr = uboBuffer->GetAddress();
    UniformBlock *uboBufferMap = (UniformBlock *) uboBuffer->Map();

    // Set the UBO parameters (number of SMs, warps per SM, threads per warp).
    GetThreadParams(device, &uboBufferMap->expectedSMCount, &uboBufferMap->expectedWarpSize, &uboBufferMap->expectedWarpsPerSM);

    // Set up a single entry SSBO
    int ssboSize = sizeof(dt::vec4);
    ssboSize = (ssboSize + 0x1F) & ~(0x1F);
    MemoryPoolAllocator ssboAllocator(device, NULL, ssboSize, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
    Buffer *ssbo = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();
    float * ssboMem = (float *)(ssbo->Map());

    queueCB->BindProgram(pgm, ShaderStageBits::COMPUTE);
    queueCB->BindStorageBuffer(ShaderStage::COMPUTE, 0, ssboAddr, ssboSize);
    queueCB->BindUniformBuffer(ShaderStage::COMPUTE, 0, uboBufferAddr, sizeof(UniformBlock));

    // Launch a single thread compute shader.  We just need to test that reading the uniforms works.
    queueCB->DispatchCompute(1, 1, 1);
    queueCB->submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    queue->Finish();
    ssbo->IlwalidateMappedRange(0, ssboSize);

    queueCB->ClearColor(0, ssboMem[0], ssboMem[1], ssboMem[2], ssboMem[3]);

    queueCB->submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    queue->Finish();
}

// Tests that std430 qualifier can be used with UBOs in LWN.  This is an LWN-only feature; in OpenGL,
// std430 is only allowed on SSBOs.
// We create a vertex shader with a UBO containing some entries whose stride/offsets will depend
// on std430 layouts working.  We test both program interface queries that the offsets/strides
// conform to std430 rules, and we also render from a UBO using std430 rules to lay out the data
// in the buffer.
//
// For details on the std430 layout rules, and how they differ from std140,
// see https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf,
// specifically section 7.6.2.2.
static void TestStd430UBOSupport(Device *device, QueueCommandBuffer *queueCB, Queue *queue)
{
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=0) out vec4 ocolor;\n"
        "layout(std430, binding = 0) uniform Block_430 {\n"
        "    vec2 testArry[3];\n" // offset: 0
                                  // stride: 8 bytes
                                  // alignment: 8 bytes

        "    int testInt;\n"      // offset: 24
                                  // alignment: 4

        "    vec3 testVec;\n"     // offset: 32
                                  // alignment: 16
        "};\n"

        // We will check "testInt"'s value is 1, and testVec.x value is 2.0f.
        // The output color in case of passing checks will come from
        // testArray[1].rg (which will be 0.0f, 1.0f).  In case of failing, will
        // draw red.
        "void main () {\n"
        "   gl_Position = vec4(position, 1.0);\n"
        "   if (testInt == 1 && testVec.x == 2.0f) {\n"
                // testArry[1].rgr should be (0.0f, 1.0f, 0.0f),
                // in other words green.
        "       ocolor = vec4(testArry[1].rgr, 1.0f);\n"
        "   } else {\n"
                // Failure
        "       ocolor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
        "   }\n"
        "}\n";

    // The fragment shader just passes through the color.
    FragmentShader fs(440);
    fs <<
        "layout(location = 0) in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "   fcolor = ocolor;\n"
        "};\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        printf("Error compiling.  Infolog:\n%s\n", g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return;
    }

    // Check the program interface query reports the correct offsets/strides for entries, as expected under std430.
    int testArryOffset = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET, "testArry[0]");
    int testArryStride = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_ARRAY_STRIDE, "testArry[0]");
    int testIntOffset = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET, "testInt");
    int testVecOffset = g_glslcHelper->ProgramGetResourceLocation(pgm, ShaderStage::VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET, "testVec");

    if (testArryOffset != 0) {
        printf("Unexpected offset %d for \"testArry\".\n", testArryOffset);
        LWNFailTest();
        return;
    }

    if (testArryStride != 8) {
        printf("Unexpected stride %d for \"testArry\".\n", testArryStride);
        LWNFailTest();
        return;
    }

    if (testIntOffset != 24) {
        printf("Unexpected offset %d for \"testInt\".\n", testIntOffset);
        LWNFailTest();
        return;
    }

    if (testVecOffset != 32) {
        printf("Unexpected offset %d for \"testVec\".\n", testVecOffset);
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) }
    };

    MemoryPoolAllocator vboAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // VBO
    VertexStream vStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vStream, Vertex, position);
    VertexArrayState testVertex = vStream.CreateVertexArrayState();
    Buffer *testVBO = vStream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress testVBOAddr = testVBO->GetAddress();

    // Build UBO buffer and copy contents of memory over.
    int sizeOfBuf = 1024;
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator(device, NULL, sizeOfBuf, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, sizeOfBuf);
    char * uboMem = (char *) ubo->Map();
    BufferAddress uboAddr = ubo->GetAddress();

    // Set the UBO data.

    memset(uboMem, 0, sizeOfBuf);

    // We set testArry[1].y to "1" here.  testArry[1] contains the "r, g"
    // components of the passing output color. (i.e. (0, 1) for green).
    // The rest of testArry entries should all be left unchanged.
    ((float *)(uboMem + testArryOffset + testArryStride))[1] = 1.0f;

    // testInt -- Shader expects 1.
    ((int *)(uboMem + testIntOffset))[0] = 1;

    // testVec -- shader expects .x component to be 2.0f.
    ((float *)(uboMem + testVecOffset))[0] = 2.0f;

    // Bind our UBO to the vertex binding 0 slot.
    queueCB->BindUniformBuffer(ShaderStage::VERTEX, UBO_BINDING_0, uboAddr, sizeOfBuf);

    queueCB->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB->BindVertexArrayState(testVertex);

    queueCB->BindVertexBuffer(0, testVBOAddr, sizeof(vertexData));
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

    // We need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory.
    queueCB->submit();
    queue->Finish();
}

static void TestLwllDistance(Device *device, QueueCommandBuffer *queueCB, Queue *queue)
{
    VertexShader vs(440);
    vs.addExtension(lwShaderExtension::ARB_lwll_distance);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "float gl_LwllDistance[1];\n"
        "out vec4 ocolor;\n"
        "void main () {\n"
        // Creates an ilwerted lwll plane where only points farther than
        // 0.8 in the z direction will be drawn.
        "    gl_LwllDistance[0] = position.z - 0.8;\n"
        // Set color based on distance.
        "    if (position.z > 0.8) { \n"
        "        ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "    } else { \n"
        "        ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    }\n"
        "    gl_Position = vec4(position, 1.0);\n"
        "};\n";

    FragmentShader fs(440);
    fs <<
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "   fcolor = ocolor;\n"
        "};\n";

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        queueCB->ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, +0.9) },
        { dt::vec3(-1.0, +1.0, +0.9) },
        { dt::vec3(+1.0, +1.0, +0.9) },
        { dt::vec3(+1.0, -1.0, +0.9) },

        { dt::vec3(-1.0, -1.0, +0.7) },
        { dt::vec3(-1.0, +1.0, +0.7) },
        { dt::vec3(+1.0, +1.0, +0.7) },
        { dt::vec3(+1.0, -1.0, +0.7) },
    };

    // Allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 8, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB->ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    queueCB->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB->BindVertexArrayState(vertex);
    queueCB->BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Fill the screen with green.
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    // If the lwll plane is not working, fill the screen with red.
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_FAN, 4, 4);

    // We need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory.
    queueCB->submit();
    queue->Finish();
}

/*
Test that extension GL_ARB_arrays_of_arrays is working properly.
Lwrrently using the "length" function and arrays values to test
multi dimensional arrays are working in the following scenarios:
 - creating a UBO containing a multi dimensional array
 - creating a local multi dimensional array
 - passing a multi dimensional array from one shader to another.
When Bug 1700459 is fixed, the following can also be enabled by
uncommenting the code:
 - Creating a multi dimensional array of UBOs
*/
static void TestArraysOfArrays(Device *device, QueueCommandBuffer *queueCB, Queue *queue)
{
    VertexShader vs(440);
    vs.addExtension(lwShaderExtension::ARB_arrays_of_arrays);
    vs <<
        "layout(location=0) in vec3 position;\n"
#if (BUG_1700459_FIXED)
        "layout(binding=2, std140 ) uniform ubo2 {\n"
        "   vec4 singleArr;\n"
        "} MultiDimUBO [" << SIZE_1 << "][" << SIZE_2 << "];\n"
#endif
        "layout(binding=1, std140 ) uniform ubo {\n"
        "   vec4 [" << SIZE_1 << "][" << SIZE_2 << "] MultiDimArr;\n"
        "};\n"
        "vec4 localMultiDimArr[" << SIZE_1 << "][" << SIZE_2 << "][" << SIZE_3 << "];\n"
        "out vec4 crossShaderMultiDimArr [" << SIZE_1 << "][" << SIZE_2 << "];\n"
        "out vec4 result;\n"
        "void main () {\n"
        "    gl_Position = vec4(position, 1.0);\n"
        "    if (\n"
        "           (MultiDimArr.length() == " << SIZE_1 << ")\n"
        "        && (MultiDimArr[" << SIZE_1 - 1 << "].length() == " << SIZE_2 << ")\n"
        "        && (MultiDimArr[" << SIZE_1 - 1 << "][" << SIZE_2 - 1 << "][3] == float(" << TEST_VALUE_1 << "))\n"
        "        && (localMultiDimArr.length() == " << SIZE_1 << ")\n"
        "        && (localMultiDimArr[" << SIZE_1 - 1 << "].length() == " << SIZE_2 << ")\n"
        "        && (localMultiDimArr[" << SIZE_1 - 1 << "][" << SIZE_2 - 1 << "].length() == " << SIZE_3 << ") ) {\n"
#if (BUG_1700459_FIXED)
        "        && (MultiDimUBO.length() == float(" << SIZE_1 << "))\n"
        "        && (MultiDimUBO[1].length() == float(" << SIZE_2 << "))\n"
        "        && (MultiDimUBO[1][1].singleVec[3] == float(" << TEST_VALUE_2 << ")) ) {\n"
#endif
        // PASS
        "           result = vec4(0, 1, 0, 1); }\n"
        // FAIL
        "    else { result = vec4(1, 0, 0, 1); }\n"
        "    crossShaderMultiDimArr[0][1][3] = float(" << TEST_VALUE_1 << ");\n"
        "    crossShaderMultiDimArr[1][0][2] = float(" << TEST_VALUE_2 << ");\n"
        "}\n";

    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::ARB_arrays_of_arrays);
    fs <<
        "in vec4 result;\n"
        "in vec4 crossShaderMultiDimArr [" << SIZE_1 << "][" << SIZE_2 << "];\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "    if (\n"
        "           (crossShaderMultiDimArr.length() == " << SIZE_1 << ")\n"
        "        && (crossShaderMultiDimArr[1].length() == " << SIZE_2 << ")\n"
        "        && (crossShaderMultiDimArr[0][1][3] == float(" << TEST_VALUE_1 << "))\n"
        "        && (crossShaderMultiDimArr[1][0][2] == float(" << TEST_VALUE_2 << "))\n"
        "        && (result == vec4(0, 1, 0, 1)) ) {\n"
        // PASS
        "           fcolor = vec4(0, 1, 0, 1); }\n"
        // FAIL
        "    else { fcolor = vec4(1, 0, 0, 1); }\n"
        "}\n";

    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        printf("Error compiling shaders:\n%s\n", g_glslcHelper->GetInfoLog());
        queueCB->ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        return;
    }

    // Create multidimensional array to pass into buffer.
    struct testArray {
        dt::vec4 subArray[SIZE_1][SIZE_2];
    };

    static const testArray multiDimArray = {
        { { dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2),
            dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2),
            dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2) },
          { dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2),
            dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2),
            dt::vec4(TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_2, TEST_VALUE_1) } }
    };

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator(device, NULL, sizeof(multiDimArray), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, sizeof(multiDimArray));
    char * uboMem = (char *) ubo->Map();

    memcpy(uboMem, (void*)(&multiDimArray), sizeof(multiDimArray));

    BufferAddress uboAddr = ubo->GetAddress();
    queueCB->BindUniformBuffer(ShaderStage::VERTEX, UBO_BINDING_1, uboAddr, sizeof(multiDimArray));

#if (BUG_1700459_FIXED)
    struct vecStruct {
        dt::vec4 singleVec;
    };

    static const vecStruct vecArray [SIZE_1][SIZE_2] = {
      { {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1)},
        {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1)},
        {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1)}  },
      { {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1)},
        {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_2)},
        {dt::vec4(TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1, TEST_VALUE_1)}  }
    };

    // Create 6 buffers to bind.
    BufferBuilder bb0;
    bb0.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator0(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo0 = uboAllocator0.allocBuffer(&bb0, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem0 = (char *) ubo0->Map();
    memcpy(uboMem0, (void*)(&vecArray[0][0]), sizeof(vecStruct));

    BufferBuilder bb1;
    bb1.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator1(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo1 = uboAllocator1.allocBuffer(&bb1, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem1 = (char *) ubo1->Map();
    memcpy(uboMem1, (void*)(&vecArray[0][1]), sizeof(vecStruct));

    BufferBuilder bb2;
    bb2.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator2(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo2 = uboAllocator2.allocBuffer(&bb2, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem2 = (char *) ubo2->Map();
    memcpy(uboMem2, (void*)(&vecArray[0][2]), sizeof(vecStruct));

    BufferBuilder bb3;
    bb3.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator3(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo3 = uboAllocator3.allocBuffer(&bb3, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem3 = (char *) ubo3->Map();
    memcpy(uboMem3, (void*)(&vecArray[1][0]), sizeof(vecStruct));

    BufferBuilder bb4;
    bb4.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator4(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo4 = uboAllocator4.allocBuffer(&bb4, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem4 = (char *) ubo4->Map();
    memcpy(uboMem4, (void*)(&vecArray[1][1]), sizeof(vecStruct));

    BufferBuilder bb5;
    bb5.SetDevice(device).SetDefaults();
    MemoryPoolAllocator uboAllocator5(device, NULL, sizeof(vecStruct), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo5 = uboAllocator5.allocBuffer(&bb5, BUFFER_ALIGN_UNIFORM_BIT, sizeof(vecStruct));
    char * uboMem5 = (char *) ubo5->Map();
    memcpy(uboMem5, (void*)(&vecArray[1][2]), sizeof(vecStruct));

    // Bind 6 buffers to create a 2 by 3 array of UBOs.
    BufferRange uboRange[6] = {
        {ubo0->GetAddress(), ubo0->GetSize()},
        {ubo1->GetAddress(), ubo1->GetSize()},
        {ubo2->GetAddress(), ubo2->GetSize()},
        {ubo3->GetAddress(), ubo3->GetSize()},
        {ubo4->GetAddress(), ubo4->GetSize()},
        {ubo5->GetAddress(), ubo5->GetSize()}
    };

    queueCB->BindUniformBuffers(ShaderStage::VERTEX, UBO_BINDING_2, 6, uboRange);
#endif

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };

    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) }
    };

    MemoryPoolAllocator vboAllocator(device, NULL, 64 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vStream, Vertex, position);
    VertexArrayState testVertex = vStream.CreateVertexArrayState();
    Buffer *testVBO = vStream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress testVBOAddr = testVBO->GetAddress();

    queueCB->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB->BindVertexArrayState(testVertex);

    queueCB->BindVertexBuffer(0, testVBOAddr, sizeof(vertexData));
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

    queueCB->submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    queue->Finish();
}

typedef void(*TestFunction_T)(Device *, QueueCommandBuffer *, Queue *);

// List of tests.  The entries start in the lower left corner, and proceed to the right and
// wrap around.
// Tests are responsible for submitting all work to the GPU.
TestFunction_T testFunctions[] =
{
    TestShaderThreadGroupUniforms,
    TestShaderThreadGroupUniformsCompute,
    TestLwllDistance,
    TestArraysOfArrays,
    TestStd430UBOSupport,
};

void LWNGlslTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer *queueCB = &deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB->ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    int maxWidthHeight = 800;
    int widthHeight = 40;
    float cellScale = 0.9f;

    int row = 0;
    int col = 0;

    for (int i = 0; i < (int)__GL_ARRAYSIZE(testFunctions); ++i) {
        queueCB->SetViewportScissor(col, row, (int)(widthHeight*cellScale), (int)(widthHeight*cellScale));
        testFunctions[i](device, queueCB, queue);
        col += widthHeight;
        if (col > maxWidthHeight) {
            col = 0;
            row += widthHeight;
        }
    }
}

OGTEST_CppTest(LWNGlslTest, lwn_glsl_builtins, );
