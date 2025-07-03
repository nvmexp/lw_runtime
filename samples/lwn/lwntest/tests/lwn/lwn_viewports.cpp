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

#define DEBUG_MODE 0
#if DEBUG_MODE
#define DEBUG_PRINT(x) printf x
#else
#define DEBUG_PRINT(x)
#endif

#define FBO_SIZE 128

using namespace lwn;
using namespace lwn::dt;

namespace {

class Test {
public:
    Test(DeviceState *testDevice, WindowOriginMode windowOriginMode, DepthMode depthMode);
    ~Test();
    bool run();

private:
    void render();
    vec4 calcExpected(int x, int y);
    bool verify();

    WindowOriginMode mWindowOriginMode;
    DepthMode mDepthMode;
    Device *mDevice;
    Queue *mQueue;
    QueueCommandBuffer &mQueueCB;

    Framebuffer mFbo;
};

Test::Test(DeviceState *testDevice, WindowOriginMode windowOriginMode, DepthMode depthMode) :
    mWindowOriginMode(windowOriginMode),
    mDepthMode(depthMode),
    mDevice(testDevice->getDevice()),
    mQueue(testDevice->getQueue()),
    mQueueCB(testDevice->getQueueCB())
{
    mFbo.setSize(FBO_SIZE, FBO_SIZE);
    mFbo.setColorFormat(0, Format::RGBA32F);
    mFbo.alloc(mDevice);
    mFbo.bind(mQueueCB);
}

Test::~Test()
{
    mFbo.destroy();
}

void Test::render()
{
    mQueueCB.SetViewport(0, 0, FBO_SIZE, FBO_SIZE);
    mQueueCB.SetScissor(0, 0, FBO_SIZE, FBO_SIZE);
    mQueueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    VertexShader vs(440);
    vs <<
        "// All the work is being done in the geometry shader using known constant values, so\n"
        "// the vertex shader doesn't actually need to do anything.\n"
        "void main() {\n"
        "}\n";
    GeometryShader gs(440);
    gs <<
        "layout(points, ilwocations=16) in;\n"
        "layout(triangle_strip, max_vertices=4) out;\n"
        "// gInfo.xy identifies the cell for the test; gInfo.z is basically the normalized x\n"
        "// coordinate.\n"
        "out vec3 gInfo;\n"
        "void main() {\n"
        "  gl_ViewportIndex = gl_IlwocationID;\n"
        "  int row = gl_IlwocationID / 4;\n"
        "  int col = gl_IlwocationID % 4;\n"
        "  gInfo = vec3(0.25 * col, 0.25 * row, -1.0);\n"
        "  gl_Position = vec4(-1.0, -1.0, -2.0, 1.0);\n"
        "  EmitVertex();\n"
        "  gInfo.z = 1.0;\n"
        "  gl_Position = vec4( 1.0, -1.0, -2.0, 1.0);\n"
        "  EmitVertex();\n"
        "  gInfo.z = -1.0;\n"
        "  gl_Position = vec4(-1.0,  1.0,  2.0, 1.0);\n"
        "  EmitVertex();\n"
        "  gInfo.z = 1.0;\n"
        "  gl_Position = vec4( 1.0,  1.0,  2.0, 1.0);\n"
        "  EmitVertex();\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 gInfo;\n"
        "out vec4 fColor;\n"
        "void main() {\n"
        "  if (gInfo.z < 0.0) {\n"
        "    fColor = vec4(0.25, gInfo.x, gInfo.y, 1.0);\n"
        "  } else {\n"
        "    fColor = vec4(gl_FragCoord.z, 0.25, 0.25, 1.0);\n"
        "  }\n"
        "}\n";
    Program *program = mDevice->CreateProgram();

    lwnTest::GLSLCHelper glslcHelper(mDevice, 0x100000UL, g_glslcLibraryHelper, g_glslcHelperCache);
    MemoryPool *scratchMemPool = mDevice->CreateMemoryPool(NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, MemoryPoolType::GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, NULL);

    if (!glslcHelper.CompileAndSetShaders(program, vs, gs, fs)) {
        DEBUG_PRINT(("Shader compile error.\n======\n%s\n\n", glslcHelper.GetInfoLog()));
        return;
    }
    mQueueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    LWNint maxViewports = 0;
    mDevice->GetInteger(DeviceInfo::MAX_VIEWPORTS, &maxViewports);
    if (maxViewports < 16) {
        return;
    }
    int viewportSize = FBO_SIZE / 4;
    LWNfloat viewports[16 * 4];
    LWNint scissors[16 * 4];
    LWNfloat depthRanges[16 * 2];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            LWNfloat* viewport = &viewports[row * 16 + col * 4];
            LWNint* scissor = &scissors[row * 16 + col * 4];
            LWNfloat* depthRange = &depthRanges[row * 8 + col * 2];
            viewport[0] = col * viewportSize;
            viewport[1] = row * viewportSize;
            viewport[2] = viewportSize;
            viewport[3] = viewportSize;
            scissor[0] = int(viewport[0]) + 1;
            scissor[1] = int(viewport[1]) + 1;
            scissor[2] = int(viewport[2]) - 2;
            scissor[3] = int(viewport[3]) - 2;
            depthRange[0] = row / 3.0;
            depthRange[1] = col / 3.0;
        }
    }

    // Set up the viewports in multiple ranges chosen at random to make sure
    // we exercise a bug we had in the original logic where we tried to use
    // the same loop counter to select a hardware viewport to update and to
    // identify the values to use.  This works only if <first> is zero.
    mQueueCB.SetViewports(0, 4, viewports);
    mQueueCB.SetViewports(4, 12, viewports + 4*4);
    mQueueCB.SetScissors(8, 8, scissors + 8*4);
    mQueueCB.SetScissors(0, 8, scissors);
    mQueueCB.SetDepthRanges(0, 16, depthRanges);
    mQueueCB.SetDepthRanges(2, 3, depthRanges + 2*2);
    mQueueCB.DrawArrays(DrawPrimitive::POINTS, 0, 1);
    mQueueCB.submit();
    mQueue->Finish();
    program->Free();
    scratchMemPool->Free();
}

vec4 Test::calcExpected(int x, int y)
{
    int viewportSize = FBO_SIZE / 4;
    int col = x / viewportSize;
    int row = y / viewportSize;
    int vpX = x % viewportSize;
    int vpY = y % viewportSize;
    if (vpX == 0 || vpY == 0 || vpX == viewportSize - 1 || vpY == viewportSize - 1) {
        // Outside scissor rect
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    float clipX = (vpX + 0.5f) * 2.0f / viewportSize - 1.0f;
    float clipY = (vpY + 0.5f) * 2.0f / viewportSize - 1.0f;
    if (mWindowOriginMode == WindowOriginMode::UPPER_LEFT) {
        clipY = -clipY;
    }
    float clipZ = 2.0f * clipY;
    float nearZ = -1.0f;
    float farZ = 1.0f;
    float zScale = 0.5f;
    float zOffset = 0.5f;
    if (mDepthMode == DepthMode::NEAR_IS_ZERO) {
        nearZ = 0.0f;
        zScale = 1.0f;
        zOffset = 0.0f;
    }
    if (clipZ < nearZ || clipZ > farZ) {
        // Depth clipping.
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
    if (clipX < 0.0f) {
        // Left side of the cell is essentially an encoded viewport index.
        return vec4(0.25, col * 0.25, row * 0.25, 1.0);
    }
    // Right side of the cell is window depth after clip-space and depth range remapping.
    float depthRangeNear = row / 3.0f;
    float depthRangeFar = col / 3.0f;
    float normalizedZ = clipZ * zScale + zOffset;
    float windowZ = depthRangeNear + normalizedZ * (depthRangeFar - depthRangeNear);
    return vec4(windowZ, 0.25, 0.25, 1.0);
}

bool compare(const vec4& v1, const vec4& v2)
{
    float dr = v1[0] - v2[0];
    float dg = v1[1] - v2[1];
    float db = v1[2] - v2[2];
    float da = v1[3] - v2[3];
    return dr*dr + dg*dg + db*db + da*da < 0.001f;
}

bool Test::verify()
{
    LWNsizei bufferSize = FBO_SIZE * FBO_SIZE * sizeof(vec4);
    MemoryPoolAllocator allocator(mDevice, NULL, bufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(mDevice).SetDefaults();
    Buffer *pbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, bufferSize);
    CopyRegion copyRegion = { 0, 0, 0, FBO_SIZE, FBO_SIZE, 1 };
    mQueueCB.CopyTextureToBuffer(mFbo.getColorTexture(0), NULL, &copyRegion, pbo->GetAddress(), CopyFlags::NONE);
    mQueueCB.submit();
    mQueue->Finish();
    vec4* pixel = static_cast<vec4*>(pbo->Map());
    for (int y = 0; y < FBO_SIZE; ++y) {
        for (int x = 0; x < FBO_SIZE; ++x) {
            vec4 expected = calcExpected(x, y);
            if (!compare(expected, *pixel)) {
                DEBUG_PRINT(
                    ("Failure at (%d, %d): Expected (%f, %f, %f, %f), got (%f, %f, %f, %f)\n",
                     x, y, expected[0], expected[1], expected[2], expected[3], (*pixel)[0],
                     (*pixel)[1], (*pixel)[2], (*pixel)[3]));
                DEBUG_PRINT(("Device initialized with %s, %s.\n",
                    (mWindowOriginMode == WindowOriginMode::LOWER_LEFT) ? "LOWER_LEFT" : "UPPER_LEFT",
                    (mDepthMode == DepthMode::NEAR_IS_MINUS_W) ? "NEAR_IS_MINUS_W" : "NEAR_IS_ZERO"));
                allocator.freeBuffer(pbo);
                return false;
            }
            ++pixel;
        }
    }
    allocator.freeBuffer(pbo);
    return true;
}

bool Test::run()
{
    render();
    return verify();
}

} // namespace

class LWLWiewportsTest {
    bool runTest(WindowOriginMode windowOriginMode, DepthMode depthMode) const;
public:
    LWNTEST_CppMethods();
};

lwString LWLWiewportsTest::getDescription() const
{
    return "Test for viewport/scissor/depth range arrays and coordinate system control. For each "
           "combination of window origin mode and depth mode, create an LWN device and supporting "
           "objects, then draw quads to an FBO. The quads are dispatched to different viewports "
           "by a geometry shader. Each quad is affected by a different scissor rectangle and "
           "depth range, and is clipped by the near clipping plane. Shading for each unclipped "
           "and unscissored fragment is based on the viewport index or the window depth "
           "coordinate. Read back the image and compare it to expected output computed on the "
           "CPU. Output is red/green.";
}

int LWLWiewportsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(21, 6);
}


bool LWLWiewportsTest::runTest(WindowOriginMode windowOriginMode, DepthMode depthMode) const
{
    bool status = false;

    DisableLWNObjectTracking();

    DeviceState *testDevice = new DeviceState(LWNdeviceFlagBits(0),
                                                                windowOriginMode, depthMode);
    if (testDevice && testDevice->isValid()) {
        testDevice->SetActive();
        Test test(testDevice, windowOriginMode, depthMode);
        status = test.run();
    }

    delete testDevice;
    DeviceState::SetDefaultActive();

    EnableLWNObjectTracking();

    return status;
}

void LWLWiewportsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    if (runTest(WindowOriginMode::LOWER_LEFT, DepthMode::NEAR_IS_MINUS_W) &&
        runTest(WindowOriginMode::UPPER_LEFT, DepthMode::NEAR_IS_MINUS_W) &&
        runTest(WindowOriginMode::LOWER_LEFT, DepthMode::NEAR_IS_ZERO)    &&
        runTest(WindowOriginMode::UPPER_LEFT, DepthMode::NEAR_IS_ZERO))
    {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWLWiewportsTest, lwn_viewports, );

