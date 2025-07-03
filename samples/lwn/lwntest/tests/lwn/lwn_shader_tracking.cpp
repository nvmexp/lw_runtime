/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#if defined(HAS_DEVTOOLS)
#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <array>
#include <numeric>

#include "cmdline.h"
#include "lwndevtools_bootstrap.h"
static const LWNdevtoolsBootstrapFunctions* devtools;

#define DEBUG_MODE 0

#if DEBUG_MODE
#define DEBUG_PRINT(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define DEBUG_PRINT(x)
#endif

using namespace lwn;

#if defined(LW_TEGRA)

#define TEST(x,m)\
    if( !(x) ) { DEBUG_PRINT((m "\n")); return false; }

static uint32_t HwStageToGLStage(uint32_t stage)
{
    //from lwn/include/lwnconsts.h
    enum __LWNHWShaderStage {
        __LWN_HW_SHADER_STAGE_VERTEX_A      = 0,
        __LWN_HW_SHADER_STAGE_VERTEX_B      = 1,
        __LWN_HW_SHADER_STAGE_TESS_CONTROL  = 2,
        __LWN_HW_SHADER_STAGE_TESS_EVAL     = 3,
        __LWN_HW_SHADER_STAGE_GEOMETRY      = 4,
        __LWN_HW_SHADER_STAGE_PIXEL         = 5,
    };

    switch(stage) {
    case  __LWN_HW_SHADER_STAGE_VERTEX_A:
    case  __LWN_HW_SHADER_STAGE_VERTEX_B:
        return LWN_SHADER_STAGE_VERTEX;

    case  __LWN_HW_SHADER_STAGE_TESS_CONTROL:
        return LWN_SHADER_STAGE_TESS_CONTROL;

    case  __LWN_HW_SHADER_STAGE_TESS_EVAL:
        return LWN_SHADER_STAGE_TESS_EVALUATION;

    case  __LWN_HW_SHADER_STAGE_GEOMETRY:
        return LWN_SHADER_STAGE_GEOMETRY;

    case  __LWN_HW_SHADER_STAGE_PIXEL:
        return LWN_SHADER_STAGE_FRAGMENT;

    default:
        return ~0U;
    }
}

static bool StagesOffsetsFromGrCtx(Queue* q, std::array<uint32_t,GLSLC_NUM_SHADER_STAGES>& offsets)
{
    bool success;
    static const size_t BUFFER_SIZE = 80000;
    uint8_t buffer[BUFFER_SIZE];
    size_t size = 0;

    success = devtools->GetGrCtxSizeForQueue((LWNqueue *)q, &size);
    TEST(success, "get ctx size failed!");
    TEST(size < BUFFER_SIZE, "ctx too big!");

    memset(buffer, 0, BUFFER_SIZE);

    success = devtools->GetGrCtxForQueue((LWNqueue *)q, buffer, size);
    TEST(success, "get ctx failed!");

    static const size_t baseShadows = 8840;
    DEBUG_PRINT(("stages %d\n", GLSLC_NUM_SHADER_STAGES));

#define INDEX_(i) (baseShadows + 4 * stage + i)
    for (uint32_t stage = 0; stage < GLSLC_NUM_SHADER_STAGES; ++stage) {
        uint32_t offset =
                buffer[INDEX_(0)]       |
                buffer[INDEX_(1)] <<  8 |
                buffer[INDEX_(2)] << 16 |
                buffer[INDEX_(3)] << 24 ;

        DEBUG_PRINT(("HW stage %d, GL stage %d => %p \n", stage, HwStageToGLStage(stage), offset));
        offsets[HwStageToGLStage(stage)] = offset;
    }
#undef INDEX_

    return true;
}

#undef TEST

#elif defined(_WIN32)
typedef PROC(WINAPI *DRVGETPROCADDRESS)(LPCSTR lpszProc);
bool doBootstrap()
{
    DEBUG_PRINT(("bootstraping\n"));
    DRVGETPROCADDRESS DrvGetProc = NULL;

    if (!DrvGetProc) {
        HMODULE hModule = LoadLibrary("lwoglv32.dll");
        if (!hModule) { hModule = LoadLibrary("lwoglv64.dll"); }

        DrvGetProc = (DRVGETPROCADDRESS)GetProcAddress(hModule, "DrvGetProcAddress");
        DEBUG_PRINT(("got getter: %p\n", (void*)DrvGetProc));
    }
    if (!DrvGetProc) { return false; }

    PFNLWNDEVTOOLSBOOTSTRAP DTbootstrap = (PFNLWNDEVTOOLSBOOTSTRAP)DrvGetProc("pq12md6bbz");
    DEBUG_PRINT(("got bootstraper: %p\n", (void*)DTbootstrap));
    if (!DTbootstrap) { return false; }

    devtools = DTbootstrap();
    DEBUG_PRINT(("bootstraped!\n"));
    return true;
}
#else

#endif

// --- base class : define test structure ---
class LWNShaderTracking
{
public:
    lwString getDescription() const;
    int isSupported(void) const {
        return true;
    }
    void doGraphics(void);
    void initGraphics(void) const { lwnDefaultInitGraphics(); }
    void exitGraphics(void) const { lwnDefaultExitGraphics(); }

    virtual ~LWNShaderTracking() { }
protected:
    bool initialize();
    void fail(const char* reason);
    virtual void payload() = 0;
    virtual void test() = 0;
    void drawSeal();
    void getExpectedPrepads();

    DeviceState         *deviceState;
    Device              *device;
    Queue               *queue;
    Sync                *sync_top;
    QueueCommandBuffer  *queueCB;
    Program             *program;

    std::array<LWNdevtoolsSphPrepad,GLSLC_NUM_SHADER_STAGES> expected_prepads;

    bool success;
};

lwString LWNShaderTracking::getDescription() const
{
    lwStringBuf sb;
    sb << "This test family (lwn_shader_tracking_graphics & lwn_shader_tracking_compute)\n"
          "binds Programs of various sorts to Queues and makes sure we can retrieve\n"
          "the associated tracking information: debug hashes, shader sizes, Program pointers."
    ;
    return sb.str();
}

bool LWNShaderTracking::initialize()
{
    success     = false;
#if defined(LW_TEGRA)
    devtools    = lwnDevtoolsBootstrap();
#elif defined(_WIN32)
    if (!doBootstrap()) { return false; }
#else
    assert(!"we should not get here, this is unsupported");
#endif

    deviceState = DeviceState::GetActive();
    device      = deviceState->getDevice();
    queueCB     = &deviceState->getQueueCB();
    queue       = deviceState->getQueue();
    sync_top    = device->CreateSync();
    program     = device->CreateProgram();
    success     = true;

    return true;
}

void LWNShaderTracking::fail(const char* reason)
{
    lwnTest::fail(reason);
    success = false;
}

void LWNShaderTracking::drawSeal()
{
    queueCB->SetScissor(100, 100, 10, 10); //x y h w, pretty random
    if (success) { queueCB->ClearColor(0, 0.0, 1.0, 0.0, 1.0); }
    else         { queueCB->ClearColor(0, 1.0, 0.0, 0.0, 1.0); }
    queueCB->submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

void LWNShaderTracking::doGraphics()
{
    if (!initialize()) { return; }
    DEBUG_PRINT(("LWNprogram is: %p\n", program));
    payload();
    getExpectedPrepads();
    test();
    drawSeal();
}

#define VERIFY_(x) \
        if (prepad.x != expected_prepads[stage].x) {\
            fail(#x " mismatch.");\
            return;\
        }

void LWNShaderTracking::getExpectedPrepads()
{
    const GLSLCoutput *compiled = g_glslcHelper->GetCompiledOutput(0);
    DEBUG_PRINT(("last compiled shader has %d sections\n", compiled->numSections));
    for (uint32_t section = 0; section < compiled->numSections; ++section) {
        // This comes from lwn_glslc.cpp in lwntest, around TestEntriesThatDontModifyBinaryOutput()
        if (compiled->headers[section].genericHeader.common.type != GLSLC_SECTION_TYPE_GPU_CODE) {
            continue;
        }

        const auto& section_headers = compiled->headers[section].gpuCodeHeader;
        uint8_t *data = (uint8_t *)compiled + section_headers.common.dataOffset;
        uint8_t *control_subsection = data + section_headers.controlOffset;

        // See lwn/glslc/lwnglslc_binary_layout.h for proper offsets
        assert(section_headers.common.size > 1912);
        auto programGlStage = *((uint32_t*)(control_subsection + 1812));
        auto ucodeSize = *((uint32_t*)(control_subsection + 1784));
        auto debugHash = *((uint64_t*)(control_subsection + 1912));

        expected_prepads[programGlStage] = { GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER, ucodeSize, {(LWNprogram *)program}, debugHash};
        DEBUG_PRINT(("expectation for stage: %d => program: %p ucodeSize: %d debugHash: %llx\n",
                     programGlStage, program, ucodeSize, debugHash));
    }
}

// --- test graphics shaders ---
class LWNShaderTrackingGraphics : public LWNShaderTracking
{
public:
    ~LWNShaderTrackingGraphics() { }
private:
    void payload();
    void test();
};

void LWNShaderTrackingGraphics::payload()
{
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";

    // Compile and call lwnProgramSetShaders.
    int original_dbg_level = lwnGlslcDebugLevel;
    lwnGlslcDebugLevel = 2;
    if (!g_glslcHelper->CompileAndSetShaders(program, fs, vs)) {
        fail("Compilation failed.");
        DEBUG_PRINT(("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }
    lwnGlslcDebugLevel = original_dbg_level;

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB->ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    queueCB->BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    // We signal the CPU that shader are bound
    queueCB->FenceSync(sync_top, SyncCondition::GRAPHICS_WORLD_SPACE_COMPLETE, 0 /*!FLUSH_FOR_CPU*/ );
    queueCB->BindVertexArrayState(vertex);
    queueCB->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    queueCB->DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

    queueCB->submit();
    queue->Flush();

    // Flush makes sure the driver gets this CommandBuffer and that sync_top gets armed.
    // Waiting for the sync takes care of the uncertainties in GPU scheduling, and makes sure
    // that the MME macros for BindProgram have updated the shadow ram locations for the new
    // ProgramRegion.
    sync_top->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
}

void LWNShaderTrackingGraphics::test()
{
#if defined(LW_TEGRA)
    std::array<uint32_t,GLSLC_NUM_SHADER_STAGES> offsets{{0, 0, 0, 0, 0, 0}};
    static const std::array<bool,GLSLC_NUM_SHADER_STAGES> expected_stages{{
        true /*LWN_SHADER_STAGE_VERTEX*/, true /*FRAGMENT*/, false, false, false, false
    }};

    LWNdevice *d = reinterpret_cast<LWNdevice*>(device);

    bool success = StagesOffsetsFromGrCtx(queue, offsets);
    if (!success) {
        fail("Could not read offsets from context.");
        return;
    }

    const static int prepad_offset_relative_to_SPH = 0x30;
    for (int stage = 0; stage < GLSLC_NUM_SHADER_STAGES; ++stage) {
        uint32_t offset = offsets[stage];

        if (!expected_stages[stage]) {
            DEBUG_PRINT(("no expectation for stage %d\n", stage));
            continue;
        }

        if (!offset) {
            fail("Expected stage not found.");
            return;
        }

        if (offset < prepad_offset_relative_to_SPH) {
            fail("Offset too small.");
            return;
        }

        offset -= prepad_offset_relative_to_SPH;
        void *prepadCpuVA = devtools->GetCpuVAforShaderHeapOffset(d, offset);
        DEBUG_PRINT(("stage: %d offset: %d VA: %p\n", stage, offset, prepadCpuVA));
        if (!prepadCpuVA) {
            fail("Could not get to CPU VA for stage.");
            return;
        }

        const auto& prepad = *(static_cast<LWNdevtoolsSphPrepad*>(prepadCpuVA));
        DEBUG_PRINT(("program: %p ucodeSize: %d debugHash: %llx\n",
                     prepad.program, prepad.ucodeSize, prepad.debugHash));

        VERIFY_(magic)
        VERIFY_(ucodeSize)
        VERIFY_(program)
        VERIFY_(debugHash)
    }
#else
    // TODO: we are missing a path to the MME shadow RAM on Windows
    // Wait for http://lwbugs/1894368 #LWN LOP D [project] - Shader Profiler
#endif
}

OGTEST_CppTest(LWNShaderTrackingGraphics, lwn_shader_tracking_graphics, );

// --- test compute shaders ---
class LWNShaderTrackingCompute : public LWNShaderTracking
{
public:
    ~LWNShaderTrackingCompute() { }
private:
    void payload();
    void test();
};

void LWNShaderTrackingCompute::payload()
{
    lwShader cs;

    cs = ComputeShader(430);
    cs << "void main() { float dummy = sin(gl_GlobalIlwocationID.x); }";
    cs.setCSGroupSize(8, 8);

    int original_dbg_level = lwnGlslcDebugLevel;
    lwnGlslcDebugLevel = 2;
    if (!g_glslcHelper->CompileAndSetShaders(program, cs)) {
        fail("Compilation failed.");
        DEBUG_PRINT(("Compile failed:\n%s\n", g_glslcHelper->GetInfoLog()));
        return;
    }
    lwnGlslcDebugLevel = original_dbg_level;

    queueCB->ClearColor(0, 1.0, 1.0, 1.0, 1.0);
    queueCB->BindProgram(program, ShaderStageBits::COMPUTE);
    queueCB->DispatchCompute(1, 1, 1);
    queueCB->Barrier(BarrierBits::ORDER_PRIMITIVES |
                     BarrierBits::ILWALIDATE_TEXTURE |
                     BarrierBits::ILWALIDATE_SHADER);

    // We dont need to wait for the GPU in this case,
    // since the magic happens in CPU at DispatchCompute,
    // but we do need to submit since CommandBufferDispatch
    // is only a token insertion.
    queueCB->submit();
}

void LWNShaderTrackingCompute::test()
{
    const LWNprogram *active_program = devtools->GetComputeActiveProgramForQueue((const LWNqueue *)queue);
    if (active_program != (void*)program) {
        fail("Incorrect active program.");
    }

    static const int stage = 5; // compute
    uint32_t ucodeSize = 0;
    uint64_t debugHash = 0;
    if (!devtools->GetUcodeSizeForComputeProgram(active_program, &ucodeSize)) {
        fail("Could not get ucode size.");
        return;
    }
    if (!devtools->GetDebugHashForComputeProgram(active_program, &debugHash)) {
        fail("Could not get hash.");
        return;
    }
    DEBUG_PRINT(("program: %p ucodeSize: %d debugHash: %llx\n",
                 active_program, ucodeSize, debugHash));

    const struct LWNdevtoolsSphPrepad prepad = {GLSLC_GFX_GPU_CODE_SECTION_DATA_MAGIC_NUMBER, ucodeSize, {(LWNprogram *)program}, debugHash};
    VERIFY_(magic);
#if defined(LW_WINDOWS)
    if ((prepad.ucodeSize == 0x80 || prepad.ucodeSize == 0x100) && expected_prepads[stage].ucodeSize == 0x40) {
        // On the Windows reference implementation, testing the ucodeSize
        // field is fundamentally flawed, since the size returned by the
        // devtools API comes from a shader compiled online for the target GPU
        // instead of the NX-specific microcode produced by GLSLC.  The sizes
        // won't match in general.  Skip an exact match test if the sizes
        // match the pattern seen on Volta/Turing (where an empty program is
        // 128B) and on Ampere (where an empty program is 256B).
    } else
#endif
    {
        VERIFY_(ucodeSize);
    }
    VERIFY_(program);
    VERIFY_(debugHash);
}

OGTEST_CppTest(LWNShaderTrackingCompute, lwn_shader_tracking_compute, );
#endif //#if defined(HAS_DEVTOOLS)
