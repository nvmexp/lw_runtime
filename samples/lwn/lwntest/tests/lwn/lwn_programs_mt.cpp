/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
#include <mutex>
#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "../../elw/cmdline.h"

// Hack to finish all compilation threads before rendering anything.
#define HACK_FORCE_SYNC_BEFORE_RENDER   0

// Hack to force each GLSLC compilation thread to run to completion before
// running the next thread to work around GLSLC thread safety issues (bug
// 1796973).
#define HACK_FORCE_SYNC_GLSLC_COMPILE   0

#define DEBUG_ENABLED 0
#if DEBUG_ENABLED
#define DEBUG_LOG(x) printf x
#else
#define DEBUG_LOG(x)
#endif

using namespace lwn;

class LWNProgramMultithreadTest
{
    int m_threads;                  // number of threads to test
    bool m_precompile;              // should GLSLC shaders be precompiled
    bool m_useSpecialization;       // should GLSLC shaders perform multi-threaded specialization compiles.

    // For pre-compiled shaders, we run small 32x32 cells because we reuse two
    // canned GLSLC compile outputs.  For shaders compiled online, we use
    // larger 64x64 cells to reduce the number of shaders the GLSLC compiler
    // needs to build.
    static const int cellSizePrecompiled = 32;
    static const int cellSizeOnline = 64;
    static int cellSize(bool precompile)    { return precompile ? cellSizePrecompiled : cellSizeOnline; }
    static int cellsX(bool precompile)      { return 640 / cellSize(precompile); }
    static int cellsY(bool precompile)      { return 480 / cellSize(precompile); }
    int cellSize() const                    { return cellSize(m_precompile); }
    int cellsX() const                      { return cellsX(m_precompile); }
    int cellsY() const                      { return cellsY(m_precompile); }

    // Structure holding per-thread information for compile threads.
    struct CompilerThreadInfo {
        bool precompile;            // Will GLSLC shaders be pre-compiled?
        bool useSpecialization;     // Will parallel compiles be performed using specialization?
        int threadIndex;            // Index of compilation thread
        int threadCount;            // Number of compilation threads

        int nLWNPrograms;           // Total number of LWN programs to build
        Program *programs;          // Array of LWN program objects

        int nGLSLPrograms;          // Total number of GLSLC programs built
        GLSLCoutput **glslcOutputs; // Array of GLSLC compiler outputs

        bool *completionFlags;      // Array of flags indicating completed compiles

        lwnUtil::GLSLCLibraryHelper *glslcLibraryHelper;
        lwnTest::GLSLCHelper *glslcHelper;
        LWOGthread *lwogThread;     // Thread descriptor
    };

public:
    // Our automatic allocation tracking support is not thread-safe.
    LWNTEST_CppMethods_NoTracking();

    LWNProgramMultithreadTest(int threads, bool precompile, bool useSpecialization) :
        m_threads(threads), m_precompile(precompile), m_useSpecialization(useSpecialization) {}

    static GLSLCoutput *compileGLSLShader(lwnTest::GLSLCHelper *helper, int id, bool precompile);
    static LWNboolean compileGLSLShaderPreSpecialized(lwnTest::GLSLCHelper *helper, bool precompile);

    static GLSLCoutput *compileGLSLShaderBackendOnly(const GLSLCcompileObject *compileObject, GLSLCLibraryHelper * libraryHelper, int id, bool precompile);
    static void compilerThread(void *data);
};

lwString LWNProgramMultithreadTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Multithreaded shader compilation/loading test; verifies that "
        "programs can be loaded conlwrrently from multiple threads.  A set "
        "of " << m_threads << " worker threads sets up LWNprogram objects "
        "in parallel, ";
    if (m_precompile) {
        sb << 
            "using pre-compiled GLSLC outputs";
    } else if (!m_useSpecialization) {
        sb << "using GLSLC to compile each shader on the fly";
    } else {
        sb << "using GLSLC to compile a pre-specialized version once, "
              "then recompile using multi-threaded shader specialization";
    }
    sb <<
        ".  A test cell is drawn with each program object and should display ";
    if (m_precompile) {
        sb << "a green and blue checkerboard of cells.";
    } else {
        sb << "a repeating gradient pattern of cells.";
    }
    return sb.str();
}

int LWNProgramMultithreadTest::isSupported() const
{
#if defined(LW_HOS)
    // For on-line GLSLC compiles, require 28MB of compiler heap per thread
    // and skip the test if memory is insufficient.
    if (!m_precompile && hosCompilerHeapMB < 28 * m_threads) {
        return 0;
    }
#endif
    return lwogCheckLWNAPIVersion(5, 0);
}

// Similar to compileGLSLShader, but instead of baking the constants into the GLSL
// source string, they are sent in as specialization constants later, so we use
// a UBO here (even though the values will be specialized at compilation time and
// the UBO memory will never be referenced).
LWNboolean LWNProgramMultithreadTest::compileGLSLShaderPreSpecialized(lwnTest::GLSLCHelper *helper,
                                                                      bool precompile)
{
    // We define source as a static variable here since the string needs to persist even once this function
    // returns.  This function only compiles the front-end portion.  The back-end portion of the program is compiled
    // later in a threaded context.  The compile object contains references to the original source strings
    // for compiling various output hashes, so the strings need to persist since GLSLC doesn't try to make
    // any internal copies.
    static const char * shadersSpecialized[2] = {
        // Vertex shader.  The r, g, and b uniforms will get specialized.
        "#version 440\n"
        "layout(location=0) in vec3 position;\n"
        // These will be specialized.
        "layout (binding = 0) uniform constantUBO {\n"
        "  float r;\n"
        "  float g;\n"
        "  float b;\n"
        "};\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = vec3(r,g,b);\n"
        "}\n",

        // Fragment shader
        "#version 440\n"
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n"
    };

    lwn::ShaderStage stages[2] = { lwn::ShaderStage::VERTEX, lwn::ShaderStage::FRAGMENT };

    // Compiles pre-specialized.  The GLSLCHelper compile object will hold the results.
    if (!helper->CompileShadersPreSpecialized(stages, 2, shadersSpecialized) ) {
        DEBUG_LOG(("Compilation error log: %s\n", helper->GetInfoLog()));
        return LWN_FALSE;
    }

    return LWN_TRUE;
 }

// Performs specialization to bake in the expected values to each shader
// variant.  Requires that <compileObject> have been previously been set up
// using a presepcialized compile step.
GLSLCoutput *LWNProgramMultithreadTest::compileGLSLShaderBackendOnly(const GLSLCcompileObject *compileObject, GLSLCLibraryHelper * libraryHelper, int id, bool precompile)
{
    // For pre-compiled tests, we generate green or blue (two total shaders).
    // For online compilation, generate a gradient pattern based on the
    // program id's location in the grid.
    float r, g, b;
    {
        int xcells = cellsX(precompile);
        int ycells = cellsY(precompile);
        int cx = id % xcells;
        int cy = id / xcells;
        r = 0.2;
        g = 0.2 + (0.8 * cx) / (xcells - 1);
        b = 0.2 + (0.8 * cy) / (ycells - 1);
    }

    // Create specialization constants for r, g, b.
    GLSLCspecializationSet set;
    GLSLCspecializationUniform uniforms[3];
    for (int i = 0; i < 3; ++i) {
        uniforms[i].elementSize = 4; // 4 bytes
        uniforms[i].numElements = 1; // 1 element in the array
        uniforms[i].uniformName = (i == 0) ? "r" : (i == 1) ? "g" : "b";
        uniforms[i].values =      (i == 0) ? &r  : (i == 1) ? &g  : &b;
    }
    set.uniforms = uniforms;
    set.numUniforms = 3;

    GLSLCspecializationBatch batch;
    batch.numEntries = 1;
    batch.entries = &set;

    GLSLCresults const * const * results = libraryHelper->glslcCompileSpecializedMT(compileObject, &batch);

    if (!results) {
        DEBUG_LOG(("Internal GLSLC error.  No results available from specialized compile.\n"));
    } else {
        bool errors = false;
        for (unsigned int i = 0; i < batch.numEntries; ++i) {
            if (!results[i]->compilationStatus->success) {
                DEBUG_LOG(("Compilation error on specialization result %d:\n%s\n", i, results[i]->compilationStatus->infoLog));
                errors = true;
            }
        }

        if (errors){
            libraryHelper->glslcFreeSpecializedResultsMT(results);
            return NULL;
        }
    }

    const GLSLCoutput *compiled = results[0]->glslcOutput;
    GLSLCoutput *copy = (GLSLCoutput *) __LWOG_MALLOC(compiled->size);
    memcpy(copy, compiled, compiled->size);
    libraryHelper->glslcFreeSpecializedResultsMT(results);

    return copy;
 }

static std::mutex lwShaderMutex; // A mutex for constructing the thread-unsafe lwShader classes.
                                 // This static is declared as global instead of function local
                                 // on purpose since MSVC120 does not guarentee thread-safe construction
                                 // of static local variables.

GLSLCoutput *LWNProgramMultithreadTest::compileGLSLShader(lwnTest::GLSLCHelper *helper, int id,
                                                          bool precompile)
{
    // For pre-compiled tests, we generate green or blue (two total shaders).
    // For online compilation, generate a gradient pattern based on the
    // program id's location in the grid.
    float r, g, b;
    if (precompile) {
        assert(id < 2);
        r = 0.2;
        g = (id & 1) ? 0.0 : 1.0;
        b = (id & 1) ? 1.0 : 0.0;
    } else {
        int xcells = cellsX(precompile);
        int ycells = cellsY(precompile);
        int cx = id % xcells;
        int cy = id / xcells;
        r = 0.2;
        g = 0.2 + (0.8 * cx) / (xcells - 1);
        b = 0.2 + (0.8 * cy) / (ycells - 1);
    }
    VertexShader vs;
    FragmentShader fs;
    lwShaderMutex.lock();
    // need mutex for lwShaderWithStageClass ctor access to global lwShaderHandlePool and to
    // make sure that the streaming operator is not used while another thread causes a
    // resize of the lwShaderHandlePool.
    vs = VertexShader(440);
    fs = FragmentShader(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = vec3(" << r << ", " << g << ", " << b << ");\n"
        "}\n";
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";
    lwShaderMutex.unlock();
    helper->EnableMultithreadCompilation(true);
    if (!helper->CompileShaders(vs, fs)) {
        DEBUG_LOG(("Compilation error log: %s\n", helper->GetInfoLog()));
        return NULL;
    }

    const GLSLCoutput *compiled = helper->GetGlslcOutput();
    GLSLCoutput *copy = (GLSLCoutput *) __LWOG_MALLOC(compiled->size);
    memcpy(copy, compiled, compiled->size);
    return copy;
}

void LWNProgramMultithreadTest::compilerThread(void *data)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    CompilerThreadInfo *threadInfo = (CompilerThreadInfo *) data;

    // Loop over all LWN programs.  Each of the N threads processes every Nth
    // program.
    for (int i = 0; i < threadInfo->nLWNPrograms; i++) {
        if ((i % threadInfo->threadCount) != threadInfo->threadIndex) {
            continue;
        }

        // Figure out which GLSL program we should be using.  For pre-compiled
        // shaders, we alternate based on the cell location.  For online
        // compiled shaders, we build the program on the fly.
        int glslProgram = i;
        if (threadInfo->precompile) {
            assert(threadInfo->nGLSLPrograms == 2);
            int xcells = cellsX(threadInfo->precompile);
            int cx = i % xcells;
            int cy = i / xcells;
            glslProgram = (cx + cy) & 1;
        } else {
            assert(threadInfo->nGLSLPrograms == cellsX(threadInfo->precompile) * cellsY(threadInfo->precompile));
            if (threadInfo->useSpecialization) {
                // The compile object we get here should be from the same helper where the front-end portion was called.
                threadInfo->glslcOutputs[glslProgram] =
                    compileGLSLShaderBackendOnly(g_glslcHelper->GetCompileObject(), threadInfo->glslcLibraryHelper, glslProgram, false);
            } else {
                // The compile object we get here should be from the same helper where the front-end portion was called.
                threadInfo->glslcOutputs[glslProgram] = compileGLSLShader(threadInfo->glslcHelper, glslProgram, false);
            }
        }

        // Set up a LWNprogram based on the appropriate GLSLC output.
        Program *pgm = &threadInfo->programs[i];
        pgm->Initialize(device);
        if (threadInfo->glslcOutputs[glslProgram]) {
            threadInfo->glslcHelper->SetShaders(pgm, threadInfo->glslcOutputs[glslProgram]);
        }

        // Mark this program as being ready for use.
        threadInfo->completionFlags[i] = true;

        // Yield after setting up each program, since the HOS scheduler
        // doesn't alternate threads of matching priority 
        lwogThreadYield();
    }
}

static void lwnProgramMTCompilerThread(void *data)
{
    LWNProgramMultithreadTest::compilerThread(data);
}

void LWNProgramMultithreadTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int xcells = cellsX();
    int ycells = cellsY();
    int nLWNPrograms = xcells * ycells;
    int nGLSLPrograms = m_precompile ? 2 : nLWNPrograms;

    CompilerThreadInfo *threadInfoArray = new CompilerThreadInfo[m_threads];
    Program *programs = new Program[nLWNPrograms];
    GLSLCoutput **glslcOutputs = new GLSLCoutput *[nGLSLPrograms];
    bool *done = new bool[nLWNPrograms];

#if defined(LW_HOS)
    // Compute the set of cores available for worker threads; we reserve core
    // #0 for the main application thread.
    uint64_t availableCores = lwogThreadGetAvailableCoreMask();
    availableCores &= ~(1ULL << lwogThreadGetLwrrentCoreNumber());
#endif

    // Set up the GLSLC output pointers, building shaders if pre-compilaton is
    // enabled.
    for (int i = 0; i < nGLSLPrograms; i++) {
        if (m_precompile) {
            glslcOutputs[i] = compileGLSLShader(g_glslcHelper, i, true);
        } else {
            glslcOutputs[i] = NULL;

            // compiles pre-specialized.  The compile object will be contained inside the g_glslcHelper
            // to be used later for a specialized compile.
            if (m_useSpecialization) {
                compileGLSLShaderPreSpecialized(g_glslcHelper, true);
            }
        }
    }

    // Clear the "done" flags for the GLSLC output.
    for (int i = 0; i < nLWNPrograms; i++) {
        done[i] = false;
    }

    // Create GLSLC helper objects for each shader thread.  For online
    // compilation, this is needed to have separate GLSLC compiler object
    // instances.  For both online and offline, separate helpers are used so
    // that each has its own SHADER_CODE pool.
    for (int i = 0; i < m_threads; i++) {
        CompilerThreadInfo *threadInfo = &threadInfoArray[i];
        threadInfo->glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper();
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
        threadInfo->glslcLibraryHelper->LoadDLL(lwnGlslcDLL);
#else
        threadInfo->glslcLibraryHelper->LoadDLL(NULL);
#endif
        threadInfo->glslcHelper = new lwnTest::GLSLCHelper(device, 1024 * 1024, threadInfo->glslcLibraryHelper, NULL);
    }

    // Fire off the compilation threads.  We need a relatively large stack
    // because one function in GLSLC burns a massive amount (bug 1797265).
    for (int i = 0; i < m_threads; i++) {
        CompilerThreadInfo *threadInfo = &threadInfoArray[i];
        threadInfo->precompile = m_precompile;
        threadInfo->useSpecialization = m_useSpecialization;
        threadInfo->threadIndex = i;
        threadInfo->threadCount = m_threads;
        threadInfo->nLWNPrograms = nLWNPrograms;
        threadInfo->programs = programs;
        threadInfo->nGLSLPrograms = nGLSLPrograms;
        threadInfo->glslcOutputs = glslcOutputs;
        threadInfo->completionFlags = done;
#if defined(LW_HOS)
        int idealCore = lwogThreadSelectCoreRoundRobin(i, availableCores);
        threadInfo->lwogThread = lwogThreadCreateOnCore(lwnProgramMTCompilerThread, threadInfo, 0xC0000, idealCore);
#else
        threadInfo->lwogThread = lwogThreadCreate(lwnProgramMTCompilerThread, threadInfo, 0xC0000);
#endif

#if HACK_FORCE_SYNC_GLSLC_COMPILE
        // Optionally force the GLSLC compiler threads to run sequentially
        // by waiting for each to complete before starting the next.
        if (!m_precompile) {
            lwogThreadWait(threadInfo->lwogThread);
            threadInfo->lwogThread = NULL;
        }
#endif
    }
    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

#if HACK_FORCE_SYNC_BEFORE_RENDER
    // Optionally wait for all the compilation threads before rendering.
    for (int i = 0; i < m_threads; i++) {
        if (threadInfoArray[i].lwogThread) {
            lwogThreadWait(threadInfoArray[i].lwogThread);
        }
    }
#endif

    // Render a cell using each LWN program object.
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < nLWNPrograms; i++) {

        // Wait for program <i> to be fully compiled.
        while (!done[i]) {
            lwogThreadYield();
        }

        int cx = i % xcells;
        int cy = i / xcells;
        int size = cellSize();
        queueCB.SetViewportScissor(cx * size + 1, cy * size + 1, size - 2, size - 2);
        queueCB.BindProgram(&programs[i], ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.submit();
        queue->Flush();
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    // Tear down all the compiler worker threads.
    for (int i = 0; i < m_threads; i++) {
        if (threadInfoArray[i].lwogThread) {
            lwogThreadWait(threadInfoArray[i].lwogThread);
        }
    }

    for (int i = 0; i < m_threads; i++) {
        delete threadInfoArray[i].glslcHelper;
        delete threadInfoArray[i].glslcLibraryHelper;
    }
    for (int i = 0; i < nLWNPrograms; i++) {
        programs[i].Finalize();
    }
    for (int i = 0; i < nGLSLPrograms; i++) {
        __LWOG_FREE(glslcOutputs[i]);
    }

    delete [] threadInfoArray;
    delete [] programs;
    delete [] glslcOutputs;
    delete [] done;
}

OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_2, (2, true, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_8, (8, true, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_2, (2, false, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_8, (8, false, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_16, (16, false, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_32, (32, false, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_64, (64, false, false));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_spec_2, (2, false, true));
OGTEST_CppTest(LWNProgramMultithreadTest, lwn_programs_mt_glslc_spec_8, (8, false, true));
