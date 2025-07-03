/*
 * Copyright (c) 2012 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNBasicComputeTest {

private:
    static const int texSize = 128;         // both images use this size texture for outputs
    static const int ctaSize = 8;           // run with 8x8 work groups
    static const int nObjects = 8;          // number of cones in the "depth test" scene
    static const int nCells = 2;

public:
    OGTEST_CppMethods();
};

int LWNBasicComputeTest::isSupported()
{
    return lwogCheckLWNAPIVersion(21, 5);
}

lwString LWNBasicComputeTest::getDescription()
{
    lwStringBuf sb;

    sb << 
        "Basic 'pretty picture' compute shader test using GLSL " << 
        "shaders.  The test should draw "
        "two images.  The image in the lower-left corner shows a view from "
        "above of a scene depicting a number of flat-shaded cones with "
        "random position, size, height, and color.  Each invocation "
        "evaluates the cone closest to the viewer above the scene and writes "
        "its color to the output image with an image store.  The image in "
        "the upper right runs a basic edge-detection algorithm on the scene "
        "in the lower left.  Each compute shader work group runs on a block "
        "of pixels in the image, with each thread fetching a pixel via "
        "texture and writing the value to shared memory.  All threads then "
        "synchronize, and the threads in the middle of the block compare "
        "their colors against those of their neighbors.  Pixels with "
        "significant differences are colored white; others are colored dark "
        "gray.  The image should show the edges of the circles in the lower "
        "left.";

    return sb.str();
}

void LWNBasicComputeTest::initGraphics(void)
{
    lwnDefaultInitGraphics();
}

void LWNBasicComputeTest::exitGraphics(void)
{
    lwnDefaultExitGraphics();
}

void LWNBasicComputeTest::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    lwShader cs;

    // Initialize the "depth tested cone" program (0).
    cs = ComputeShader(430);
    cs <<
        "layout(binding=0,rgba32f) uniform image2D image;\n"
        "uniform ObjectParams {\n"
        "  vec4 params[" << 2 * nObjects << "];\n"
        "};\n"
        "void main() {\n"
        "  vec4 color = vec4(0.5, 0.0, 0.0, 1.0);\n"
        "  float maxHeight = 0.0;\n"
        "  vec2 xy = vec2(gl_GlobalIlwocationID.xy) + 0.5;\n"
        "  for (int i = 0; i < " << nObjects << "; i++) {\n"
        "    vec2 dxy = params[2*i+0].xy - xy;\n"
        "    float oneOverObjRadius = params[2*i+0].z;\n"
        "    float objHeight = params[2*i+0].w;\n"
        "    float height = (1.0 - (length(dxy) * oneOverObjRadius)) * objHeight;\n"
        "    if (height > maxHeight) {\n"
        "      maxHeight = height;\n"
        "      color = params[2*i+1];\n"
        "    }\n"
        "  }\n"
        "  imageStore(image, ivec2(gl_GlobalIlwocationID.xy), color);\n"
        "}\n";
    cs.setCSGroupSize(ctaSize, ctaSize);
    Program *pgm0 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm0, cs)) {
        LWNFailTest();
        return;
    }

    // Initialize the edge detection program (1).
    cs = lwGLSLComputeShader(430);
    cs <<
        "layout(binding=0) uniform sampler2D texture;\n"
        "layout(binding=0,rgba8) uniform image2D image;\n"
        "shared vec3 colors["<<ctaSize<<"]["<<ctaSize<<"];\n"
        "const ivec2 deltas[4] = {{-1,0},{+1,0},{0,-1},{0,+1}};\n"
        "void main() {\n"
        "  ivec2 basexy;\n"

        // Figure out a global (x,y) for each invocation.  Each NxN work
        // group generates an (N-2)x(N-2) region of the final image.
        "  basexy = (ivec2(gl_WorkGroupSize.xy)-2) * ivec2(gl_WorkGroupID) +\n"
        "            ivec2(gl_LocalIlwocationID.xy) - 1;\n"

        // Our invocation shouldn't generate a texel in the final image if
        // it's on the edge of the workgroup, or if it's unclamped global
        // (x,y) is outside the image.
        "  bool shouldWrite =\n"
        "    (all(greaterThan(gl_LocalIlwocationID.xy, uvec2(0))) &&\n"
        "     all(lessThan(gl_LocalIlwocationID.xy, uvec2("<<ctaSize-1<<"))) &&\n"
        "     all(lessThan(basexy, ivec2("<<texSize<<"))));\n"

        // Each invocation fetches a texel from its clamped global (x,y)
        // and writes the results to shared memory.  Inject a shared
        // memory barrier and exelwtion barrier to ensure all shared
        // memory writes finish before starting the next phase.
        "  basexy = clamp(basexy, 0, " << texSize-1 << ");\n"
        "  vec3 texel = texelFetch(texture, basexy, 0).xyz;\n"
        "  colors[gl_LocalIlwocationID.y][gl_LocalIlwocationID.x] = texel;\n"
        "  memoryBarrierShared();\n"
        "  barrier();\n"

        // For ilwocations producing a pixel in the final image, compare 
        // our source texel value with the four neighboring texels from
        // shared memory.  If there's a significant difference in any 
        // neighbor, color the pixels white.  Otherwise, use a dark grey.
        "  if (shouldWrite)\n"
        "  {\n"
        "    vec4 color = vec4(0.3, 0.3, 0.3, 0.3);\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "      uvec2 deltaxy = uvec2(ivec2(gl_LocalIlwocationID.xy) + deltas[i]);\n"
        "      vec3 altTexel = colors[deltaxy.y][deltaxy.x];\n"
        "      vec3 diff = texel - altTexel;\n"
        "      if (dot(abs(diff),vec3(1,1,1)) > 0.1) {\n"
        "        color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "        break;\n"
        "      }\n"
        "    }\n"
        "    imageStore(image, basexy, color);\n"
        "  }\n"
        "}\n";
    cs.setCSGroupSize(ctaSize, ctaSize);
    cs.setCSSharedMemory(ctaSize * ctaSize * 16);
    Program *pgm1 = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm1, cs)) {
        LWNFailTest();
        return;
    }

    // Set up a dummy program to display the results on-screen.
    VertexShader vs(430);
    vs <<
        "in vec2 pos;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  tc = pos * 0.5 + 0.5;\n"
        "}\n";
    FragmentShader fs(430);
    fs <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, tc);\n"
        "}\n";
    Program *displayProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(displayProgram, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Sampler used to display the images on-screen.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *smp = sb.CreateSampler();

    // Allocate a texture for each cell to be used for image loads/stores and
    // texture lookups.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFlags(TextureFlags::IMAGE);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texSize, texSize);
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(1);
    LWNsizeiptr texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, nCells * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    Texture *textures[nCells];
    TextureHandle texHandles[nCells];
    ImageHandle imageHandles[nCells];
    for (int i = 0; i < nCells; i++) {
        textures[i] = texAllocator.allocTexture(&tb);
        LWNuint textureID = textures[i]->GetRegisteredTextureID();
        LWNuint imageID = g_lwnTexIDPool->RegisterImage(textures[i]);
        texHandles[i] = device->GetTextureHandle(textureID, smp->GetRegisteredID());
        imageHandles[i] = device->GetImageHandle(imageID);
    }

    // Set up a uniform buffer to hold randomly-generated "object" data.
    LWNsizeiptr uboSize = nObjects * 8 * sizeof(LWNfloat);
    uboSize = 256 * ((uboSize + 255) / 256);
    MemoryPoolAllocator uboAllocator(device, NULL, uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    LWNfloat *uboMem = (LWNfloat *) ubo->Map();
    for (int i = 0; i < nObjects; i++) {
        uboMem[8*i+0] = lwFloatRand(0, texSize);
        uboMem[8*i+1] = lwFloatRand(0, texSize);
        uboMem[8*i+2] = 1.0 / lwFloatRand(texSize/16.0, texSize/4.0);
        uboMem[8*i+3] = lwFloatRand(0, 100.0);
        for (int j = 4; j < 8; j++) {
            uboMem[8*i+j] = lwFloatRand(0.3, 1.0);
        }
    }

    // Set up a vertex buffer to display primitives on-screen.
    struct Vertex {
        dt::vec2 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0) },
        { dt::vec2(-1.0, +1.0) },
        { dt::vec2(+1.0, -1.0) },
        { dt::vec2(+1.0, +1.0) },
    };
    LWNsizeiptr vboSize = sizeof(vertexData);
    MemoryPoolAllocator vboAllocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.ClearColor(0, 0.0, 0.2, 0.0, 0.0);

    // Set up and run the "cones" program and generate random
    // positions/sizes/colors.
    queueCB.BindProgram(pgm0, ShaderStageBits::COMPUTE);
    queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr, uboSize);
    queueCB.BindImage(ShaderStage::COMPUTE, 0, imageHandles[0]);
    queueCB.DispatchCompute(texSize/ctaSize, texSize/ctaSize, 1);

    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ILWALIDATE_SHADER);

    // Set up and run the edge detection shader, which will use the texture
    // generated by the first shader as an input.
    queueCB.BindProgram(pgm1, ShaderStageBits::COMPUTE);
    queueCB.BindTexture(ShaderStage::COMPUTE, 0, texHandles[0]);
    queueCB.BindImage(ShaderStage::COMPUTE, 0, imageHandles[1]);
    queueCB.DispatchCompute((texSize+ctaSize-3)/(ctaSize-2), (texSize+ctaSize-3)/(ctaSize-2), 1);

    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_TEXTURE | BarrierBits::ILWALIDATE_SHADER);

    // Display the cells to different corners of the window by running our
    // display program.
    assert(nCells == 2);
    queueCB.BindProgram(displayProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);
    queueCB.BindVertexArrayState(vertex);
    for (int i = 0; i < 2; i++) {
        int dstx = i ? (lwrrentWindowWidth-8)-texSize : 8;
        int dsty = i ? (lwrrentWindowHeight-8)-texSize : 8;
        queueCB.SetViewportScissor(dstx, dsty, texSize, texSize);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandles[i]);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNBasicComputeTest, lwn_04compute, );

//////////////////////////////////////////////////////////////////////////

class LWNComputeIndirectTest {

private:
    static const int ctaSize = 8;           // run with 8x8 work groups
    static const int cellSize = 64;         // 64x64 cells to check test results
    static const int cellMargin = 4;

    // Data structure used in SSBOs for testing compute indirect data.  The
    // shader reads from one structure and writes to another, where it:
    //
    // - Dispatch uses <gpuDispatchSize> or <cpuDispatchSize> in the first
    //   SSBO as the indirect data, depending on whether the indirect data 
    //   are written by the CPU or GPU.
    //
    // - <cpuDispatchSize> in the first SSBO will hold the expected dispatch
    //   size even if the data are produced on the GPU.
    //
    // - Copies <nextDispatchSize> from the first SSBO to the <gpuDispatchSize>
    //   entry for the second SSBO.
    //
    // - Counts the number of ilwocations in the dispatch using atomic adds
    //   in <counter> in the second SSBO.
    //
    // - Checks <cpuDispatchSize> from the first SSBO against gl_NumWorkGroups,
    //   to verify the builtin works.  Writes 1 (match) or 0 (no match) into
    //   <checkBuiltin> in the second SSBO.
    //
    // - <cpuDispatch> in the first SSBO indicates whether <cpuDispatchSize>
    //   should be used for the actual dispatch.
    //
    struct ComputeInfo {
        dt::uvec4 gpuDispatchSize;
        dt::uvec4 cpuDispatchSize;
        dt::uvec4 nextDispatchSize;
        unsigned int counter;
        unsigned int checkBuiltin;
        unsigned int cpuDispatch;
        unsigned int dummy2;
    };
    ct_assert(sizeof(ComputeInfo) == 64);


    // Data structure describing a specific compute dispatch, including its
    // expected size, and whether the size should come from values written by
    // the CPU or GPU.
    struct ComputeJob {
        int x, y, z;
        bool cpuDispatch;
    };

    void initDispatch(ComputeInfo *info, const ComputeJob *job);
    bool checkResults(const ComputeInfo *src);

public:
    OGTEST_CppMethods();
};

int LWNComputeIndirectTest::isSupported()
{
    return lwogCheckLWNAPIVersion(38, 9);
}

lwString LWNComputeIndirectTest::getDescription()
{
    lwStringBuf sb;

    sb <<
        "Basic test of LWN indirect compute dispatches.  This test sets up "
        "properties of the dispatch in one SSBO and has the dispatch write some "
        "values to SSBO memory.  At the end, we check each dispatch to see if "
        "correct values are written and display red/green.  This test exercises "
        "cases where indirect dispatch data are written by the CPU and cases "
        "where they are written by the GPU (where INDIRECT_DATA barriers are "
        "required).";

    return sb.str();
}

void LWNComputeIndirectTest::initGraphics(void)
{
    lwnDefaultInitGraphics();
}

void LWNComputeIndirectTest::exitGraphics(void)
{
    lwnDefaultExitGraphics();
}

void LWNComputeIndirectTest::initDispatch(ComputeInfo *info, const ComputeJob *job)
{
    // Set up the info for this dispatch.  <gpuDispatchSize> will be
    // overwritten by by the previous job (if any).  <nextDispatchSize> should
    // come from the next job and will be used to fill in its
    // <gpuDispatchSize>.
    const ComputeJob *nextJob = job + 1;
    info->gpuDispatchSize = dt::uvec4(0);
    info->cpuDispatchSize = dt::uvec4(job->x, job->y, job->z, 0);
    info->nextDispatchSize = dt::uvec4(nextJob->x, nextJob->y, nextJob->z, 0);
    info->counter = 0;
    info->checkBuiltin = 0;
    info->cpuDispatch = job->cpuDispatch ? 1 : 0;
}

bool LWNComputeIndirectTest::checkResults(const ComputeInfo *src)
{
    const ComputeInfo *next = src + 1;

    // The <gpuDispatchSize> entry in the next job's slot should be written
    // from our <nextDispatchSize>.
    if (all(next->gpuDispatchSize != src->nextDispatchSize)) {
        return false;
    }

    // Our <counter> value should count all the threads exelwted.  (Use the
    // CPU dispatch size to get the expected value.)
    if (src->counter != (src->cpuDispatchSize[0] * src->cpuDispatchSize[1] *
                         src->cpuDispatchSize[2] * ctaSize * ctaSize))
    {
        return false;
    }

    // The shader code should write a passing result when checking the
    // dispatch size built-in against our expected (CPU) size.
    if (src->checkBuiltin != 1) {
        return false;
    }

    return true;
}

class QueueMemoryCompare {
public:

    explicit QueueMemoryCompare(Queue* queue) : m_cmdMem(0), m_ctrlMem(0), m_compMem(0) {
        m_cmdMem  = queue->GetTotalCommandMemoryUsed();
        m_ctrlMem = queue->GetTotalControlMemoryUsed();
        m_compMem = queue->GetTotalComputeMemoryUsed();
    }

    bool operator >= (const QueueMemoryCompare& rhs) const {
        return ((m_cmdMem >= rhs.m_cmdMem) && (m_ctrlMem >= rhs.m_ctrlMem) && (m_compMem >= rhs.m_compMem));
    }
    bool operator <= (const QueueMemoryCompare& rhs) const {
        return ((m_cmdMem <= rhs.m_cmdMem) && (m_ctrlMem <= rhs.m_ctrlMem) && (m_compMem <= rhs.m_compMem));
    }
    bool isZero() const {
        return ((m_cmdMem == 0) && (m_ctrlMem == 0) && (m_compMem == 0));
    }
    size_t getCmdMem()  const { return m_cmdMem;  }
    size_t getCtrlMem() const { return m_ctrlMem; }
    size_t getCompMem() const { return m_compMem; }

private:
    size_t m_cmdMem;
    size_t m_ctrlMem;
    size_t m_compMem;
};

void LWNComputeIndirectTest::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Get memory used at beginning of the test
    QueueMemoryCompare memBegin(queue);

    static const int cellsX = lwrrentWindowWidth / cellSize;
    static const int cellsY = lwrrentWindowHeight / cellSize;
    CellIterator2D cell(cellsX, cellsY);

    // Our compute shader has an array of two SSBOs bound -- the first for the
    // current job and the second for the next job.
    lwShader cs = ComputeShader(430);
    cs <<
        "layout(std430, binding = 0) buffer SSBO {\n"
        "  uvec4 gpuDispatchSize;\n"
        "  uvec4 cpuDispatchSize;\n"
        "  uvec4 nextDispatchSize;\n"
        "  uint counter;\n"
        "  uint checkBuiltin;\n"
        "  uint cpuDispatch;\n"
        "  uint dummy;\n"
        "} buffers[2];\n"
        "void main() {\n"

        // Write the next dispatch size into the <gpuDispatchSize> slot for
        // the next job.
        "  buffers[1].gpuDispatchSize = buffers[0].nextDispatchSize;\n"

        // Count all the threads for this job.
        "  atomicAdd(buffers[0].counter, 1);\n"

        // Check the gl_NumWorkGroups built-in against the expected (CPU)
        // dispatch size for this job, and write 1/0 in the check output.
        "  if (all(equal(buffers[0].cpuDispatchSize.xyz, gl_NumWorkGroups))) {\n"
        "    buffers[0].checkBuiltin = 1;\n"
       "  } else {\n"
       "    buffers[0].checkBuiltin = 0;\n"
       "  }\n"
        "}\n";
    cs.setCSGroupSize(ctaSize, ctaSize);
    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, cs)) {
        LWNFailTest();
        return;
    }

    // Set up an SSBO to hold ComputeInfo structures for a full set of jobs.
    const int ssboSize = 64 * 1024;
    MemoryPoolAllocator ssboAllocator(device, NULL, ssboSize, LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *ssbo = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    ComputeInfo *computePtr = (ComputeInfo *) ssbo->Map();
    BufferAddress computeGpuAddr = ssbo->GetAddress();

    // Test a variety of indirect dispatch sizes, with some sizes coming from
    // CPU-written values and others from GPU-written values.  We leave a
    // dummy entry at the end of the array so the last job can look at it when
    // setting up its <nextDispatchSize> value.
    ComputeJob jobs[] = {
        {  1, 1, 1, true  },
        {  2, 2, 2, true  },
        {  4, 3, 7, true  },
        { 34, 2, 1, false },
        {  8, 6, 1, false },
        {  1, 9, 7, false },
        {  0, 0, 0, false },
    };
    LWNuint nJobs = __GL_ARRAYSIZE(jobs) - 1;

    // Use the CPU to initialize job descriptors for all jobs.
    for (LWNuint i = 0; i < nJobs; i++) {
        initDispatch(computePtr + i, jobs + i);
    }

    // Flush the CPU caches to make sure the GPU sees what we wrote before we
    // start work.
    ssbo->FlushMappedRange(0, ssboSize);

    queueCB.BindProgram(pgm, ShaderStageBits::COMPUTE);
    for (LWNuint i = 0; i < nJobs; i++) {
        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 0, computeGpuAddr + i * sizeof(ComputeInfo), sizeof(ComputeInfo));
        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, 1, computeGpuAddr + (i+1) * sizeof(ComputeInfo), sizeof(ComputeInfo));
        if (jobs[i].cpuDispatch) {
            // When using CPU-written data, no synchronization is required,
            // and we get the data from the <cpuDispatchSize> slot.
            queueCB.DispatchComputeIndirect(computeGpuAddr + i * sizeof(ComputeInfo) +
                                            offsetof(ComputeInfo, cpuDispatchSize));
        } else {
            // When using CPU-written data, we need to wait for the previous
            // compute work (which writes the dispatch size), and pull the
            // data from the <gpuDispatchSize> slot.
            queueCB.Barrier(BarrierBits::ORDER_INDIRECT_DATA);
            queueCB.DispatchComputeIndirect(computeGpuAddr + i * sizeof(ComputeInfo) +
                                            offsetof(ComputeInfo, gpuDispatchSize));
        }
    }

    // Wait for the GPU to finish, and then ilwalidate CPU caches to make sure
    // we can read what the GPU wrote.
    queueCB.submit();
    queue->Finish();

    // Get memory used after compute dispatch
    QueueMemoryCompare memAfterCompute(queue);

    ssbo->IlwalidateMappedRange(0, ssboSize);

    // Loop over all the jobs, check the results, and clear a cell to either
    // green or red.
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    for (LWNuint i = 0; i < nJobs; i++) {
        queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
                                   cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
        if (checkResults(computePtr + i)) {
            queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
        } else {
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        }
        cell++;
    }
    queueCB.submit();

    // Get memory used after graphics command
    QueueMemoryCompare memAfterGfx(queue);

    queue->ResetMemoryUsageCounts();
    // Get memory used after memory usage counters reset. Make sure that we have no
    // unflushed commands at this point since we expect the used memory to be zero
    // after the reset.
    QueueMemoryCompare memAfterReset(queue);

    queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
                               cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);

    cell++;

    // Basic check if memory usage values are reasonable. We expect that the values of used
    // memory are increasing and that the graphics command do not use compute memory
    if ((memBegin <= memAfterCompute) && (memAfterCompute <= memAfterGfx) && memAfterReset.isZero() &&
        (memAfterCompute.getCompMem() == memAfterGfx.getCompMem())) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNComputeIndirectTest, lwn_compute_indirect, );
