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
#include "lwn_basic.h"
#include <time.h>

static struct LWNSampleTestCPPInterface g_lwnSampleTestCPPInterface;
LWNSampleTestCPPInterface *LWNSampleTestConfig::m_cpp_interface = &g_lwnSampleTestCPPInterface;

using namespace lwn;

// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_BASIC_DO_PRINTF is set to 1.
#define LWN_BASIC_DO_PRINTF     0
static void log_output(const char *fmt, ...)
{
#if LWN_BASIC_DO_PRINTF
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}

//////////////////////////////////////////////////////////////////////////

static const char *vsstring = 
    "layout(location = 0) in vec4 position;\n"
    "layout(location = 1) in vec4 tc;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "};\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  gl_Position = position*scale;\n"
    // This line exists to trick the compiler into putting a value in the compiler
    // constant bank, so we can exercise binding that bank
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      gl_Position = vec4(0,0,0,0);\n"
    "  }\n"
    "  ftc = tc;\n"
    "}\n";

static const char *vsstring_bindless =
    "layout(location = 0) in vec4 position;\n"
    "layout(location = 1) in vec4 tc;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    sampler2D bindlessTex;\n"
    "};\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  gl_Position = position*scale;\n"
    // This line exists to trick the compiler into putting a value in the compiler
    // constant bank, so we can exercise binding that bank
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      gl_Position = vec4(0,0,0,0);\n"
    "  }\n"
    "  ftc = tc;\n"
    "}\n";

// Version of the fragment shader using bound textures.
static const char *fsstring = 
    "layout(binding = 0) uniform sampler2D boundTex;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "};\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 ftc; };\n"
    "void main() {\n"
    "  color = texture(boundTex, ftc.xy);\n"
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      color = vec4(0,0,0,0);\n"
    "  }\n"
    "}\n";

// Version of the fragment shader using bindless textures.
static const char *fsstring_bindless =
    // No non-block sampler uniform <tex>.
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    sampler2D bindlessTex;\n"
    "};\n"
    "layout(location = 0) out vec4 color;\n"
    "in IO { vec4 ftc; };\n"
    "void main() {\n"
    "  color = texture(sampler2D(bindlessTex), ftc.xy);\n"
    "  if (scale.z != 1.0 + 1.0/65536.0) {\n"
    "      color = vec4(0,0,0,0);\n"
    "  }\n"
    "}\n";

// Optional geometry shader (disabled by default) to flip our triangles upside down
// and double the frequency of the texture coordinates.
static const char *gsstring =
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    gl_Position = gl_in[i].gl_Position * vec4(1.0, -1.0, 1.0, 1.0);\n"
    "    ftc = vi[i].ftc * 2.0;\n"
    "    EmitVertex();\n"
    "  }\n"
    "}\n";

// Optional tessellation control and evaluation shaders (disabled by default).
// The tessellation control shader swaps X and Y coordinates, doubles the
// frequency of the texture coordinates, and sets a tessellation LOD of 4.
// The tessellation evaluation shader interpolates position and tessellation
// coordinate according to the tessellation coordinate, doubles the frequency
// of the texture coordinates, and increases the triangle sizes by 20%.
static const char *tcsstring =
    "#define iid gl_IlwocationID\n"
    "layout(vertices=3) out;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; } vo[];\n"
    "void main() {\n"
    "  gl_out[iid].gl_Position = gl_in[iid].gl_Position.yxzw;\n"
    "  vo[iid].ftc = vi[iid].ftc * 2.0;\n"
    "  gl_TessLevelOuter[0] = 4.0;\n"
    "  gl_TessLevelOuter[1] = 4.0;\n"
    "  gl_TessLevelOuter[2] = 4.0;\n"
    "  gl_TessLevelInner[0] = 4.0;\n"
    "}\n";

static const char *tesstring =
    "layout(triangles) in;\n"
    "in IO { vec4 ftc; } vi[];\n"
    "out IO { vec4 ftc; };\n"
    "void main() {\n"
    "  gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x + \n"
    "                 gl_in[1].gl_Position * gl_TessCoord.y + \n"
    "                 gl_in[2].gl_Position * gl_TessCoord.z);\n"
    "  gl_Position.xy *= 1.2;\n"
    "  ftc = 2.0 * (vi[0].ftc * gl_TessCoord.x +\n"
    "               vi[1].ftc * gl_TessCoord.y +\n"
    "               vi[2].ftc * gl_TessCoord.z);\n"
    "}\n";


// Two triangles that intersect
static LWNfloat vertexData[] = {-0.5f, -0.5f, 0.5f, 
                                0.5f, -0.5f,  0.5f,
                                -0.5f, 0.5f,  0.5f,

                                0.5f, -0.5f, 0.5f,
                                -0.5f, -0.5f, 0.3f,
                                0.5f, 0.5f, 0.9f};

// Simple 0/1 texcoords in rgba8 format (used to be color data)
static uint8_t texcoordData[] = {0, 0, 0xFF, 0xFF,
                                 0xFF, 0, 0, 0xFF,
                                 0, 0xFF, 0, 0xFF,

                                 0, 0, 0xFF, 0xFF,
                                 0xFF, 0, 0, 0xFF,
                                 0, 0xFF, 0, 0xFF};

static int offscreenWidth = 640, offscreenHeight = 480;
static char *offscreenImage = NULL;

static ShaderStageBits allShaderStages = (ShaderStageBits::VERTEX | ShaderStageBits::TESS_CONTROL |
                                          ShaderStageBits::TESS_EVALUATION | ShaderStageBits::GEOMETRY |
                                          ShaderStageBits::FRAGMENT);

static LWNsizeiptr poolSize = 0x1000000UL; // 16MB pool size
static LWNsizeiptr coherentPoolSize = 0x100000UL; // 1MB pool size
static LWNsizeiptr programPoolSize = 0x100000UL; // 1MB pool size

typedef struct {
    float scale[4];
    LWNtextureHandle bindlessTex;
} UniformBlock;

static void LWNAPIENTRY debugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                                      LWNdebugCallbackSeverity severity, LWNstring message, void *userParam)
{
    log_output("Debug callback:\n");
    log_output("  source:       0x%08x\n", source);
    log_output("  type:         0x%08x\n", type);
    log_output("  id:           0x%08x\n", id);
    log_output("  severity:     0x%08x\n", severity);
    log_output("  userParam:    0x%08x%08x\n",
               uint32_t(uint64_t(uintptr_t(userParam)) >> 32), uint32_t(uintptr_t(userParam)));
    log_output("  message:\n    %s\n", message);
}

void LWNSampleTestConfig::cppDisplay()
{
    Device *device = m_cpp_interface->device;
    Queue *queue = m_cpp_interface->queue;
    CommandBuffer *queueCB = m_cpp_interface->queueCB;
    CommandBufferMemoryManager *cmdMem = m_cpp_interface->cmdMemMgr;
    CommandHandle queueCBHandle;
    BufferBuilder bufferBuilder;
    TextureBuilder textureBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    textureBuilder.SetDevice(device).SetDefaults();

    // Create a GLSLC helper instead of using the global version
    // since the device may be a debug device depending on the
    // test variant.
    lwnTest::GLSLCHelper glslcHelper(device, programPoolSize, g_glslcLibraryHelper, g_glslcHelperCache);
    MemoryPool *scratchMemPool = device->CreateMemoryPool(NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, MemoryPoolType::GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, NULL);

    // For "-debug", call generateDebugMessages() to generate some stupid
    // errors to exercise the callbacks.  We use C instead of C++ code here
    // because it's easier to create error conditions.
    if (m_debug) {
        lwnDeviceInstallDebugCallback(reinterpret_cast<LWNdevice *>(device), debugCallback, (void *)0x8675309, LWN_TRUE);
        generateDebugMessages();
    }

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator coherent_allocator(device, NULL, coherentPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Create programs from the device, provide them shader code and compile/link them
    Program *pgm = device->CreateProgram();

    // XXX This is a hack because we don't have an IL. I'm just jamming through the strings 
    // as if they were an IL blob
    ShaderStage stages[5];
    const char *sources[5];
    int nSources = 2;

    VertexShader vs(440);
    FragmentShader fs(440);
    GeometryShader gs(440);
    TessControlShader tcs(440);
    TessEvaluationShader tes(440);

    sources[0] = m_bindless ? vsstring_bindless : vsstring;
    stages[0] = ShaderStage::VERTEX;
    sources[1] = m_bindless ? fsstring_bindless : fsstring;
    stages[1] = ShaderStage::FRAGMENT;

    vs.addExtension(lwShaderExtension::LW_gpu_shader5);
    fs.addExtension(lwShaderExtension::LW_gpu_shader5);
    gs.addExtension(lwShaderExtension::LW_gpu_shader5);
    tcs.addExtension(lwShaderExtension::LW_gpu_shader5);
    tes.addExtension(lwShaderExtension::LW_gpu_shader5);

    if (m_bindless) {
        // Required for using sampler2D in a UBO
        vs.addExtension(lwShaderExtension::LW_bindless_texture);
        fs.addExtension(lwShaderExtension::LW_bindless_texture);
    }

    vs << sources[0];
    fs << sources[1];

    if (m_geometryShader) {
        sources[nSources] = gsstring;
        stages[nSources] = ShaderStage::GEOMETRY;
        gs << sources[nSources];
        nSources++;
    }
    if (m_tessControlShader) {
        sources[nSources] = tcsstring;
        stages[nSources] = ShaderStage::TESS_CONTROL;
        tcs << sources[nSources];
        nSources++;
    }
    if (m_tessEvalShader) {
        sources[nSources] = tesstring;
        stages[nSources] = ShaderStage::TESS_EVALUATION;
        tes << sources[nSources];
        nSources++;
    }

    if (!glslcHelper.CompileAndSetShaders(pgm, vs, fs,
                            m_geometryShader ? gs : Shader(),
                            m_tessControlShader ? tcs : Shader(),
                            m_tessEvalShader ? tes : Shader())) {
        log_output("Shader compile error. infoLog =\n%s\n", glslcHelper.GetInfoLog());
    }

    // Check to make sure program interfaces work as expected.
    LWNint location;
    location = glslcHelper.ProgramGetResourceLocation(pgm, ShaderStage::VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in vertex.\n", location);
    }
    location = glslcHelper.ProgramGetResourceLocation(pgm, ShaderStage::FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in fragment.\n", location);
    }
    if (!m_bindless) {
        location = glslcHelper.ProgramGetResourceLocation(pgm, ShaderStage::FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_SAMPLER, "boundTex");
        if (location != 0) {
            log_output("tex has location %d in fragment.\n", location);
        }
    }

    // Create new state vectors
    BlendState blend;
    ChannelMaskState cmask;
    ColorState color;
    DepthStencilState depth;
    MultisampleState multisample;
    PolygonState polygon;
    VertexAttribState attribState[2];
    VertexStreamState streamState[2];

    blend.SetDefaults();

    cmask.SetDefaults();

    color.SetDefaults();

    depth.SetDefaults();
    depth.SetDepthTestEnable(LWN_TRUE);
    depth.SetDepthWriteEnable(LWN_TRUE);
    depth.SetDepthFunc(DepthFunc::LESS);

    multisample.SetDefaults();

    polygon.SetDefaults();
    if (m_wireframe) {
        polygon.SetPolygonMode(PolygonMode::LINE);
    } else {
        polygon.SetPolygonMode(PolygonMode::FILL);
    }
    //polygon.SetDiscardEnable(LWN_TRUE);

    //lwnColorBlendEnable(color, /*MRT index*/0, LWN_TRUE);
    //lwnColorBlendFunc(color, /*MRT index*/0, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE);
    //lwnColorMask(color, /*MRT index*/0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
    //lwnColorLogicOpEnable(color, LWN_TRUE);
    //lwnColorLogicOp(color, LWN_LOGIC_OP_XOR);

    // Set the state vector to use two vertex attributes.
    //
    // Interleaved pos+color
    // position = attrib 0 = 3*float at relativeoffset 0
    // texcoord = attrib 1 = rgba8 at relativeoffset 0
    attribState[0].SetDefaults().SetFormat(Format::RGB32F, 0).SetStreamIndex(0);
    attribState[1].SetDefaults().SetFormat(Format::RGBA8, 0).SetStreamIndex(1);
    streamState[0].SetDefaults().SetStride(12);
    streamState[1].SetDefaults().SetStride(4);

    // Create a vertex buffer and fill it with data
    bufferBuilder.SetDefaults();

    Buffer *vbo = coherent_allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData) + sizeof(texcoordData));

    // create persistent mapping
    void *ptr = vbo->Map();
    // fill ptr with vertex data followed by color data
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));

    // Get a handle to be used for setting the buffer as a vertex buffer
    BufferAddress vboAddr = vbo->GetAddress();

    // Create an index buffer and fill it with data
    unsigned short indexData[6] = {0, 1, 2, 3, 4, 5};
    Buffer *ibo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, indexData, sizeof(indexData),
                                     BUFFER_ALIGN_INDEX_BIT, true);

    // Get a handle to be used for setting the buffer as an index buffer
    BufferAddress iboAddr = ibo->GetAddress();

    Texture *rtTex = NULL;
    Texture *depthTex = NULL;
    Texture *tex1x = NULL;

    textureBuilder.SetDefaults();
    textureBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    textureBuilder.SetSize2D(offscreenWidth, offscreenHeight);
    textureBuilder.SetTarget(TextureTarget::TARGET_2D);
    textureBuilder.SetFormat(Format::RGBA8);

    if (m_multisample) {
        tex1x = allocator.allocTexture(&textureBuilder);

        textureBuilder.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
        textureBuilder.SetSamples(4);

        multisample.SetSamples(4);
    }

    rtTex = allocator.allocTexture(&textureBuilder);

    textureBuilder.SetFormat(Format::DEPTH24_STENCIL8);
    depthTex = allocator.allocTexture(&textureBuilder);

    queueCB->SetRenderTargets(1, &rtTex, NULL, depthTex, NULL);

    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(device).SetDefaults();
    // Commented out experiments to test different state settings.    
    // samplerBuilder.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = samplerBuilder.CreateSampler();
    LWNuint samplerID = sampler->GetRegisteredID();

    const int texWidth = 4, texHeight = 4;
    textureBuilder.SetDefaults().SetTarget(TextureTarget::TARGET_2D).
        SetFormat(Format::RGBA8).SetSize2D(texWidth, texHeight);
    Texture *texture = allocator.allocTexture(&textureBuilder);
    LWNuint textureID = texture->GetRegisteredTextureID();

    // Bindless requires a combined texture/sampler handle.  Bound will
    // separately bind texture and sampler handles.
    TextureHandle texHandle = device->GetTextureHandle(textureID, samplerID);

    bufferBuilder.SetDefaults();
    Buffer *pbo = coherent_allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texWidth*texHeight*4);
    BufferAddress pboAddr = pbo->GetAddress();
    unsigned char *texdata = (unsigned char *) pbo->Map();
    // fill with texture data
    for (int j = 0; j < texWidth; ++j) {
        for (int i = 0; i < texHeight; ++i) {
            texdata[4*(j*texWidth+i)+0] = 0xFF*((i+j)&1);
            texdata[4*(j*texWidth+i)+1] = 0xFF*((i+j)&1);
            texdata[4*(j*texWidth+i)+2] = 0xFF*((i+j)&1);
            texdata[4*(j*texWidth+i)+3] = 0xFF;
        }
    }

    // XXX missing pixelpack object
    // Download the texture data
    CopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
    queueCB->CopyBufferToTexture(pboAddr, texture, NULL, &copyRegion, CopyFlags::NONE);

    // Set up a uniform buffer holding transformation code as well as the texture
    // handle for "-bindless".
    float scale = 1.5f;
    if (m_benchmark) {
        scale = 0.2f;
    }
    UniformBlock uboData;
    uboData.bindlessTex = texHandle;
    uboData.scale[0] = scale;
    uboData.scale[1] = scale;
    uboData.scale[2] = 1.0f + 1.0f / 65536.0;
    uboData.scale[3] = 1.0f;
    Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                     BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    BufferAddress uboAddr = ubo->GetAddress();

    // Some scissored clears
    {
        queueCB->SetScissor(0, 0, offscreenWidth, offscreenHeight);
        LWNfloat clearColor[] = {0, 0, 0, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
        queueCB->ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    }
    {
        queueCB->SetScissor(offscreenWidth/2, 0, offscreenWidth/2, offscreenHeight/2);
        LWNfloat clearColor[] = {0, 0.5, 0, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
    }
    {
        queueCB->SetScissor(0, offscreenHeight/2, offscreenWidth/2, offscreenHeight/2);
        LWNfloat clearColor[] = {0, 0, 0.5, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
    }
    queueCB->SetScissor(0, 0, offscreenWidth, offscreenHeight);
    queueCB->SetViewport(0, 0, offscreenWidth, offscreenHeight);

    if (m_benchmark) {
        queue->Finish();
    }
    clock_t startTime = clock();
    unsigned int numIterations = m_benchmark ? 10000000 : 1;

    DrawPrimitive drawPrimitive = DrawPrimitive::TRIANGLES;
    if (m_tessEvalShader) {
        LWNfloat levels[4] = { 2, 2, 2, 2 };
        drawPrimitive = DrawPrimitive::PATCHES;
        queueCB->SetPatchSize(3);
        queueCB->SetInnerTessellationLevels(levels);
        queueCB->SetOuterTessellationLevels(levels);
    }

    if (m_submitMode == QUEUE) {
        for (unsigned int i = 0; i < numIterations; ++i) {
            // Bind the program, vertex state, and any required control structures.
            queueCB->BindProgram(pgm, allShaderStages);
            queueCB->BindVertexAttribState(2, attribState);
            queueCB->BindVertexStreamState(2, streamState);
            queueCB->BindBlendState(&blend);
            queueCB->BindChannelMaskState(&cmask);
            queueCB->BindColorState(&color);
            queueCB->BindDepthStencilState(&depth);
            queueCB->BindMultisampleState(&multisample);
            queueCB->BindPolygonState(&polygon);
            queueCB->SetSampleMask(~0);
            queueCB->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            queueCB->BindVertexBuffer(1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
            queueCB->BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(uboData));
            queueCB->BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));
            if (!m_bindless) {
                queueCB->BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
            }
            queueCB->DrawElements(drawPrimitive, IndexType::UNSIGNED_SHORT, 6, iboAddr);
        }
    }

    // Flush the queue command buffer and recycle for future use (if needed).
    queueCBHandle = queueCB->EndRecording();
    queue->SubmitCommands(1, &queueCBHandle);
    queueCB->BeginRecording();

    if (m_submitMode == COMMAND) {
        const int N = 4;
        CommandBuffer *cmd[N];
        CommandHandle cmdHandle[N];
        for (int i = 0; i < N; ++i) {
            cmd[i] = device->CreateCommandBuffer();
            cmdMem->populateCommandBuffer(cmd[i], CommandBufferMemoryManager::Coherent);
            cmd[i]->BeginRecording();
            // Bind the program, vertex state, and any required control structures.
            cmd[i]->BindProgram(pgm, allShaderStages);
            cmd[i]->BindVertexAttribState(2, attribState);
            cmd[i]->BindVertexStreamState(2, streamState);
            cmd[i]->BindBlendState(&blend);
            cmd[i]->BindChannelMaskState(&cmask);
            cmd[i]->BindColorState(&color);
            cmd[i]->BindDepthStencilState(&depth);
            cmd[i]->BindMultisampleState(&multisample);
            cmd[i]->BindPolygonState(&polygon);
            cmd[i]->SetSampleMask(~0);
            cmd[i]->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            cmd[i]->BindVertexBuffer(1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
            cmd[i]->BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(uboData));
            cmd[i]->BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));
            if (!m_bindless) {
                cmd[i]->BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
            }
            cmd[i]->DrawElements(drawPrimitive, IndexType::UNSIGNED_SHORT, 6, iboAddr);
            cmdHandle[i] = cmd[i]->EndRecording();
        }
        for (unsigned int j = 0; j < numIterations; j += N) {
            queue->SubmitCommands(N, &cmdHandle[0]);
        }
        for (int i = 0; i < N; ++i) {
            cmd[i]->Free();
        }
    } else if (m_submitMode == COMMAND_TRANSIENT) {
        const int N = 4;
        CommandBuffer *cmd[N];
        CommandHandle cmdHandle[N];
        for (unsigned int j = 0; j < numIterations; j += N) {
            for (int i = 0; i < N; ++i) {
                cmd[i] = device->CreateCommandBuffer();
                cmdMem->populateCommandBuffer(cmd[i], CommandBufferMemoryManager::Coherent);
                cmd[i]->BeginRecording();
                // Bind the program, vertex state, and any required control structures.
                cmd[i]->BindProgram(pgm, allShaderStages);
                cmd[i]->BindVertexAttribState(2, attribState);
                cmd[i]->BindVertexStreamState(2, streamState);
                cmd[i]->BindBlendState(&blend);
                cmd[i]->BindChannelMaskState(&cmask);
                cmd[i]->BindColorState(&color);
                cmd[i]->BindDepthStencilState(&depth);
                cmd[i]->BindMultisampleState(&multisample);
                cmd[i]->BindPolygonState(&polygon);
                cmd[i]->SetSampleMask(~0);
                cmd[i]->BindVertexBuffer(0, vboAddr, sizeof(vertexData));
                cmd[i]->BindVertexBuffer(1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
                cmd[i]->BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, sizeof(uboData));
                cmd[i]->BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(uboData));
                if (!m_bindless) {
                    cmd[i]->BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
                }
                cmd[i]->DrawElements(drawPrimitive, IndexType::UNSIGNED_SHORT, 6, iboAddr);
                cmdHandle[i] = cmd[i]->EndRecording();
            }
            queue->SubmitCommands(N, &cmdHandle[0]);
            for (int i = 0; i < N; ++i) {
                cmd[i]->Free();
            }
        }
    }

    if (m_multisample) {
        queueCB->Downsample(rtTex, tex1x);

        // The contents of the lwrrently bound depth and colors are
        // not going to be needed after the Downsample operation.  The
        // below discard operations allow the GPU to not write the
        // corresponding GPU L2 cachelines into main memory.  This is
        // a potential memory bandwidth optimization.
        queueCB->DiscardColor(0);
        queueCB->DiscardDepthStencil();
    }

    queueCBHandle = queueCB->EndRecording();
    queue->SubmitCommands(1, &queueCBHandle);
    queueCB->BeginRecording();

    if (m_benchmark) {
        queue->Finish();
        clock_t lwrrentTime = clock();
        log_output("%f\n", 1.0f*numIterations*CLOCKS_PER_SEC/(lwrrentTime - startTime));

        // Inhibit unused variable warnings if output logging is disabled.
        (void)startTime;
        (void)lwrrentTime;
    }

#if 0
    if (m_multisample) {
        queue.PresentTextureSync(tex1x, NULL);
    } else {
        queue.PresentTextureSync(rtTex, NULL);
    }
    queue.Finish();
#else
    // Copy the resulting image back to system memory.
    int offscreenSize = offscreenWidth * offscreenHeight * 4;
    if (!offscreenImage) {
        offscreenImage = new char[offscreenSize];
    }
    if (offscreenImage)
    {
        queue->Finish();
        Texture *texture = m_multisample ? tex1x : rtTex;
        ReadTextureDataRGBA8(device, queue, queueCB, texture, offscreenWidth, offscreenHeight, offscreenImage);
    }
#endif

    pgm->Free();
    coherent_allocator.freeBuffer(vbo);
    coherent_allocator.freeBuffer(ibo);
    coherent_allocator.freeBuffer(pbo);
    coherent_allocator.freeBuffer(ubo);
    allocator.freeTexture(texture);
    allocator.freeTexture(rtTex);
    allocator.freeTexture(depthTex);
    sampler->Free();
    if (m_multisample) {
        allocator.freeTexture(tex1x);
    }
    scratchMemPool->Free();
}

//////////////////////////////////////////////////////////////////////////

class LWNBasicCPPTest : public LWNSampleTestConfig
{
public:
    enum SpecialVariant {
        NORMAL_TEST,
        BINDLESS,
        DEBUG_LAYER,
        MULTISAMPLE,
        GEOMETRY_SHADER,
        TESS_CONTROL_AND_EVALUATION_SHADER,
        TESS_EVALUATION_SHADER,
    };

    OGTEST_CppMethods();

    LWNBasicCPPTest(LWNSampleTestConfig::SubmitMode submitMode, SpecialVariant special)
    {
        m_submitMode = submitMode;
        switch (special) {
        case NORMAL_TEST:
            break;
        case BINDLESS:
            m_bindless = true;
            break;
        case DEBUG_LAYER:
            m_debug = true;
            break;
        case MULTISAMPLE:
            m_multisample = true;
            break;
        case GEOMETRY_SHADER:
            m_geometryShader = true;
            break;
        case TESS_CONTROL_AND_EVALUATION_SHADER:
            m_tessControlShader = true;
            m_tessEvalShader = true;
            m_wireframe = true;
            break;
        case TESS_EVALUATION_SHADER:
            m_tessEvalShader = true;
            m_wireframe = true;
            break;
        default:
            assert(0);
            break;
        }
    }

};

void LWNBasicCPPTest::initGraphics()
{
    lwnDefaultInitGraphics();
    DisableLWNObjectTracking();

    DeviceState *deviceState = DeviceState::GetActive();
    m_cpp_interface->device = deviceState->getDevice();
    m_cpp_interface->queue = deviceState->getQueue();
    m_cpp_interface->queueCB = &(deviceState->getQueueCB());
    m_cpp_interface->cmdMemMgr = &g_lwnCommandMem;
}

void LWNBasicCPPTest::exitGraphics()
{
    g_lwnQueueCB->resetCounters();
    lwnDefaultExitGraphics();
}

int LWNBasicCPPTest::isSupported()
{
    if (m_debug && !g_lwnDeviceCaps.supportsDebugLayer) {
        return 0;
    }
    return lwogCheckLWNAPIVersion(0,1);
}

lwString LWNBasicCPPTest::getDescription()
{
    lwStringBuf sb;
    sb << "Broad but simple coverage of much of the LWN API. Draws two intersecting, ";
    if (m_wireframe) {
        sb << "wireframe, ";
    }
    sb << "depth-tested, textured triangles.\nUses ";

    if (m_debug) {
        sb << "a debug context to catch some errors; test uses ";
    }

    if (m_bindless) {
        sb << "bindless textures; uses ";
    }

    if (m_multisample) {
        sb << "a multisample render target; uses ";
    }

    if (m_geometryShader) {
        sb << "geometry shaders; uses ";
    }

    if (m_tessEvalShader) {
        sb << "tessellation evaluation ";
        if (m_tessControlShader) {
            sb << "and tessellation control shaders; uses ";
        } else {
            sb << "shaders, and sets tessellation control parameters via the LWN API; uses ";
        }
    }

    switch (m_submitMode) {
        case COMMAND_TRANSIENT:
            sb << "transient command buffers";
            break;
        case COMMAND:
            sb << "command buffers";
            break;
        case QUEUE:
            sb << "the queue command buffer";
            break;
    }

    sb << "; test uses the C++ bindings for LWN.";

    return sb.str();
}

void LWNBasicCPPTest::doGraphics()
{
    DeviceState *testDevice = NULL;
    bool initFailed = false;

    // For the debug variant of the test, enable debug support and then create
    // temporary device, queue, and command buffer objects to use for the test
    // run.
    if (m_debug) {
        testDevice =
            new DeviceState(LWNdeviceFlagBits(LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT |
                                              LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_4_BIT));

        if (testDevice && testDevice->isValid()) {
            testDevice->SetActive();

            m_cpp_interface->device = testDevice->getDevice();
            m_cpp_interface->queue = testDevice->getQueue();
            m_cpp_interface->queueCB = testDevice->getQueueCB();
            m_cpp_interface->cmdMemMgr = testDevice->getCmdBufMemoryManager();

            // Program the C interface also, because the debug tests use it
            // when generating debug messages.
            m_c_interface->device = reinterpret_cast<LWNdevice *>(testDevice->getDevice());
            m_c_interface->queue = reinterpret_cast<LWNqueue *>(testDevice->getQueue());
            m_c_interface->queueCB = reinterpret_cast<LWNcommandBuffer *>(&testDevice->getQueueCB());
            m_c_interface->cmdMemMgr = testDevice->getCmdBufMemoryManager();
        } else {
            initFailed = true;
        }
    }

    if (!initFailed) {
        cppDisplay();
    }

    if (m_debug) {
        delete testDevice;
        DeviceState::SetDefaultActive();

        // If our debug initialization failed, display a red window using the
        // window framebuffer class.
        if (initFailed) {
            QueueCommandBuffer &queueCB = DeviceState::GetActive()->getQueueCB();
            g_lwnWindowFramebuffer.setSize(lwrrentWindowWidth, lwrrentWindowHeight);
            g_lwnWindowFramebuffer.bind();
            queueCB.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
            LWNfloat red[] = { 1.0, 0.0, 0.0, 1.0 };
            queueCB.ClearColor(0, red, ClearColorMask::RGBA);
            queueCB.submit();
            g_lwnWindowFramebuffer.present();
        }
    }

    if (!initFailed) {
        // Copy system memory copy of results to the global WindowFramebuffer.
        assert(offscreenWidth == lwrrentWindowWidth);
        assert(offscreenHeight == lwrrentWindowHeight);
        g_lwnWindowFramebuffer.writePixels(offscreenImage);
        if (offscreenImage) {
            delete [] offscreenImage;
            offscreenImage = NULL;
        }
    }
}

// Variants using pipelines with all command transfer mechanisms (queue,
// command buffer, transient command buffer).
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_queue_cpp,   (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::NORMAL_TEST));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_command_cpp, (LWNSampleTestConfig::COMMAND, LWNBasicCPPTest::NORMAL_TEST));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_command_transient_cpp, (LWNSampleTestConfig::COMMAND_TRANSIENT, LWNBasicCPPTest::NORMAL_TEST));

// Special variants using miscellaneous features.
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_bindless_cpp,    (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::BINDLESS));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_debug_cpp,       (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::DEBUG_LAYER));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_geometry_cpp,    (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::GEOMETRY_SHADER));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_multisample_cpp, (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::MULTISAMPLE));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_tess_tcs_cpp,    (LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::TESS_CONTROL_AND_EVALUATION_SHADER));
OGTEST_CppTest(LWNBasicCPPTest, lwn_basic_tess_tesonly_cpp,(LWNSampleTestConfig::QUEUE, LWNBasicCPPTest::TESS_EVALUATION_SHADER));
