/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define LWN_USE_CPP_INTERFACE       1
#define LWN_OVERLOAD_CPP_OBJECTS
#include "lwnexample.h"
using namespace lwn;

#if defined __ANDROID__
#include <android/log.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "LWNTest", __VA_ARGS__))
#define log_output  LOGI
#else
#define log_output  printf
#endif

static Texture *AllocTex(TextureBuilder *builder)
{
    return reinterpret_cast<Texture *>
        (g_texAllocator->allocTexture(reinterpret_cast<LWNtextureBuilder *>(builder)));
}

static void FreeTex(Texture *texture)
{
    g_texAllocator->freeTexture(reinterpret_cast<LWNtexture *>(texture));
}

static Buffer *AllocBuf(BufferBuilder *builder, BufferAlignBits alignBits, size_t size)
{
    return reinterpret_cast<Buffer *>
        (g_bufferAllocator->allocBuffer(reinterpret_cast<LWNbufferBuilder *>(builder),
        alignBits, size));
}

static void FreeBuf(Buffer *buffer)
{
    g_bufferAllocator->freeBuffer(reinterpret_cast<LWNbuffer *>(buffer));
}

//////////////////////////////////////////////////////////////////////////

static const char *vsstring = 
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(location = 0) in vec4 position;\n"
    "layout(location = 1) in vec4 tc;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    uint64_t bindlessTex;\n"
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
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    "layout(binding = 0) uniform sampler2D boundTex;\n"
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    uint64_t bindlessTex;\n"
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
    "#version 440 core\n"
    "#extension GL_LW_gpu_shader5:require\n"
    // No non-block sampler uniform <tex>.
    "layout(binding = 0) uniform Block {\n"
    "    vec4 scale;\n"
    "    uint64_t bindlessTex;\n"
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
    "#version 440 core\n"
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
    "#version 440 core\n"
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
    "#version 440 core\n"
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
static float vertexData[] = {-0.5f, -0.5f, 0.5f, 
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

static int offscreenWidth = 100, offscreenHeight = 100;


static ShaderStageBits allShaderStages = (ShaderStageBits::VERTEX | ShaderStageBits::TESS_CONTROL |
                                          ShaderStageBits::TESS_EVALUATION | ShaderStageBits::GEOMETRY |
                                          ShaderStageBits::FRAGMENT);

typedef struct {
    float scale[4];
    LWNtextureHandle bindlessTex;
} UniformBlock;

static size_t programPoolSize = 0x100000UL; // 1MB pool size

static void LWNAPIENTRY debugCallback(DebugCallbackSource::Enum source, DebugCallbackType::Enum type, int id,
                                      DebugCallbackSeverity::Enum severity, const char *message, void *userParam)
{
    log_output("Debug callback:\n");
    log_output("  source:       0x%08x\n", source);
    log_output("  type:         0x%08x\n", type);
    log_output("  id:           0x%08x\n", id);
    log_output("  severity:     0x%08x\n", severity);
    log_output("  userParam:    0x%08x%08x\n",
               uint32_t(uint64_t(userParam) >> 32), uint32_t(uint64_t(userParam)));
    log_output("  message:\n    %s\n", message);
}

static Buffer *AllocAndFillBufferCPP(Device *device, Queue *queue, CommandBuffer *queueCB,
                                    void *data, int sizeofdata, BufferAlignBits alignBits, bool useCopy)
{
    Buffer *buffer;
    BufferBuilder *bb = device->CreateBufferBuilder();

    if (useCopy) {
        Buffer *tempbo = AllocBuf(bb, BUFFER_ALIGN_COPY_READ_BIT, sizeofdata);
        BufferAddress tempbo_addr = tempbo->GetAddress();
        void *ptr = tempbo->Map();
        memcpy(ptr, data, sizeofdata);

        buffer = AllocBuf(bb, BufferAlignBits(alignBits | BUFFER_ALIGN_COPY_WRITE_BIT), sizeofdata);
        BufferAddress buffer_addr = buffer->GetAddress();

        queueCB->CopyBufferToBuffer(tempbo_addr, buffer_addr, sizeofdata, CopyFlags::NONE);

        // Flush command buffer contents before calling Finish; reopen the
        // command buffer for further recording.
        CommandHandle handle = queueCB->EndRecording();
        queue->SubmitCommands(1, &handle);
        queueCB->BeginRecording();

        queue->Finish();
        FreeBuf(tempbo);
    } else {
        buffer = AllocBuf(bb, alignBits, sizeofdata);
        void *ptr = buffer->Map();
        memcpy(ptr, data, sizeofdata);
    }

    bb->Free();
    return buffer;
}


LWNBasicWindowCPP* LWNSampleTestConfig::cppCreateWindow(LWNnativeWindow nativeWindow, int w, int h)
{
    TextureBuilder textureBuilder;

    LWNBasicWindowCPP *appWindow = new LWNBasicWindowCPP;

    textureBuilder.SetDefaults().SetDevice(m_cpp_interface->device);

    m_windowWidth  = w;
    m_windowHeight = h;

    textureBuilder.SetDefaults();
    textureBuilder.SetFlags(TextureFlags::COMPRESSIBLE | TextureFlags::DISPLAY);
    textureBuilder.SetSize2D(m_windowWidth, m_windowHeight);
    textureBuilder.SetTarget(TextureTarget::TARGET_2D);
    textureBuilder.SetFormat(Format::RGBA8);

    for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
        appWindow->presentTexture[i] = AllocTex(&textureBuilder);
    }

    lwn::objects::WindowBuilder windowBuilder;

    windowBuilder.SetDevice(m_cpp_interface->device);
    windowBuilder.SetDefaults();
    windowBuilder.SetTextures(NUM_PRESENT_TEXTURES, ((lwn::objects::Texture * const *)appWindow->presentTexture));
    windowBuilder.SetNativeWindow(nativeWindow);

    if (!appWindow->win.Initialize(&windowBuilder)) {
        for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
            FreeTex(appWindow->presentTexture[i]);
        }

        delete appWindow;
        return NULL;
    }

    return appWindow;
}

void LWNSampleTestConfig::cppDeleteWindow(void)
{
    if (m_cpp_interface->window) {
        m_cpp_interface->window->win.Finalize();

        for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
            FreeTex(m_cpp_interface->window->presentTexture[i]);
        }

        delete m_cpp_interface->window;
        m_cpp_interface->window = NULL;
    }
}

void LWNSampleTestConfig::cppDisplay()
{
    Device *device = m_cpp_interface->device;
    Queue *queue = m_cpp_interface->queue;
    CommandBuffer *queueCB = m_cpp_interface->queueCB;
    LWNcommandBufferMemoryManager *cmdMem = m_cpp_interface->cmdMemMgr;
    CommandHandle queueCBHandle;
    BufferBuilder bufferBuilder;
    TextureBuilder textureBuilder;
    LWNBasicWindowCPP  *appWin = m_cpp_interface->window;

    bufferBuilder.SetDefaults().SetDevice(device);
    textureBuilder.SetDefaults().SetDevice(device);

    // For "-debug", call generateDebugMessages() to generate some stupid
    // errors to exercise the callbacks.  We use C instead of C++ code here
    // because it's easier to create error conditions.
    if (m_debug) {
        device->InstallDebugCallback(debugCallback, (void *) 0x8675309, LWN_TRUE);
        generateDebugMessages();
    }

    // Create programs from the device, provide them shader code and compile/link them
    Program * pgm = device->CreateProgram();

    // XXX This is a hack because we don't have an IL. I'm just jamming through the strings 
    // as if they were an IL blob
    ShaderStage stages[5];
    const char *sources[5];
    int nSources = 2;
    sources[0] = vsstring;
    stages[0] = ShaderStage::VERTEX;
    sources[1] = m_bindless ? fsstring_bindless : fsstring;
    stages[1] = ShaderStage::FRAGMENT;
    if (m_geometryShader) {
        sources[nSources] = gsstring;
        stages[nSources] = ShaderStage::GEOMETRY;
        nSources++;
    }
    if (m_tessControlShader) {
        sources[nSources] = tcsstring;
        stages[nSources] = ShaderStage::TESS_CONTROL;
        nSources++;
    }
    if (m_tessEvalShader) {
        sources[nSources] = tesstring;
        stages[nSources] = ShaderStage::TESS_EVALUATION;
        nSources++;
    }


    GLSLCHelper glslcHelper(reinterpret_cast<LWNdevice *>(device), programPoolSize, m_cpp_interface->glslcLibraryHelper);

    if (!glslcHelper.CompileAndSetShaders(reinterpret_cast<LWNprogram *>(pgm), reinterpret_cast<LWNshaderStage *>(stages), nSources, sources)) {
        log_output("Shader compile error. infoLog =\n%s\n", glslcHelper.GetInfoLog());
    }

    // Check to make sure program interfaces work as expected.
    int location;
    location = glslcHelper.ProgramGetResourceLocation(reinterpret_cast<LWNprogram *>(pgm), LWN_SHADER_STAGE_VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in vertex.\n", location);
    }
    location = glslcHelper.ProgramGetResourceLocation(reinterpret_cast<LWNprogram *>(pgm), LWN_SHADER_STAGE_FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in fragment.\n", location);
    }
    if (!m_bindless) {
        location = glslcHelper.ProgramGetResourceLocation(reinterpret_cast<LWNprogram *>(pgm), LWN_SHADER_STAGE_FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_SAMPLER, "boundTex");
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

    blend.SetDefaults();
    cmask.SetDefaults();
    color.SetDefaults();
    depth.SetDefaults();
    multisample.SetDefaults();
    polygon.SetDefaults();

    depth.SetDepthTestEnable(LWN_TRUE);
    depth.SetDepthWriteEnable(LWN_TRUE);
    depth.SetDepthFunc(DepthFunc::LESS);

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
    lwn::objects::VertexAttribState vertexAttribs[2];
    lwn::objects::VertexStreamState vertexStreams[2];
    vertexAttribs[0].SetDefaults();
    vertexAttribs[1].SetDefaults();
    vertexStreams[0].SetDefaults();
    vertexStreams[1].SetDefaults();
    vertexAttribs[0].SetFormat(Format::RGB32F, 0);
    vertexAttribs[1].SetFormat(Format::RGBA8, 0);
    vertexAttribs[0].SetStreamIndex(0);
    vertexAttribs[1].SetStreamIndex(1);
    vertexStreams[0].SetStride(12);
    vertexStreams[1].SetStride(4);

    // Create a vertex buffer and fill it with data
    bufferBuilder.SetDefaults();
    Buffer *vbo = AllocBuf(&bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData) + sizeof(texcoordData));

    // create persistent mapping
    void *ptr = vbo->Map();
    // fill ptr with vertex data followed by color data
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));

    // Get a handle to be used for setting the buffer as a vertex buffer
    BufferAddress vboAddr = vbo->GetAddress();

    // Create an index buffer and fill it with data
    unsigned short indexData[6] = {0, 1, 2, 3, 4, 5};
    Buffer *ibo = AllocAndFillBufferCPP(device, queue, queueCB, indexData, sizeof(indexData),
                                        BUFFER_ALIGN_INDEX_BIT, true);

    // Get a handle to be used for setting the buffer as an index buffer
    BufferAddress iboAddr = ibo->GetAddress();

    int m_rtBufferIdx;

    Sync displayReleaseSync;
    displayReleaseSync.Initialize(device);
    appWin->win.AcquireTexture(&displayReleaseSync, &m_rtBufferIdx);
    queue->WaitSync(&displayReleaseSync);
    displayReleaseSync.Finalize();

    Texture *depthTex;
    Texture *tex4x;
    Texture *rtTex;

    textureBuilder.SetDefaults();
    textureBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    textureBuilder.SetSize2D(offscreenWidth, offscreenHeight);
    textureBuilder.SetTarget(TextureTarget::TARGET_2D);
    textureBuilder.SetFormat(Format::RGBA8);

    rtTex = AllocTex(&textureBuilder);

    lwn::objects::Texture *colorTargets[] = { rtTex };

    if (m_multisample) {

        textureBuilder.SetTarget(TextureTarget::TARGET_2D_MULTISAMPLE);
        textureBuilder.SetSamples(4);

        multisample.SetSamples(4);

        tex4x = AllocTex(&textureBuilder);

        colorTargets[0] = tex4x;
    }


    textureBuilder.SetFormat(Format::DEPTH24_STENCIL8);
    depthTex = AllocTex(&textureBuilder);

    queueCB->SetRenderTargets(1, colorTargets, NULL, depthTex, NULL);

    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDefaults().SetDevice(device);

    // Commented out experiments to test different state settings.    
    // samplerBuilder.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *sampler = samplerBuilder.CreateSampler();
    int samplerID = lwnGetRegisteredSamplerID(sampler);

    const int texWidth = 4, texHeight = 4;
    textureBuilder.SetDefaults().SetTarget(TextureTarget::TARGET_2D).
        SetFormat(Format::RGBA8).SetSize2D(texWidth, texHeight);
    Texture *texture = AllocTex(&textureBuilder);
    int textureID = lwnGetRegisteredTextureID(texture);

    // Bindless requires a combined texture/sampler handle.  Bound will
    // also need to bind a combined handle.
    TextureHandle texHandle = device->GetTextureHandle(textureID, samplerID);

    bufferBuilder.SetDefaults();
    Buffer *pbo = AllocBuf(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texWidth*texHeight*4);
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
    Buffer *ubo = AllocAndFillBufferCPP(device, queue, queueCB, &uboData, sizeof(uboData),
                                        BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    BufferAddress uboAddr = ubo->GetAddress();

    // Some scissored clears
    {
        queueCB->SetScissor(0, 0, offscreenWidth, offscreenHeight);
        float clearColor[] = {0, 0, 0, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
        queueCB->ClearDepthStencil(1.0, LWN_TRUE, 0, 0);
    }
    {
        queueCB->SetScissor(offscreenWidth/2, 0, offscreenWidth/2, offscreenHeight/2);
        float clearColor[] = {0, 0.5, 0, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
    }
    {
        queueCB->SetScissor(0, offscreenHeight/2, offscreenWidth/2, offscreenHeight/2);
        float clearColor[] = {0, 0, 0.5, 1};
        queueCB->ClearColor(0, clearColor, ClearColorMask::RGBA);
    }
    queueCB->SetScissor(0, 0, offscreenWidth, offscreenHeight);
    queueCB->SetViewport(0, 0, offscreenWidth, offscreenHeight);

    if (m_benchmark) {
        queue->Finish();
    }
    clock_t startTime = clock();
    unsigned int numIterations = m_benchmark ? 10000000 : 1;

    // In benchmark mode, we'll be drawing a lot of primitives in a single
    // frame.  In QUEUE and COMMAND_TRANSIENT modes, we'll need to flush every
    // so often so we don't run out of command buffer memory.  10M draw calls
    // needs quite a bit.
    unsigned int numIterationsPerFence = 10000;

    DrawPrimitive drawPrimitive = DrawPrimitive::TRIANGLES;
    if (m_tessEvalShader) {
        float levels[4] = { 2, 2, 2, 2 };
        drawPrimitive = DrawPrimitive::PATCHES;
        queueCB->SetPatchSize(3);
        queueCB->SetInnerTessellationLevels(levels);
        queueCB->SetOuterTessellationLevels(levels);
    }

    if (m_submitMode == QUEUE) {
        for (unsigned int i = 0; i < numIterations; ++i) {
            // Bind the program, vertex state, and any required control structures.
            queueCB->BindProgram(pgm, allShaderStages);
            queueCB->BindVertexAttribState(2, vertexAttribs);
            queueCB->BindVertexStreamState(2, vertexStreams);
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

            // Flush and insert fences periodically so we don't run out of
            // command memory.
            if (i != 0 && (i % numIterationsPerFence) == 0) {
                queueCBHandle = queueCB->EndRecording();
                queue->SubmitCommands(1, &queueCBHandle);
                insertCompletionTrackerFence(m_cpp_interface->completionTracker, reinterpret_cast<LWNqueue *>(queue));
                queueCB->BeginRecording();
            }
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
            cmdMem->populateCommandBuffer(reinterpret_cast<LWNcommandBuffer *>(cmd[i]), LWNcommandBufferMemoryManager::Coherent);
            cmd[i]->BeginRecording();
            // Bind the program, vertex state, and any required control structures.
            cmd[i]->BindProgram(pgm, allShaderStages);
            cmd[i]->BindVertexAttribState(2, vertexAttribs);
            cmd[i]->BindVertexStreamState(2, vertexStreams);
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
                cmdMem->populateCommandBuffer(reinterpret_cast<LWNcommandBuffer *>(cmd[i]), LWNcommandBufferMemoryManager::Coherent);
                cmd[i]->BeginRecording();
                // Bind the program, vertex state, and any required control structures.
                cmd[i]->BindProgram(pgm, allShaderStages);
                cmd[i]->BindVertexAttribState(2, vertexAttribs);
                cmd[i]->BindVertexStreamState(2, vertexStreams);
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

            // Insert fences periodically so we don't run out of command
            // memory.
            if (j != 0 && (j % numIterationsPerFence) < N) {
                insertCompletionTrackerFence(m_cpp_interface->completionTracker, reinterpret_cast<LWNqueue *>(queue));
            }
        }
    }

    if (m_multisample) {
        queueCB->Downsample(tex4x, rtTex);

        // The contents of the lwrrently bound depth and colors are
        // not going to be needed after the Downsample operation.  The
        // below discard operations allow the GPU to not write the
        // corresponding GPU L2 cachelines into main memory.  This is
        // a potential memory bandwidth optimization.
        queueCB->DiscardColor(0);
        queueCB->DiscardDepthStencil();
    }

    // Since the windows size might be different to the offscreen texture size, the
    // offsceen texture is copied and scaled to the present texture.
    CopyRegion srcRegion = { 0, 0, 0, offscreenWidth, offscreenHeight, 1 };
    CopyRegion dstRegion = { 0, 0, 0, m_windowWidth, m_windowHeight, 1 };

    queueCB->CopyTextureToTexture(rtTex, 0, &srcRegion, appWin->presentTexture[m_rtBufferIdx], 0, &dstRegion,
                                  CopyFlags::LINEAR_FILTER);

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

    insertCompletionTrackerFence(m_cpp_interface->completionTracker, reinterpret_cast<LWNqueue *>(queue));

    queue->PresentTexture(&appWin->win, m_rtBufferIdx);

    pgm->Free();
    FreeBuf(vbo);
    FreeBuf(ibo);
    FreeBuf(pbo);
    FreeBuf(ubo);
    FreeTex(texture);
    FreeTex(depthTex);
    FreeTex(rtTex);
    sampler->Free();
    if (m_multisample) {
        FreeTex(tex4x);
    }
}
