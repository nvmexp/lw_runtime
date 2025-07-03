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

#define LWN_USE_C_INTERFACE         1
#include "lwnexample.h"

#if defined __ANDROID__
#include <android/log.h>
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "LWNTest", __VA_ARGS__))
#define log_output  LOGI
#else
#define log_output  printf
#endif

static size_t programPoolSize = 0x100000UL; // 1MB pool size

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

static LWNshaderStageBits allShaderStages = LWNshaderStageBits(LWN_SHADER_STAGE_VERTEX_BIT |
                                                               LWN_SHADER_STAGE_TESS_CONTROL_BIT |
                                                               LWN_SHADER_STAGE_TESS_EVALUATION_BIT |
                                                               LWN_SHADER_STAGE_GEOMETRY_BIT | 
                                                               LWN_SHADER_STAGE_FRAGMENT_BIT);

typedef struct {
    float scale[4];
    LWNtextureHandle bindlessTex;
} UniformBlock;

static void LWNAPIENTRY debugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                                      LWNdebugCallbackSeverity severity, const char *message, void *userParam)
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


void LWNSampleTestConfig::generateDebugMessages()
{
    LWNdevice *device = m_c_interface->device;

    const char *brokenShader = "#version 400\nI am a string, but not a shader.\n";
    LWNprogram *dummyPgm = lwnDeviceCreateProgram(device);
    LWNshaderStage stages[2];
    const char *sources[2];
    sources[0] = brokenShader;
    sources[1] = brokenShader;
    stages[0] = LWN_SHADER_STAGE_VERTEX;
    stages[1] = LWN_SHADER_STAGE_FRAGMENT;

    lwnUtil::GLSLCHelper glslcHelper(device, programPoolSize, m_c_interface->glslcLibraryHelper);

    if (!glslcHelper.CompileAndSetShaders(dummyPgm, stages, 2, sources)) {
        log_output("Shader compile error. infoLog=\n%s\n", glslcHelper.GetInfoLog());
    }

    lwnProgramFree(dummyPgm);

    LWNtextureBuilder *texBuilder = lwnDeviceCreateTextureBuilder(device);
    LWNsamplerBuilder *smpBuilder = lwnDeviceCreateSamplerBuilder(device);
    lwnTextureBuilderSetFlags(texBuilder, LWNtextureFlags(0xAAAAAAAA));   // illegal - bad bitfield
    lwnSamplerBuilderSetMinMagFilter(smpBuilder, LWN_MIN_FILTER_NEAREST, (LWNmagFilter)0x4412);  // illegal - bad enum
    lwnTextureBuilderFree(texBuilder);
    lwnSamplerBuilderFree(smpBuilder);
}

static LWNbuffer *AllocAndFillBuffer(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmdBuf,
                                     void *data, int sizeofdata, BufferAlignBits alignBits, bool useCopy)
{
    LWNbuffer *buffer = NULL;
    LWNbufferBuilder *bb = lwnDeviceCreateBufferBuilder(device);

    if (useCopy) {
        LWNbuffer *tempbo = g_bufferAllocator->allocBuffer(bb, BUFFER_ALIGN_COPY_READ_BIT, sizeofdata);
        LWNbufferAddress tempbo_addr = lwnBufferGetAddress(tempbo);
        void *ptr = lwnBufferMap(tempbo);
        memcpy(ptr, data, sizeofdata);

        buffer = g_bufferAllocator->allocBuffer(bb, BufferAlignBits(alignBits | BUFFER_ALIGN_COPY_WRITE_BIT), sizeofdata);
        LWNbufferAddress buffer_addr = lwnBufferGetAddress(buffer);
        lwnCommandBufferCopyBufferToBuffer(cmdBuf, tempbo_addr, buffer_addr,sizeofdata, LWN_COPY_FLAGS_NONE);

        // Flush command buffer contents before calling Finish; reopen the
        // command buffer for further recording.
        LWNcommandHandle handle = lwnCommandBufferEndRecording(cmdBuf);
        lwnQueueSubmitCommands(queue, 1, &handle);
        lwnCommandBufferBeginRecording(cmdBuf);
        lwnQueueFinish(queue);

        g_bufferAllocator->freeBuffer(tempbo);

    } else {
        buffer = g_bufferAllocator->allocBuffer(bb, alignBits, sizeofdata);
        void *ptr = lwnBufferMap(buffer);
        memcpy(ptr, data, sizeofdata);
    }

    lwnBufferBuilderFree(bb);

    return buffer;
}



LWNBasicWindowC* LWNSampleTestConfig::cCreateWindow(LWNnativeWindow nativeWindow, int w, int h)
{
    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(m_c_interface->device);

    LWNBasicWindowC *appWindow = new LWNBasicWindowC;

    m_windowWidth  = w;
    m_windowHeight = h;

    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetFlags(textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT | LWN_TEXTURE_FLAGS_DISPLAY_BIT);
    lwnTextureBuilderSetSize2D(textureBuilder, m_windowWidth, m_windowHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
        appWindow->presentTexture[i] = g_texAllocator->allocTexture(textureBuilder);
    }

    lwnTextureBuilderFree(textureBuilder);

    LWNwindowBuilder windowBuilder;
    lwnWindowBuilderSetDevice(&windowBuilder, m_c_interface->device);
    lwnWindowBuilderSetDefaults(&windowBuilder);
    lwnWindowBuilderSetTextures(&windowBuilder, NUM_PRESENT_TEXTURES, appWindow->presentTexture);
    lwnWindowBuilderSetNativeWindow(&windowBuilder, nativeWindow);

    if (!lwnWindowInitialize(&appWindow->win, &windowBuilder)) {
        for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
            g_texAllocator->freeTexture(appWindow->presentTexture[i]);
        }

        delete appWindow;
        return NULL;
    }

    return appWindow;
}

void LWNSampleTestConfig::cDeleteWindow()
{
    if (m_c_interface->window) {
        lwnWindowFinalize(&m_c_interface->window->win);

        for (unsigned int i = 0; i < NUM_PRESENT_TEXTURES; ++i) {
            g_texAllocator->freeTexture(m_c_interface->window->presentTexture[i]);
        }

        delete m_c_interface->window;
        m_c_interface->window = NULL;
    }
}

void LWNSampleTestConfig::cDisplay()
{
    LWNdevice *device = m_c_interface->device;
    LWNqueue *queue = m_c_interface->queue;
    LWNcommandBuffer *queueCB = m_c_interface->queueCB;
    LWNcommandBufferMemoryManager *cmdMem = m_c_interface->cmdMemMgr;
    LWNcommandHandle queueCBHandle;
    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);
    LWNBasicWindowC *appWin = m_c_interface->window;

    // For "-debug", call generateDebugMessages() to generate some stupid
    // errors to exercise the callbacks.
    if (m_debug) {
        lwnDeviceInstallDebugCallback(device, debugCallback, (void *) 0x8675309, LWN_TRUE);
        generateDebugMessages();
    }

    // Create programs from the device, provide them shader code and compile/link them
    LWNprogram *pgm = lwnDeviceCreateProgram(device);

    // XXX This is a hack because we don't have an IL. I'm just jamming through the strings 
    // as if they were an IL blob
    LWNshaderStage stages[5];
    const char *sources[5];
    int nSources = 2;
    sources[0] = vsstring;
    stages[0] = LWN_SHADER_STAGE_VERTEX;
    sources[1] = m_bindless ? fsstring_bindless : fsstring;
    stages[1] = LWN_SHADER_STAGE_FRAGMENT;
    if (m_geometryShader) {
        sources[nSources] = gsstring;
        stages[nSources] = LWN_SHADER_STAGE_GEOMETRY;
        nSources++;
    }
    if (m_tessControlShader) {
        sources[nSources] = tcsstring;
        stages[nSources] = LWN_SHADER_STAGE_TESS_CONTROL;
        nSources++;
    }
    if (m_tessEvalShader) {
        sources[nSources] = tesstring;
        stages[nSources] = LWN_SHADER_STAGE_TESS_EVALUATION;
        nSources++;
    }

    lwnUtil::GLSLCHelper glslcHelper(device, programPoolSize, m_c_interface->glslcLibraryHelper);

    if (!glslcHelper.CompileAndSetShaders(pgm, stages, nSources, sources)) {
        log_output("Shader compile error. infoLog=\n%s\n", glslcHelper.GetInfoLog());
    }

    // Check to make sure program interfaces work as expected.
    int location;
    location = glslcHelper.ProgramGetResourceLocation(pgm, LWN_SHADER_STAGE_VERTEX, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in vertex.\n", location);
    }
    location = glslcHelper.ProgramGetResourceLocation(pgm, LWN_SHADER_STAGE_FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK, "Block");
    if (location != 0) {
        log_output("Block has location %d in fragment.\n", location);
    }
    if (!m_bindless) {
        location = glslcHelper.ProgramGetResourceLocation(pgm, LWN_SHADER_STAGE_FRAGMENT, LWN_PROGRAM_RESOURCE_TYPE_SAMPLER, "boundTex");
        if (location != 0) {
            log_output("tex has location %d in fragment.\n", location);
        }
    }

    // Create new state vectors
    LWNblendState *blend = lwnDeviceCreateBlendState(device);
    LWNchannelMaskState *cmask = lwnDeviceCreateChannelMaskState(device);
    LWNcolorState *color = lwnDeviceCreateColorState(device);
    LWNdepthStencilState *depth = lwnDeviceCreateDepthStencilState(device);
    LWNmultisampleState *multisample = lwnDeviceCreateMultisampleState(device);
    LWNpolygonState *polygon = lwnDeviceCreatePolygonState(device);

    lwnDepthStencilStateSetDepthTestEnable(depth, LWN_TRUE);
    lwnDepthStencilStateSetDepthWriteEnable(depth, LWN_TRUE);
    lwnDepthStencilStateSetDepthFunc(depth, LWN_DEPTH_FUNC_LESS);

    if (m_wireframe) {
        lwnPolygonStateSetPolygonMode(polygon, LWN_POLYGON_MODE_LINE);
    } else {
        lwnPolygonStateSetPolygonMode(polygon, LWN_POLYGON_MODE_FILL);
    }
    //lwnPolygonStateDiscardEnable(raster, LWN_TRUE);

    //lwnColorStateBlendEnable(color, /*MRT index*/0, LWN_TRUE);
    //lwnBlendStateBlendFunc(color, /*MRT index*/0, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE, LWN_BLEND_FUNC_ONE);
    //lwnChannelMaskStateMask(color, /*MRT index*/0, LWN_TRUE, LWN_TRUE, LWN_TRUE, LWN_TRUE);
    //lwnColorStateLogicOp(color, LWN_LOGIC_OP_XOR);

    // Set the state vector to use two vertex attributes.
    //
    // Interleaved pos+color
    // position = attrib 0 = 3*float at relativeoffset 0
    // texcoord = attrib 1 = rgba8 at relativeoffset 0
    LWLwertexAttribState attribState[2];
    LWLwertexStreamState streamState[2];

    lwlwertexAttribStateSetDefaults(attribState + 0);
    lwlwertexAttribStateSetDefaults(attribState + 1);
    lwlwertexAttribStateSetFormat(attribState + 0, LWN_FORMAT_RGB32F, 0);
    lwlwertexAttribStateSetFormat(attribState + 1, LWN_FORMAT_RGBA8, 0);
    lwlwertexAttribStateSetStreamIndex(attribState + 0, 0);
    lwlwertexAttribStateSetStreamIndex(attribState + 1, 1);

    lwlwertexStreamStateSetDefaults(streamState + 0);
    lwlwertexStreamStateSetDefaults(streamState + 1);
    lwlwertexStreamStateSetStride(streamState + 0, 12);
    lwlwertexStreamStateSetStride(streamState + 1, 4);

    // Create a vertex buffer and fill it with data
    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *vbo = g_bufferAllocator->allocBuffer(bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData)+sizeof(texcoordData));
    // create persistent mapping
    void *ptr = lwnBufferMap(vbo);
    // fill ptr with vertex data followed by color data
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));

    // Get a handle to be used for setting the buffer as a vertex buffer
    LWNbufferAddress vboAddr = lwnBufferGetAddress(vbo);

    // Create an index buffer and fill it with data
    unsigned short indexData[6] = {0, 1, 2, 3, 4, 5};
    LWNbuffer *ibo = AllocAndFillBuffer(device, queue, queueCB, indexData, sizeof(indexData), BUFFER_ALIGN_INDEX_BIT, true);

    // Get a handle to be used for setting the buffer as an index buffer
    LWNbufferAddress iboAddr = lwnBufferGetAddress(ibo);

    int m_rtBufferIdx;

    LWNsync textureAvailableSync;
    lwnSyncInitialize(&textureAvailableSync, device);
    lwnWindowAcquireTexture(&appWin->win, &textureAvailableSync, &m_rtBufferIdx);
    lwnQueueWaitSync(queue, &textureAvailableSync);
    lwnSyncFinalize(&textureAvailableSync);

    LWNtexture *depthTex = NULL;
    LWNtexture *tex4x = NULL;
    LWNtexture *rtTex = NULL;

    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetFlags(textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    rtTex = g_texAllocator->allocTexture(textureBuilder);

    LWNtexture *rt = rtTex;

    if (m_multisample) {
        // Set up the builder to create multisample textures.
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D_MULTISAMPLE);
        lwnTextureBuilderSetSamples(textureBuilder, 4);

        lwnMultisampleStateSetSamples(multisample, 4);

        // Allocate a single-sample texture for the multi-texture configuration.
        tex4x = g_texAllocator->allocTexture(textureBuilder);

        // Choose msx4 texture as rendertarget
        rt = tex4x;
    }

    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_DEPTH24_STENCIL8);
    depthTex = g_texAllocator->allocTexture(textureBuilder);

    lwnCommandBufferSetRenderTargets(queueCB, 1, &rt, NULL, depthTex, NULL);

    LWNsamplerBuilder *samplerBuilder = lwnDeviceCreateSamplerBuilder(device);
    
    // Commented out experiments to test different state settings.    
    // lwnSamplerBuilderSetMinMagFilter(samplerBuilder, LWN_MIN_FILTER_NEAREST, LWN_MAG_FILTER_NEAREST);

    LWNsampler *sampler = lwnSamplerBuilderCreateSampler(samplerBuilder);
    int samplerID = lwnSamplerGetRegisteredID(sampler);

    const int texWidth = 4, texHeight = 4;
    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(textureBuilder, texWidth, texHeight);
    LWNtexture *texture = g_texAllocator->allocTexture(textureBuilder);
    int textureID = lwnTextureGetRegisteredTextureID(texture);

    // Build a combined texture/sampler handle.
    LWNtextureHandle texHandle = lwnDeviceGetTextureHandle(device, textureID ,samplerID);

    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *pbo = g_bufferAllocator->allocBuffer(bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texWidth*texHeight*4);
    LWNbufferAddress pboAddr = lwnBufferGetAddress(pbo);

    unsigned char *texdata = (unsigned char *)lwnBufferMap(pbo);
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
    LWNcopyRegion copyRegion = { 0, 0, 0, texWidth, texHeight, 1 };
    lwnCommandBufferCopyBufferToTexture(queueCB, pboAddr, texture, NULL, &copyRegion, LWN_COPY_FLAGS_NONE);

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
    LWNbuffer *ubo = AllocAndFillBuffer(device, queue, queueCB, &uboData, sizeof(uboData), BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = lwnBufferGetAddress(ubo);

    // Some scissored clears
    {
        lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);
        float clearColor[] = {0,0,0,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
        lwnCommandBufferClearDepthStencil(queueCB, 1.0, LWN_TRUE, 0, 0);
    }
    {
        lwnCommandBufferSetScissor(queueCB, offscreenWidth/2, 0, offscreenWidth/2, offscreenHeight/2);
        float clearColor[] = {0,0.5,0,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    }
    {
        lwnCommandBufferSetScissor(queueCB, 0, offscreenHeight/2, offscreenWidth/2, offscreenHeight/2);
        float clearColor[] = {0,0,0.5,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    }
    lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);
    lwnCommandBufferSetViewport(queueCB, 0, 0, offscreenWidth, offscreenHeight);

    if (m_benchmark) {
        lwnQueueFinish(queue);
    }
    clock_t startTime = clock();
    unsigned int numIterations = m_benchmark ? 10000000 : 1;

    // In benchmark mode, we'll be drawing a lot of primitives in a single
    // frame.  In QUEUE and COMMAND_TRANSIENT modes, we'll need to flush every
    // so often so we don't run out of command buffer memory.  10M draw calls
    // needs quite a bit.
    unsigned int numIterationsPerFence = 10000;

    LWNdrawPrimitive drawPrimitive = LWN_DRAW_PRIMITIVE_TRIANGLES;
    if (m_tessEvalShader) {
        float levels[4] = { 2, 2, 2, 2 };
        drawPrimitive = LWN_DRAW_PRIMITIVE_PATCHES;
        lwnCommandBufferSetPatchSize(queueCB, 3);
        lwnCommandBufferSetInnerTessellationLevels(queueCB, levels);
        lwnCommandBufferSetOuterTessellationLevels(queueCB, levels);
    }

    if (m_submitMode == QUEUE) {
        for (unsigned int i = 0; i < numIterations; ++i) {
            // Bind the program, vertex state, and any required control structures.
            lwnCommandBufferBindProgram(queueCB, pgm, allShaderStages);
            lwnCommandBufferBindVertexAttribState(queueCB, 2, attribState);
            lwnCommandBufferBindVertexStreamState(queueCB, 2, streamState);
            lwnCommandBufferBindBlendState(queueCB, blend);
            lwnCommandBufferBindChannelMaskState(queueCB, cmask);
            lwnCommandBufferBindColorState(queueCB, color);
            lwnCommandBufferBindDepthStencilState(queueCB, depth);
            lwnCommandBufferBindMultisampleState(queueCB, multisample);
            lwnCommandBufferBindPolygonState(queueCB, polygon);
            lwnCommandBufferSetSampleMask(queueCB, ~0);
            lwnCommandBufferBindVertexBuffer(queueCB, 0, vboAddr, sizeof(vertexData));
            lwnCommandBufferBindVertexBuffer(queueCB, 1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
            lwnCommandBufferBindUniformBuffer(queueCB, LWN_SHADER_STAGE_VERTEX, 0, uboAddr, sizeof(uboData));
            lwnCommandBufferBindUniformBuffer(queueCB, LWN_SHADER_STAGE_FRAGMENT, 0, uboAddr, sizeof(uboData));
            if (!m_bindless) {
                lwnCommandBufferBindTexture(queueCB, LWN_SHADER_STAGE_FRAGMENT, 0, texHandle);
            }
            lwnCommandBufferDrawElements(queueCB, drawPrimitive, LWN_INDEX_TYPE_UNSIGNED_SHORT, 6, iboAddr);

            // Flush and insert fences periodically so we don't run out of
            // command memory.
            if (i != 0 && (i % numIterationsPerFence) == 0) {
                queueCBHandle = lwnCommandBufferEndRecording(queueCB);
                lwnQueueSubmitCommands(queue, 1, &queueCBHandle);
                insertCompletionTrackerFence(m_c_interface->completionTracker, queue);
                lwnCommandBufferBeginRecording(queueCB);
            }
        }
    }
    
    // Flush the queue command buffer open for future reuse (if needed).
    queueCBHandle = lwnCommandBufferEndRecording(queueCB);
    lwnQueueSubmitCommands(queue, 1, &queueCBHandle);
    lwnCommandBufferBeginRecording(queueCB);

    if (m_submitMode == COMMAND) {
        const int N = 4;
        LWNcommandBuffer *cmd[N];
        LWNcommandHandle cmdHandle[N];
        for (int i = 0; i < N; ++i) {
            cmd[i] = lwnDeviceCreateCommandBuffer(device);
            cmdMem->populateCommandBuffer(cmd[i], LWNcommandBufferMemoryManager::Coherent);
            lwnCommandBufferBeginRecording(cmd[i]);
            // Bind the program, vertex state, and any required control structures.
            lwnCommandBufferBindProgram(cmd[i], pgm, allShaderStages);
            lwnCommandBufferBindVertexAttribState(cmd[i], 2, attribState);
            lwnCommandBufferBindVertexStreamState(cmd[i], 2, streamState);
            lwnCommandBufferBindBlendState(cmd[i], blend);
            lwnCommandBufferBindChannelMaskState(cmd[i], cmask);
            lwnCommandBufferBindColorState(cmd[i], color);
            lwnCommandBufferBindDepthStencilState(cmd[i], depth);
            lwnCommandBufferBindMultisampleState(cmd[i], multisample);
            lwnCommandBufferBindPolygonState(cmd[i], polygon);
            lwnCommandBufferSetSampleMask(cmd[i], ~0);
            lwnCommandBufferBindVertexBuffer(cmd[i], 0, vboAddr, sizeof(vertexData));
            lwnCommandBufferBindVertexBuffer(cmd[i], 1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
            lwnCommandBufferBindUniformBuffer(cmd[i], LWN_SHADER_STAGE_VERTEX, 0, uboAddr, sizeof(uboData));
            lwnCommandBufferBindUniformBuffer(cmd[i], LWN_SHADER_STAGE_FRAGMENT, 0, uboAddr, sizeof(uboData));
            if (!m_bindless) {
                lwnCommandBufferBindTexture(cmd[i], LWN_SHADER_STAGE_FRAGMENT, 0, texHandle);
            }
            lwnCommandBufferDrawElements(cmd[i], drawPrimitive, LWN_INDEX_TYPE_UNSIGNED_SHORT, 6, iboAddr);
            cmdHandle[i] = lwnCommandBufferEndRecording(cmd[i]);
        }
        for (unsigned int j = 0; j < numIterations; j += N) {
            lwnQueueSubmitCommands(queue, N, &cmdHandle[0]);
        }
        for (int i = 0; i < N; ++i) {
            lwnCommandBufferFree(cmd[i]);
        }
    } else if (m_submitMode == COMMAND_TRANSIENT) {
        const int N = 4;
        LWNcommandBuffer *cmd[N];
        LWNcommandHandle cmdHandle[N];
        for (unsigned int j = 0; j < numIterations; j += N) {
            for (int i = 0; i < N; ++i) {
                cmd[i] = lwnDeviceCreateCommandBuffer(device);
                cmdMem->populateCommandBuffer(cmd[i], LWNcommandBufferMemoryManager::Coherent);
                lwnCommandBufferBeginRecording(cmd[i]);
                // Bind the program, vertex state, and any required control structures.
                lwnCommandBufferBindProgram(cmd[i], pgm, allShaderStages);
                lwnCommandBufferBindVertexAttribState(cmd[i], 2, attribState);
                lwnCommandBufferBindVertexStreamState(cmd[i], 2, streamState);
                lwnCommandBufferBindBlendState(cmd[i], blend);
                lwnCommandBufferBindChannelMaskState(cmd[i], cmask);
                lwnCommandBufferBindColorState(cmd[i], color);
                lwnCommandBufferBindDepthStencilState(cmd[i], depth);
                lwnCommandBufferBindMultisampleState(cmd[i], multisample);
                lwnCommandBufferBindPolygonState(cmd[i], polygon);
                lwnCommandBufferSetSampleMask(cmd[i], ~0);
                lwnCommandBufferBindVertexBuffer(cmd[i], 0, vboAddr, sizeof(vertexData));
                lwnCommandBufferBindVertexBuffer(cmd[i], 1, vboAddr + sizeof(vertexData), sizeof(texcoordData));
                lwnCommandBufferBindUniformBuffer(cmd[i], LWN_SHADER_STAGE_VERTEX, 0, uboAddr, sizeof(uboData));
                lwnCommandBufferBindUniformBuffer(cmd[i], LWN_SHADER_STAGE_FRAGMENT, 0, uboAddr, sizeof(uboData));
                if (!m_bindless) {
                    lwnCommandBufferBindTexture(cmd[i], LWN_SHADER_STAGE_FRAGMENT, 0, texHandle);
                }
                lwnCommandBufferDrawElements(cmd[i], drawPrimitive, LWN_INDEX_TYPE_UNSIGNED_SHORT, 6, iboAddr);
                cmdHandle[i] = lwnCommandBufferEndRecording(cmd[i]);
            }
            lwnQueueSubmitCommands(queue, N, &cmdHandle[0]);
            for (int i = 0; i < N; ++i) {
                lwnCommandBufferFree(cmd[i]);
            }

            // Insert fences periodically so we don't run out of command
            // memory.
            if (j != 0 && (j % numIterationsPerFence) < N) {
                insertCompletionTrackerFence(m_c_interface->completionTracker, queue);
            }
        }
    }

    if (m_multisample) {
        lwnCommandBufferDownsample(queueCB, tex4x, rtTex);

        // The contents of the lwrrently bound depth and colors are
        // not going to be needed after the Downsample operation.  The
        // below discard operations allow the GPU to not write the
        // corresponding GPU L2 cachelines into main memory.  This is
        // a potential memory bandwidth optimization.
        lwnCommandBufferDiscardColor(queueCB, 0);
        lwnCommandBufferDiscardDepthStencil(queueCB);
    }

    // Since the windows size might be different to the offscreen texture size, the
    // offsceen texture is copied and scaled to the present texture.
    LWNcopyRegion srcRegion = { 0, 0, 0, offscreenWidth, offscreenWidth, 1 };
    LWNcopyRegion dstRegion = { 0, 0, 0, m_windowWidth, m_windowHeight, 1 };
    lwnCommandBufferCopyTextureToTexture(queueCB, rtTex, NULL, &srcRegion,
                                         appWin->presentTexture[m_rtBufferIdx], NULL, &dstRegion,
                                         LWN_COPY_FLAGS_LINEAR_FILTER_BIT);

    queueCBHandle = lwnCommandBufferEndRecording(queueCB);
    lwnQueueSubmitCommands(queue, 1, &queueCBHandle);
    lwnCommandBufferBeginRecording(queueCB);

    if (m_benchmark) {
        lwnQueueFinish(queue);
        clock_t lwrrentTime = clock();
        log_output("%f\n", 1.0f*numIterations*CLOCKS_PER_SEC/(lwrrentTime - startTime));

        // Inhibit unused variable warnings if output logging is disabled.
        (void)startTime;
        (void)lwrrentTime;
    }

    insertCompletionTrackerFence(m_c_interface->completionTracker, queue);

    lwnQueueFinish(queue);

    lwnQueuePresentTexture(queue, &appWin->win, m_rtBufferIdx);

    lwnProgramFree(pgm);
    lwnBlendStateFree(blend);
    lwnChannelMaskStateFree(cmask);
    lwnColorStateFree(color);
    lwnDepthStencilStateFree(depth);
    lwnPolygonStateFree(polygon);
    lwnMultisampleStateFree(multisample);
    lwnBufferBuilderFree(bufferBuilder);
    g_bufferAllocator->freeBuffer(vbo);
    g_bufferAllocator->freeBuffer(ibo);
    g_bufferAllocator->freeBuffer(pbo);
    g_bufferAllocator->freeBuffer(ubo);
    lwnTextureBuilderFree(textureBuilder);
    g_texAllocator->freeTexture(texture);
    g_texAllocator->freeTexture(depthTex);
    g_texAllocator->freeTexture(rtTex);
    lwnSamplerBuilderFree(samplerBuilder);
    lwnSamplerFree(sampler);
    if (m_multisample) {
        g_texAllocator->freeTexture(tex4x);
    }
}

