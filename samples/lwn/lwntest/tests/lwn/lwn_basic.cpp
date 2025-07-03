/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_c.h"
#include "lwn_utils.h"
#include "lwn_basic.h"
#include <time.h>

using namespace lwn;

static struct LWNSampleTestCInterface g_lwnSampleTestCInterface;
LWNSampleTestCInterface *LWNSampleTestConfig::m_c_interface = &g_lwnSampleTestCInterface;

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

static LWNshaderStageBits allShaderStages = LWNshaderStageBits(LWN_SHADER_STAGE_VERTEX_BIT |
                                                               LWN_SHADER_STAGE_TESS_CONTROL_BIT |
                                                               LWN_SHADER_STAGE_TESS_EVALUATION_BIT |
                                                               LWN_SHADER_STAGE_GEOMETRY_BIT | 
                                                               LWN_SHADER_STAGE_FRAGMENT_BIT);
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


void LWNSampleTestConfig::generateDebugMessages()
{
    LWNdevice *device = m_c_interface->device;

    const char *brokenShader = "#version 400\nI am a string, but not a shader.\n";
    LWNprogram *dummyPgm = lwnDeviceCreateProgram(device);
    const char *sources[2];
    sources[0] = brokenShader;
    sources[1] = brokenShader;

    // Create a GLSLC helper instead of using the global version
    // since the device may be a debug device depending on the
    // test variant.
    lwnTest::GLSLCHelper glslcHelper(device, programPoolSize, g_glslcLibraryHelper, g_glslcHelperCache);
    LWNmemoryPool * scratchMemoryPool = lwnDeviceCreateMemoryPool(device, NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemoryPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, NULL);

    // Temporarily disable logging in the GLSLCHelper since these shaders are expected to fail
    // compilation and we don't want to print the info log as a debug message.
    GLSLCLogger * logger = glslcHelper.GetLogger();

    bool wasLoggerEnabled = logger->IsEnabled();
    logger->SetEnable(false);
    VertexShader vs(440);
    vs << sources[0];
    FragmentShader fs(440);
    fs << sources[1];
    
    glslcHelper.CompileAndSetShaders(dummyPgm, vs, fs);
    logger->SetEnable(wasLoggerEnabled);

    lwnProgramFree(dummyPgm);

    LWNtextureBuilder *texBuilder = lwnDeviceCreateTextureBuilder(device);
    LWNsamplerBuilder *smpBuilder = lwnDeviceCreateSamplerBuilder(device);
    lwnTextureBuilderSetFlags(texBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);      // legal
    lwnTextureBuilderSetFlags(texBuilder, LWNtextureFlags(0xAAAAAAAA));   // illegal - bad bitfield
    lwnSamplerBuilderSetMinMagFilter(smpBuilder, LWN_MIN_FILTER_NEAREST, (LWNmagFilter)0x4412);  // illegal - bad enum
    lwnTextureBuilderFree(texBuilder);
    lwnSamplerBuilderFree(smpBuilder);
    lwnMemoryPoolFree(scratchMemoryPool);
}

void LWNSampleTestConfig::cDisplay()
{
    LWNdevice *device = m_c_interface->device;
    LWNqueue *queue = m_c_interface->queue;
    LWNcommandBuffer *queueCB = m_c_interface->queueCB;
    lwnUtil::CommandBufferMemoryManager *cmdMem = m_c_interface->cmdMemMgr;
    LWNcommandHandle queueCBHandle;
    LWNbufferBuilder *bufferBuilder = lwnDeviceCreateBufferBuilder(device);
    LWNtextureBuilder *textureBuilder = lwnDeviceCreateTextureBuilder(device);

    // For "-debug", call generateDebugMessages() to generate some stupid
    // errors to exercise the callbacks.
    if (m_debug) {
        lwnDeviceInstallDebugCallback(device, debugCallback, (void *) 0x8675309, LWN_TRUE);
        generateDebugMessages();
    }

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator coherent_allocator(device, NULL, coherentPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Create programs from the device, provide them shader code and compile/link them
    LWNprogram *pgm = lwnDeviceCreateProgram(device);

    // XXX This is a hack because we don't have an IL. I'm just jamming through the strings 
    // as if they were an IL blob
    const char *sources[5];
    int nSources = 2;

    VertexShader vs(440);
    FragmentShader fs(440);
    GeometryShader gs(440);
    TessControlShader tcs(440);
    TessEvaluationShader tes(440);

    sources[0] = m_bindless ? vsstring_bindless : vsstring;
    sources[1] = m_bindless ? fsstring_bindless : fsstring;

    vs.addExtension(lwShaderExtension::LW_gpu_shader5);
    fs.addExtension(lwShaderExtension::LW_gpu_shader5);
    if (m_bindless) {
        // Required for using sampler2D in a UBO
        vs.addExtension(lwShaderExtension::LW_bindless_texture);
        fs.addExtension(lwShaderExtension::LW_bindless_texture);
    }

    vs << sources[0];
    fs << sources[1];

    if (m_geometryShader) {
        sources[nSources] = gsstring;
        gs << sources[nSources];
        nSources++;
    }
    if (m_tessControlShader) {
        sources[nSources] = tcsstring;
        tcs << sources[nSources];
        nSources++;
    }
    if (m_tessEvalShader) {
        sources[nSources] = tesstring;
        tes << sources[nSources];
        nSources++;
    }

    // Create a GLSLC helper instead of using the global version
    // since the device may be a debug device.
    lwnTest::GLSLCHelper glslcHelper(device, programPoolSize, g_glslcLibraryHelper, g_glslcHelperCache);
    LWNmemoryPool * scratchMemoryPool = lwnDeviceCreateMemoryPool(device, NULL, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    glslcHelper.SetShaderScratchMemory(scratchMemoryPool, 0, DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, queueCB); 

    LWNboolean compileSuccess = false;
    compileSuccess = glslcHelper.CompileAndSetShaders(pgm, vs, fs,
                            m_geometryShader ? gs : Shader(),
                            m_tessControlShader ? tcs : Shader(),
                            m_tessEvalShader ? tes : Shader());

    if (!compileSuccess) {
        log_output("Shader compile error. infoLog =\n%s\n", glslcHelper.GetInfoLog());
    }

    // Check to make sure program interfaces work as expected.
    LWNint location;
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
    LWNbuffer *vbo = coherent_allocator.allocBuffer(bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, sizeof(vertexData)+sizeof(texcoordData));

    // create persistent mapping
    void *ptr = lwnBufferMap(vbo);
    // fill ptr with vertex data followed by color data
    memcpy(ptr, vertexData, sizeof(vertexData));
    memcpy((char *)ptr + sizeof(vertexData), texcoordData, sizeof(texcoordData));

    // Get a handle to be used for setting the buffer as a vertex buffer
    LWNbufferAddress vboAddr = lwnBufferGetAddress(vbo);

    // Create an index buffer and fill it with data
    unsigned short indexData[6] = {0, 1, 2, 3, 4, 5};
    LWNbuffer *ibo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, indexData, sizeof(indexData),
                                        BUFFER_ALIGN_INDEX_BIT, true);

    // Get a handle to be used for setting the buffer as an index buffer
    LWNbufferAddress iboAddr = lwnBufferGetAddress(ibo);

    LWNtexture *rtTex = NULL;
    LWNtexture *depthTex = NULL;
    LWNtexture *tex1x = NULL;

    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetFlags(textureBuilder, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    lwnTextureBuilderSetSize2D(textureBuilder, offscreenWidth, offscreenHeight);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);

    if (m_multisample) {
        // Allocate a single-sample texture for the multi-texture configuration.
        tex1x = allocator.allocTexture(textureBuilder);

        // Set up the builder to create multisample textures.
        lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D_MULTISAMPLE);
        lwnTextureBuilderSetSamples(textureBuilder, 4);

        lwnMultisampleStateSetSamples(multisample, 4);
    }

    rtTex = allocator.allocTexture(textureBuilder);

    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_DEPTH24_STENCIL8);
    depthTex = allocator.allocTexture(textureBuilder);

    lwnCommandBufferSetRenderTargets(queueCB, 1, &rtTex, NULL, depthTex, NULL);

    LWNsamplerBuilder *samplerBuilder = lwnDeviceCreateSamplerBuilder(device);
    
    // Commented out experiments to test different state settings.    
    // lwnSamplerBuilderSetMinMagFilter(samplerBuilder, LWN_MIN_FILTER_NEAREST, LWN_MAG_FILTER_NEAREST);

    LWNsampler *sampler = lwnSamplerBuilderCreateSampler(samplerBuilder);
    LWNuint samplerID = lwnSamplerGetRegisteredID(sampler);

    const int texWidth = 4, texHeight = 4;
    lwnTextureBuilderSetDefaults(textureBuilder);
    lwnTextureBuilderSetTarget(textureBuilder, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(textureBuilder, LWN_FORMAT_RGBA8);
    lwnTextureBuilderSetSize2D(textureBuilder, texWidth, texHeight);
    LWNtexture *texture = allocator.allocTexture(textureBuilder);
    LWNuint textureID = lwnTextureGetRegisteredTextureID(texture);

    // Build a combined texture/sampler handle.
    LWNtextureHandle texHandle = lwnDeviceGetTextureHandle(device, textureID ,samplerID);

    lwnBufferBuilderSetDefaults(bufferBuilder);
    LWNbuffer *pbo = coherent_allocator.allocBuffer(bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, texWidth*texHeight*4);
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
    LWNbuffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &uboData, sizeof(uboData),
                                        BUFFER_ALIGN_UNIFORM_BIT, false);

    // Get a handle to be used for setting the buffer as a uniform buffer
    LWNbufferAddress uboAddr = lwnBufferGetAddress(ubo);

    // Some scissored clears
    {
        lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);
        LWNfloat clearColor[] = {0,0,0,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
        lwnCommandBufferClearDepthStencil(queueCB, 1.0, LWN_TRUE, 0, 0);
    }
    {
        lwnCommandBufferSetScissor(queueCB, offscreenWidth/2, 0, offscreenWidth/2, offscreenHeight/2);
        LWNfloat clearColor[] = {0,0.5,0,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    }
    {
        lwnCommandBufferSetScissor(queueCB, 0, offscreenHeight/2, offscreenWidth/2, offscreenHeight/2);
        LWNfloat clearColor[] = {0,0,0.5,1};
        lwnCommandBufferClearColor(queueCB, 0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    }
    lwnCommandBufferSetScissor(queueCB, 0, 0, offscreenWidth, offscreenHeight);
    lwnCommandBufferSetViewport(queueCB, 0, 0, offscreenWidth, offscreenHeight);

    if (m_benchmark) {
        lwnQueueFinish(queue);
    }
    clock_t startTime = clock();
    unsigned int numIterations = m_benchmark ? 10000000 : 1;

    LWNdrawPrimitive drawPrimitive = LWN_DRAW_PRIMITIVE_TRIANGLES;
    if (m_tessEvalShader) {
        LWNfloat levels[4] = { 2, 2, 2, 2 };
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
            cmdMem->populateCommandBuffer(cmd[i], CommandBufferMemoryManager::Coherent);
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
                cmdMem->populateCommandBuffer(cmd[i], CommandBufferMemoryManager::Coherent);
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
        }
    }

    if (m_multisample) {
        lwnCommandBufferDownsample(queueCB, rtTex, tex1x);

        // The contents of the lwrrently bound depth and colors are
        // not going to be needed after the Downsample operation.  The
        // below discard operations allow the GPU to not write the
        // corresponding GPU L2 cachelines into main memory.  This is
        // a potential memory bandwidth optimization.
        lwnCommandBufferDiscardColor(queueCB, 0);
        lwnCommandBufferDiscardDepthStencil(queueCB);
    }

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

#if 0
    if (m_multisample) {
        lwnQueuePresentTextureSync(queue, tex1x, NULL);
    } else {
        lwnQueuePresentTextureSync(queue, rtTex, NULL);
    }
    lwnQueueFinish(queue);
#else
    // Copy the resulting image back to system memory.
    int offscreenSize = offscreenWidth * offscreenHeight * 4;
    if (!offscreenImage) {
        offscreenImage = new char[offscreenSize];
    }
    if (offscreenImage)
    {
        lwnQueueFinish(queue);
        LWNtexture *texture = m_multisample ? tex1x : rtTex;
        ReadTextureDataRGBA8(device, queue, queueCB, texture, offscreenWidth, offscreenHeight, offscreenImage);
    }
#endif

    lwnProgramFree(pgm);
    lwnBlendStateFree(blend);
    lwnChannelMaskStateFree(cmask);
    lwnColorStateFree(color);
    lwnDepthStencilStateFree(depth);
    lwnPolygonStateFree(polygon);
    lwnMultisampleStateFree(multisample);
    lwnBufferBuilderFree(bufferBuilder);
    coherent_allocator.freeBuffer(vbo);
    coherent_allocator.freeBuffer(ibo);
    coherent_allocator.freeBuffer(pbo);
    coherent_allocator.freeBuffer(ubo);
    lwnTextureBuilderFree(textureBuilder);
    allocator.freeTexture(texture);
    allocator.freeTexture(rtTex);
    allocator.freeTexture(depthTex);
    lwnSamplerBuilderFree(samplerBuilder);
    lwnSamplerFree(sampler);
    if (m_multisample) {
        allocator.freeTexture(tex1x);
    }
    lwnMemoryPoolFree(scratchMemoryPool);
}

//////////////////////////////////////////////////////////////////////////

class LWNBasicTest : public LWNSampleTestConfig
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

    LWNBasicTest(LWNSampleTestConfig::SubmitMode submitMode, SpecialVariant special)
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

void LWNBasicTest::initGraphics()
{
    lwnDefaultInitGraphics();
    DisableLWNObjectTracking();

    m_c_interface->device = g_lwnDevice;
    m_c_interface->queue = g_lwnQueue;
    m_c_interface->queueCB = g_lwnQueueCB;
    m_c_interface->cmdMemMgr = &g_lwnCommandMem;
}

void LWNBasicTest::exitGraphics()
{
    g_lwnQueueCB->resetCounters();
    lwnDefaultExitGraphics();
}

int LWNBasicTest::isSupported()
{
    if (m_debug && !g_lwnDeviceCaps.supportsDebugLayer) {
        return 0;
    }
    return lwogCheckLWNAPIVersion(0,1);
}

lwString LWNBasicTest::getDescription()
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

    sb << "; test uses the C bindings for LWN.";

    return sb.str();
}

void LWNBasicTest::doGraphics()
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
            m_c_interface->device = testDevice->getDevice();
            m_c_interface->queue = testDevice->getQueue();
            m_c_interface->queueCB = testDevice->getQueueCB();
            m_c_interface->cmdMemMgr = testDevice->getCmdBufMemoryManager();
        } else {
            initFailed = true;
        }
    }

    if (!initFailed) {
        cDisplay();
    }

    if (m_debug) {
        delete testDevice;
        DeviceState::SetDefaultActive();

        // If our debug initialization failed, display a red window using the
        // window framebuffer class.
        if (initFailed) {
            QueueCommandBuffer &queueCB = *g_lwnQueueCB;
            g_lwnWindowFramebuffer.bind();
            lwnCommandBufferSetScissor(queueCB, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
            LWNfloat red[] = { 1.0, 0.0, 0.0, 1.0 };
            lwnCommandBufferClearColor(queueCB, 0, red, LWN_CLEAR_COLOR_MASK_RGBA);
            queueCB.submit();
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
OGTEST_CppTest(LWNBasicTest, lwn_basic_queue,   (LWNSampleTestConfig::QUEUE,   LWNBasicTest::NORMAL_TEST));
OGTEST_CppTest(LWNBasicTest, lwn_basic_command, (LWNSampleTestConfig::COMMAND, LWNBasicTest::NORMAL_TEST));
OGTEST_CppTest(LWNBasicTest, lwn_basic_command_transient, (LWNSampleTestConfig::COMMAND_TRANSIENT, LWNBasicTest::NORMAL_TEST));

// Special variants using miscellaneous features.
OGTEST_CppTest(LWNBasicTest, lwn_basic_bindless,    (LWNSampleTestConfig::QUEUE, LWNBasicTest::BINDLESS));
OGTEST_CppTest(LWNBasicTest, lwn_basic_debug,       (LWNSampleTestConfig::QUEUE, LWNBasicTest::DEBUG_LAYER));
OGTEST_CppTest(LWNBasicTest, lwn_basic_geometry,    (LWNSampleTestConfig::QUEUE, LWNBasicTest::GEOMETRY_SHADER));
OGTEST_CppTest(LWNBasicTest, lwn_basic_multisample, (LWNSampleTestConfig::QUEUE, LWNBasicTest::MULTISAMPLE));
OGTEST_CppTest(LWNBasicTest, lwn_basic_tess_tcs,    (LWNSampleTestConfig::QUEUE, LWNBasicTest::TESS_CONTROL_AND_EVALUATION_SHADER));
OGTEST_CppTest(LWNBasicTest, lwn_basic_tess_tesonly,(LWNSampleTestConfig::QUEUE, LWNBasicTest::TESS_EVALUATION_SHADER));
