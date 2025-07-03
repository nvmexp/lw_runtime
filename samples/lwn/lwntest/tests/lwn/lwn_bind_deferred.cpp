/*
 * Copyright (c) 2021, LWPU Corporation.  All rights reserved.
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

class LWNDeferredBindTest
{
    // What type of binding should we test for a given cell?  We use a separate
    // bind mode for each column.
    enum BindMode {
        BindModeImmediate,                  // test regular immediate binds, all units at once
        BindModeDeferred,                   // test deferred binds, all units at once
        BindModeDeferredConselwtive,        // test deferred binds, one unit at a time
        BindModeDeferredCallCommands,       // test deferred binds imported by CallCommands
        BindModeDeferredCopyCommands,       // test deferred binds imported by CopyCommands
        TotalBindModeCount,
    };
    static const int cellsX = TotalBindModeCount;

    // What shader type should we test for a given cell?  We use a different
    // shader type for each row.
    enum ShaderType {
        ShaderTypeFragment,
        ShaderTypeVertex,
        ShaderTypeCompute,
        TotalShaderTypeCount,
    };
    static const int cellsY = TotalShaderTypeCount;

    // We check for correct deferred binds by creating different resources that
    // have one of several canned colors and testing that the values are
    // returned properly.
    static const int nColors = 4;
    static const dt::vec4 testColorValues[];

    // We use 4x4 constant color input textures for texture/image lookup tests.
    static const int inputTexSize = 4;

    // We use 8x8 output textures (written by image stores) to display our
    // pass/fail results.  We have a 4x4 grid to test each combination of colors
    // obtained from textures and samplers, and use a 2x2 grid within each of
    // those cells to check both a texture lookup result and a sampler border
    // color lookup result.
    static const int resTexSize = nColors * 2;
    void doGraphicsWithTemporaryDevice() const;

    // Structure holding arrays of bindings for each resource type tested by
    // this test.
    struct BindingSet {
        TextureHandle combinedHandles[nColors * nColors];
        SeparateTextureHandle texOnlyHandles[nColors];
        SeparateSamplerHandle samplerOnlyHandles[nColors];
        ImageHandle imageHandles[nColors];
        BufferRange uboRanges[nColors];
        BufferRange ssboRanges[nColors];

        // Fill a binding set by copying from <from>, but rotating the binding
        // arrays by <rotate>.
        void copyRotated(const BindingSet *from, int rotate)
        {
            for (int i = 0; i < nColors; i++) {
                int rotated = (i + rotate) % nColors;
                texOnlyHandles[i] = from->texOnlyHandles[rotated];
                samplerOnlyHandles[i] = from->samplerOnlyHandles[rotated];
                imageHandles[i] = from->imageHandles[rotated];
                uboRanges[i] = from->uboRanges[rotated];
                ssboRanges[i] = from->ssboRanges[rotated];
            }
            for (int i = 0; i < nColors * nColors; i++) {
                int rotated = (i + rotate) % (nColors * nColors);
                combinedHandles[i] = from->combinedHandles[rotated];
            }
        }
    };

    // Make calls to bind all of the resources in the binding set using
    // immediate multi-bind calls.
    void bindAllImmediate(CommandBuffer *cmdBuf, ShaderStage stage, const BindingSet *bindings) const
    {
        cmdBuf->BindTextures(stage, 0, nColors * nColors, bindings->combinedHandles);
        cmdBuf->BindSeparateTextures(stage, 0, nColors, bindings->texOnlyHandles);
        cmdBuf->BindSeparateSamplers(stage, 0, nColors, bindings->samplerOnlyHandles);
        cmdBuf->BindUniformBuffers(stage, 0, nColors, bindings->uboRanges);
        cmdBuf->BindStorageBuffers(stage, 0, nColors, bindings->ssboRanges);
        cmdBuf->BindImages(stage, 0, nColors, bindings->imageHandles);
    }

    // Make calls to bind all of the resources in the binding set using deferred
    // multi-bind calls.
    void bindAllDeferred(CommandBuffer *cmdBuf, ShaderStage stage, const BindingSet *bindings) const
    {
        if (m_fastpath) {
            cmdBuf->BindTexturesDeferred_fastpath(stage, 0, nColors * nColors, bindings->combinedHandles);
            cmdBuf->BindSeparateTexturesDeferred_fastpath(stage, 0, nColors, bindings->texOnlyHandles);
            cmdBuf->BindSeparateSamplersDeferred_fastpath(stage, 0, nColors, bindings->samplerOnlyHandles);
            cmdBuf->BindUniformBuffersDeferred_fastpath(stage, 0, nColors, bindings->uboRanges);
            cmdBuf->BindStorageBuffersDeferred_fastpath(stage, 0, nColors, bindings->ssboRanges);
            cmdBuf->BindImagesDeferred_fastpath(stage, 0, nColors, bindings->imageHandles);
        } else {
            cmdBuf->BindTexturesDeferred(stage, 0, nColors * nColors, bindings->combinedHandles);
            cmdBuf->BindSeparateTexturesDeferred(stage, 0, nColors, bindings->texOnlyHandles);
            cmdBuf->BindSeparateSamplersDeferred(stage, 0, nColors, bindings->samplerOnlyHandles);
            cmdBuf->BindUniformBuffersDeferred(stage, 0, nColors, bindings->uboRanges);
            cmdBuf->BindStorageBuffersDeferred(stage, 0, nColors, bindings->ssboRanges);
            cmdBuf->BindImagesDeferred(stage, 0, nColors, bindings->imageHandles);
        }
    }

    // Make calls to bind all of the resources in the binding set using deferred
    // multi-bind calls, but binding only a single resource at a time.
    void bindAllDeferredConselwtive(CommandBuffer *cmdBuf, ShaderStage stage, const BindingSet *bindings) const
    {
        if (m_fastpath) {
            for (int i = 0; i < nColors * nColors; i++) {
                cmdBuf->BindTexturesDeferred_fastpath(stage, i, 1, bindings->combinedHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindSeparateTexturesDeferred_fastpath(stage, i, 1, bindings->texOnlyHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindSeparateSamplersDeferred_fastpath(stage, i, 1, bindings->samplerOnlyHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindUniformBuffersDeferred_fastpath(stage, i, 1, bindings->uboRanges + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindStorageBuffersDeferred_fastpath(stage, i, 1, bindings->ssboRanges + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindImagesDeferred_fastpath(stage, i, 1, bindings->imageHandles + i);
            }
        } else {
            for (int i = 0; i < nColors * nColors; i++) {
                cmdBuf->BindTexturesDeferred(stage, i, 1, bindings->combinedHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindSeparateTexturesDeferred(stage, i, 1, bindings->texOnlyHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindSeparateSamplersDeferred(stage, i, 1, bindings->samplerOnlyHandles + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindUniformBuffersDeferred(stage, i, 1, bindings->uboRanges + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindStorageBuffersDeferred(stage, i, 1, bindings->ssboRanges + i);
            }
            for (int i = 0; i < nColors; i++) {
                cmdBuf->BindImagesDeferred(stage, i, 1, bindings->imageHandles + i);
            }
        }
    }

    // Do we use "fastpath" entry points for deferred binds?
    bool m_fastpath;

public:
    LWNDeferredBindTest(bool fastpath) : m_fastpath(fastpath) {}
    LWNTEST_CppMethods();
};

// Canned colors (red, green, blue, and white) exercised by this test.
const dt::vec4 LWNDeferredBindTest::testColorValues[] = {
    dt::vec4(1.0, 0.0, 0.0, 1.0),
    dt::vec4(0.0, 1.0, 0.0, 1.0),
    dt::vec4(0.0, 0.0, 1.0, 1.0),
    dt::vec4(1.0, 1.0, 1.0, 1.0),
};

lwString LWNDeferredBindTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test exercising the " <<
        (m_fastpath ? "fastpath" : "regular") <<
        "version of the deferred bind API entry points. This test exercises "
        "binds of separate and combined texture handles, separate sampler handles, "
        "image handles, uniform buffer ranges, and storage buffer ranges.  Each "
        "cell tests a variety of resource bindings, with the shader verifying "
        "whether the values it reads back match expected values.  A pixel will be "
        "rendered in green if all lookups produce expected results and red otherwise.  "
        "The columns in this test exercise different bind types (immediate, deferred, "
        "deferred one item at a time, deferred submitted with CallCommands and/or "
        "CopyCommands).  The rows exercise different shader types (fragment, vertex, "
        "compute).";
    return sb.str();
}

int LWNDeferredBindTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(55, 10);
}

// We need to do our main <doGraphics> logic in a separate LWN device because
// our default device in lwntest does not enable separate textures/samplers,
// which are tested here.
void LWNDeferredBindTest::doGraphicsWithTemporaryDevice() const
{
    ct_assert(__GL_ARRAYSIZE(testColorValues) == nColors);
    cellTestInit(cellsX, cellsY);

    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Generate common shader code used to test the bindings for a passed-in
    // (x,y) value that tests a set of bindings corresponding to the passed
    // in values and then uses an image store to write red/green results
    // to that (x,y) coordinate in our result image.
    lwStringBuf selfCheckCode;
    selfCheckCode << "const vec4 testColors[] = {\n";
    for (size_t i = 0; i < __GL_ARRAYSIZE(testColorValues); i++) {
        selfCheckCode <<
            "  vec4(" <<
            testColorValues[i][0] << ", " << testColorValues[i][1] << ", " <<
            testColorValues[i][2] << ", " << testColorValues[i][3] << "),\n";
    }
    selfCheckCode <<
        "};\n"
        "\n"
        "layout(binding=0) uniform sampler2D combinedtex[" << nColors << "];\n"
        "layout(binding=0) uniform texture2D septex[" << nColors << "];\n"
        "layout(binding=0) uniform sampler sepsmp[" << nColors << "];\n"
        "layout(binding=0, rgba8) uniform image2D img[" << nColors << "];\n"
        "layout(binding=0) uniform UBO { vec4 color; } ubo[" << nColors << "];\n"
        "layout(binding=0) buffer SSBO { vec4 color; } ssbo[" << nColors << "];\n"
        "layout(binding=" << nColors << ") writeonly uniform image2D resimg;\n"
        "\n"
        "bool checkColor(vec4 color, int expectedIndex) {\n"
        "  return all(lessThanEqual(abs(color - testColors[expectedIndex]), vec4(0.005)));\n"
        "}\n"
        "\n"

        // Our test exercising N test color values uses a grid of 2N x 2N
        // "sub-cells".
        //
        // When testing textures, each 2x2 array of sub-cells tests a unique
        // combination of the N textures and N samplers.  Half of this 2x2 array
        // tests the "texture" portion by doing a lookup at (0.5, 0.5) and
        // expecting to see a color from one of the constant-colored textures.
        // The other half tests the "sampler" portion by doing a lookup at (1.5,
        // 1.5) and expecting to see the border color from one of the samplers.
        //
        // When testing images, uniform buffers, and storage buffers, we use the
        // same grid layout but are effectively only testing the "texture"
        // portion.
        "void checkAll(int x, int y)\n"
        "{\n"
        "  bool pass = true;\n"
        "  int texIndex = x / 2;\n"
        "  int smpIndex = y / 2;\n"
        "  int combinedIndex = texIndex + smpIndex * 4;\n"
        "  bool testSampler = (0 != ((x + y) & 1));\n"
        "  vec2 tc = testSampler ? vec2(1.5) : vec2(0.5);\n"

        // Test combined texture handles.
        "  vec4 value1 = texture(combinedtex[combinedIndex], tc);\n"
        "  pass = pass && checkColor(value1, testSampler ? smpIndex : texIndex);\n"

        // Construct and test a combination of separate texture/sampler handles.
        "  sampler2D combined = sampler2D(septex[texIndex], sepsmp[smpIndex]);\n"
        "  vec4 value2 = texture(combined, tc);\n"
        "  pass = pass && checkColor(value2, testSampler ? smpIndex : texIndex);\n"

        // Test image loads.
        "  vec4 value3 = imageLoad(img[texIndex], ivec2(0,0));\n"
        "  pass = pass && checkColor(value3, texIndex);\n"

        // Test uniform buffer loads.
        "  vec4 value4 = ubo[texIndex].color;"
        "  pass = pass && checkColor(value4, texIndex);\n"

        // Test shader storage buffer loads.
        "  vec4 value5 = ssbo[texIndex].color;"
        "  pass = pass && checkColor(value5, texIndex);\n"

        // Compute a final color for this sub-cell.  By default we display
        // red/green based on <pass> (green if all lookups behaved as expected).
        "  vec4 color = vec4(pass ? 0.0 : 1.0, pass ? 1.0 : 0.0, 0.0, 1.0);\n";

#define OVERRIDE_DISPLAY_RESULT     0
#if OVERRIDE_DISPLAY_RESULT != 0
    // Debug hack to directly display the results of a texture, image, uniform
    // buffer, or storage buffer lookup instead of red/green self-check results.
    // Set to values 1, 2, 3, 4, or 5 to display specific lookup values (1 =
    // value1).  The default value of 0 disables this feature.
#define QUOTE(name)                 #name
#define STR(macro)                  QUOTE(macro)
#define OVERRIDE_DISPLAY_VAR_NAME   "value" STR(OVERRIDE_DISPLAY_RESULT)
    selfCheckCode << "  color = " << OVERRIDE_DISPLAY_VAR_NAME << ";\n";
#endif

    selfCheckCode <<
        "  ivec2 icoord = ivec2(x,y);\n"
        "  imageStore(resimg, icoord, color);\n"
        "}\n";

    // Basic compute shader where each invocation computes a red/green result
    // for a single result texel.
    ComputeShader cstest(450);
    cstest.setCSGroupSize(resTexSize, resTexSize);
    cstest.addExtension(lwShaderExtension::LW_separate_texture_types);
    cstest.addExtension(lwShaderExtension::LW_bindless_texture);
    cstest << selfCheckCode.str();
    cstest <<
        "void main() {\n"
        "  int x = int(gl_LocalIlwocationID.x);\n"
        "  int y = int(gl_LocalIlwocationID.y);\n"
        "  checkAll(x, y);\n"
        "}\n";
    Program *cstestPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(cstestPgm, cstest)) {
        cstestPgm->Free();
        cstestPgm = NULL;
    }

    // Basic compute shader where each vertex triggers an invocation computing a
    // red/green result for a single result texel.
    VertexShader vstest(450);
    vstest.addExtension(lwShaderExtension::LW_separate_texture_types);
    vstest.addExtension(lwShaderExtension::LW_bindless_texture);
    vstest << selfCheckCode.str();
    vstest <<
        "void main() {\n"
        "  int x = gl_VertexID % " << resTexSize << ";\n"
        "  int y = gl_VertexID / " << resTexSize << ";\n"
        "  checkAll(x, y);\n"
        "}\n";
    Program *vstestPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(vstestPgm, vstest)) {
        vstestPgm->Free();
        vstestPgm = NULL;
    }

    // Basic fragment shader test where each fragment triggers an invocation
    // computing a red/green result for a single result texel.  We use a dumb
    // pass-through vertex shader to draw a full-viewport triangle strip.
    VertexShader vspass(450);
    vspass <<
        "void main() {\n"
        "  int x = gl_VertexID % 2;\n"
        "  int y = gl_VertexID / 2;\n"
        "  gl_Position = vec4(2.0 * x - 1.0, 2.0 * y - 1.0, 0.0, 1.0);\n"
        "}\n";
    FragmentShader fstest(450);
    fstest.addExtension(lwShaderExtension::LW_separate_texture_types);
    fstest.addExtension(lwShaderExtension::LW_bindless_texture);
    fstest << selfCheckCode.str();
    fstest <<
        "void main() {\n"
        "  int x = int(gl_FragCoord.x);\n"
        "  int y = int(gl_FragCoord.y);\n"
        "  checkAll(x, y);\n"
        "  discard;\n"
        "}\n";
    Program *fstestPgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(fstestPgm, vspass, fstest)) {
        fstestPgm->Free();
        fstestPgm = NULL;
    }

    // Allocate a sampler for each of the <nColors> color values for the test.
    // The samplers will differ only in border color values.
    SamplerBuilder sb;
    Sampler *samplers[nColors];
    sb.SetDevice(device).SetDefaults().
        SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER).
        SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    for (int i = 0; i < nColors; i++) {
        sb.SetBorderColor(testColorValues[i].ptr());
        samplers[i] = sb.CreateSampler();
    }

    // Allocate a texture for each of the <nColors> color values for the test.
    // The contents of the textures will differ only in color values -- each
    // texture will be filled with the single color for the corresponding entry
    // in the color value array.
    TextureBuilder tb;
    Texture *textures[nColors];
    tb.SetDevice(device).SetDefaults().SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(inputTexSize, inputTexSize).SetFormat(Format::RGBA8).
        SetLevels(1).SetFlags(TextureFlags::IMAGE);
    size_t texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, nColors * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    for (int i = 0; i < nColors; i++) {
        textures[i] = texAllocator.allocTexture(&tb);
    }

    // Allocate and fill a buffer object with texel values for each of the input
    // textures we just allocated, and then download the values to the texture.
    BufferBuilder texbb;
    size_t colorDataSizePerTexture = inputTexSize * inputTexSize * sizeof(dt::u8lwec4);
    size_t colorDataSize = nColors * colorDataSizePerTexture;
    texbb.SetDevice(device).SetDefaults();
    MemoryPoolAllocator texSrcAllocator(device, NULL, colorDataSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *texSrcDataBuffer = texSrcAllocator.allocBuffer(&texbb, BUFFER_ALIGN_COPY_READ_BIT, colorDataSize);
    BufferAddress texSrcDataBufferAddr = texSrcDataBuffer->GetAddress();
    dt::u8lwec4 *texSrcBase = (dt::u8lwec4 *) texSrcDataBuffer->Map();
    dt::u8lwec4 *texSrcData = texSrcBase;
    CopyRegion copyRegion = { 0, 0, 0, inputTexSize, inputTexSize, 1 };
    for (int i = 0; i < nColors; i++) {
        dt::u8lwec4 texelData = dt::u8lwec4(testColorValues[i][0], testColorValues[i][1],
                                            testColorValues[i][2], testColorValues[i][3]);
        for (int j = 0; j < inputTexSize * inputTexSize; j++) {
            *texSrcData++ = texelData;
        }
        queueCB.CopyBufferToTexture(texSrcDataBufferAddr + i * colorDataSizePerTexture,
                                    textures[i], NULL, &copyRegion, CopyFlags::NONE);
    }

    // Allocate and fill a buffer object used to hold values matching our color
    // array entries for UBO and SSBO accesses.
    BufferBuilder bb;
    size_t databufBytesPerColor = 512;
    size_t databufSize = nColors * databufBytesPerColor;
    bb.SetDevice(device).SetDefaults();
    MemoryPoolAllocator databufAllocator(device, NULL, databufSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *databuf = databufAllocator.allocBuffer(&bb, BufferAlignBits(BUFFER_ALIGN_UNIFORM_BIT | BUFFER_ALIGN_SHADER_STORAGE_BIT), databufSize);
    BufferAddress databufAddr = databuf->GetAddress();
    dt::vec4 *databufBase = (dt::vec4 *) databuf->Map();
    dt::vec4 *databufPtr = databufBase;
    for (int i = 0; i < nColors; i++) {
        for (size_t j = 0; j < databufBytesPerColor / sizeof(dt::vec4); j++) {
            *databufPtr++ = testColorValues[i];
        }
    }

    // Allocate one texture per cell to receive test results via image stores.
    TextureBuilder restb;
    Texture *restex[cellsX][cellsY];
    ImageHandle restexImage[cellsX][cellsY];
    restb.SetDevice(device).SetDefaults().SetTarget(TextureTarget::TARGET_2D).
        SetSize2D(resTexSize, resTexSize).SetFormat(Format::RGBA8).SetLevels(1).
        SetFlags(TextureFlags::IMAGE);
    size_t restexStorageSize = restb.GetPaddedStorageSize();
    MemoryPoolAllocator restexAllocator(device, NULL, cellsX * cellsY * restexStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    for (int x = 0; x < cellsX; x++) {
        for (int y = 0; y < cellsY; y++) {
            restex[x][y] = restexAllocator.allocTexture(&restb);
            restexImage[x][y] = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(restex[x][y]));
        }
    }

    // Set up an arrays of "expected" bindings for each resource type, iterating
    // over all of our canned colors.
    BindingSet expectedBindings;
    for (int i = 0; i < nColors; i++) {
        expectedBindings.texOnlyHandles[i] = device->GetSeparateTextureHandle(textures[i]->GetRegisteredTextureID());
        expectedBindings.samplerOnlyHandles[i] = device->GetSeparateSamplerHandle(samplers[i]->GetRegisteredID());
        expectedBindings.imageHandles[i] = device->GetImageHandle(g_lwnTexIDPool->RegisterImage(textures[i]));
        expectedBindings.uboRanges[i].address = databufAddr + i * databufBytesPerColor;
        expectedBindings.uboRanges[i].size = databufBytesPerColor;
        expectedBindings.ssboRanges[i].address = databufAddr + i * databufBytesPerColor;
        expectedBindings.ssboRanges[i].size = databufBytesPerColor;
        for (int j = 0; j < nColors; j++) {
            expectedBindings.combinedHandles[i + nColors * j] =
                device->GetTextureHandle(textures[i]->GetRegisteredTextureID(),
                                         samplers[j]->GetRegisteredID());
        }
    }

    // Set up an alternate "rotated" binding set that will pick the wrong
    // binding for each entry.
    BindingSet rotatedBindings;
    rotatedBindings.copyRotated(&expectedBindings, 1);

    // Set up separate arrays of bindings for each cell, which will be used for
    // deferred binds.  We initialize each array with rotated bindings that we
    // will overwrite after recording but before submission.
    BindingSet testBindings[cellsX * cellsY];
    for (int i = 0; i < cellsX * cellsY; i++) {
        testBindings[i].copyRotated(&expectedBindings, 1);
    }

    // Set up separate API command buffers used to record binding and draw calls
    // that will either be submitted directly to queues or pulled into another
    // API command buffer via CallCommands or CopyCommands.
    CommandBuffer *apiCmdBuf = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(apiCmdBuf, CommandBufferMemoryManager::Coherent);
    CommandBuffer *callCmdBuf = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(callCmdBuf, CommandBufferMemoryManager::Coherent);
    CommandBuffer *copyCmdBuf = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(copyCmdBuf, CommandBufferMemoryManager::NonCoherent);

    // Submit a few commands via the queue command buffer.  To facilitate the
    // fragment shader test, we set up our render targets with nothing bound and
    // a viewport that matches the size of the result texture.
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.SetRenderTargets(0, NULL, NULL, NULL, NULL);
    queueCB.SetViewportScissor(0, 0, resTexSize, resTexSize);
    queueCB.submit();

    // Find the window texture, where we will copy results for each cell from
    // its result texture.
    Texture *wintex = g_lwnWindowFramebuffer.getAcquiredTexture();

    apiCmdBuf->BeginRecording();
    for (int cy = 0; cy < cellsY; cy++) {

        // For each row, bind a program appropriate for the shader type being
        // tested.
        ShaderStage stage;
        switch (cy) {
        case ShaderTypeFragment:
            stage = ShaderStage::FRAGMENT;
            apiCmdBuf->BindProgram(fstestPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case ShaderTypeVertex:
            stage = ShaderStage::VERTEX;
            apiCmdBuf->BindProgram(vstestPgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case ShaderTypeCompute:
            stage = ShaderStage::COMPUTE;
            apiCmdBuf->BindProgram(cstestPgm, ShaderStageBits::COMPUTE);
            break;
        default:
            assert(0);
            break;
        }

        for (int cx = 0; cx < cellsX; cx++) {
            int cellIndex = cy * cellsY + cx;
            BindMode bindMode = BindMode(cx);

            // Pick a command buffer to record commands to draw the cell.  By
            // default, we use the API command buffer we created, but we use one
            // of the secondary command buffers for call/copy tests.
            CommandBuffer *cellCmdBuf = apiCmdBuf;
            switch (bindMode) {
            case BindModeDeferredCallCommands:
                cellCmdBuf = callCmdBuf;
                callCmdBuf->BeginRecording();
                break;
            case BindModeDeferredCopyCommands:
                cellCmdBuf = copyCmdBuf;
                copyCmdBuf->BeginRecording();
                break;
            default:
                break;
            }

            // Submit commands to do the type of binds being tested.
            switch (bindMode) {
            case BindModeImmediate:
                testBindings[cellIndex].copyRotated(&expectedBindings, 0);
                bindAllImmediate(cellCmdBuf, stage, testBindings + cellIndex);
                break;
            case BindModeDeferred:
            case BindModeDeferredCallCommands:
            case BindModeDeferredCopyCommands:
                bindAllImmediate(cellCmdBuf, stage, &rotatedBindings);  // set "bad" bindings first
                bindAllDeferred(cellCmdBuf, stage, testBindings + cellIndex);
            case BindModeDeferredConselwtive:
                bindAllImmediate(cellCmdBuf, stage, &rotatedBindings);  // set "bad" bindings first
                bindAllDeferredConselwtive(cellCmdBuf, stage, testBindings + cellIndex);
                break;
            default:
                assert(0);
                break;
            }

            // Bind the result texture for the cell as an image.
            cellCmdBuf->BindImage(stage, nColors, restexImage[cx][cy]);

            // Submit the appropriate type of draw/dispatch to trigger the
            // appropriate shaders.
            switch (cy) {
            case ShaderTypeFragment:
                if (fstestPgm) {
                    cellCmdBuf->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
                }
                break;
            case ShaderTypeVertex:
                if (vstestPgm) {
                    cellCmdBuf->DrawArrays(DrawPrimitive::POINTS, 0, resTexSize * resTexSize);
                }
                break;
            case ShaderTypeCompute:
                if (cstestPgm) {
                    cellCmdBuf->DispatchCompute(1, 1, 1);
                }
                break;
            default:
                assert(0);
                break;
            }

            // Copy the result texture to the appropriate region of the window.
            CopyRegion srcRegion = { 0, 0, 0, resTexSize, resTexSize, 1 };
            CopyRegion dstRegion = { 0, 0, 0, 0, 0, 1 };
            cellGetRectPadded(cx, cy, 4,
                              &dstRegion.xoffset, &dstRegion.yoffset,
                              &dstRegion.width, &dstRegion.height);
            cellCmdBuf->CopyTextureToTexture(restex[cx][cy], NULL, &srcRegion,
                                             wintex, NULL, &dstRegion, CopyFlags::NONE);

            // For call/copy command cells, finish recording the secondary
            // command buffer and submit to the main API command buffer.
            CommandHandle secondaryHandle;
            switch (bindMode) {
            case BindModeDeferredCallCommands:
                secondaryHandle = callCmdBuf->EndRecording();
                apiCmdBuf->CallCommands(1, &secondaryHandle);
                break;
            case BindModeDeferredCopyCommands:
                secondaryHandle = copyCmdBuf->EndRecording();
                apiCmdBuf->CopyCommands(1, &secondaryHandle);
                break;
            default:
                break;
            }
        }

        // Unbind the program used for this row.
        switch (cy) {
        case ShaderTypeFragment:
        case ShaderTypeVertex:
            apiCmdBuf->BindProgram(NULL, ShaderStageBits::ALL_GRAPHICS_BITS);
            break;
        case ShaderTypeCompute:
            apiCmdBuf->BindProgram(NULL, ShaderStageBits::COMPUTE);
            break;
        default:
            assert(0);
            break;
        }
    }

    // Finish recording our API command buffer, at a time when each of the test
    // binding sets still have "bad" rotated bindings.
    CommandHandle testCommands = apiCmdBuf->EndRecording();

    // Before submitting, fix up the binding sets to hold the correct
    // non-rotated values.
    for (int i = 0; i < cellsX * cellsY; i++) {
        testBindings[i].copyRotated(&expectedBindings, 0);
    }

    queue->SubmitCommands(1, &testCommands);

    // Switch back to the window framebuffer, finish our previous submitted
    // rendering, and clean up any resources not cleaned up by allocators.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queue->Finish();
    apiCmdBuf->Free();
    callCmdBuf->Free();
}


void LWNDeferredBindTest::doGraphics() const
{
    // Set up a temporary device state that behaves like our default device,
    // except that it also sets the separate texture support flag that is needed
    // by this test.
    DisableLWNObjectTracking();
    DeviceFlagBits deviceFlags = (DeviceState::GetActive()->getDeviceFlags() |
                                  DeviceFlagBits::ENABLE_SEPARATE_SAMPLER_TEXTURE_SUPPORT);
    DeviceState *tmpDeviceState = new DeviceState(deviceFlags);
    if (!tmpDeviceState || !tmpDeviceState->isValid()) {
        delete tmpDeviceState;
        DeviceState::SetDefaultActive();
        LWNFailTest();
        EnableLWNObjectTracking();
        return;
    }
    tmpDeviceState->SetActive();
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();

    // Do rendering with the temporary device state.
    doGraphicsWithTemporaryDevice();

    // Clean up and return to our normal processing.
    delete tmpDeviceState;
    DeviceState::SetDefaultActive();
    g_lwnQueueCB->resetCounters();
    EnableLWNObjectTracking();
}

OGTEST_CppTest(LWNDeferredBindTest, lwn_bind_deferred,          (false));
OGTEST_CppTest(LWNDeferredBindTest, lwn_bind_deferred_fastpath, (true));
