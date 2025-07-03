/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

class LWNImageSwizzleTest
{
    static const int cellSize = 32;         // render in 32x32 cells with 4-pixel margins
    static const int cellMargin = 4;
    const int testImageWidth = 2;           // all of our test textures are 2x2 and consume at most 512B
    const int testImageHeight = 2;
    const size_t testImageMaxBytes = 512;

public:
    // ImageHandleType -- Should image loads/stores be tested with dedicated
    // image handles or with texture handles?
    enum ImageHandleType {
        ImageHandles,
        TextureHandles,
    };

    // Different shader types we generate for this test.
    enum ShaderType {
        ShaderTypeVertexPassthrough,        // basic pass-through vertex shader
        ShaderTypeFragmentTexelFetch,       // fragment shader displaying a texel loaded via texelFetch()
        ShaderTypeFragmentImageLoad,        // fragment shader displaying a texel loaded via imageLoad()
        ShaderTypeFragmentImageStore,       // fragment shader storing texel values using imageStore()
        ShaderTypeFragmentShaderOutput,     // fragment shader writing texel values using fragment shader outputs
    };
private:
    // Information on the formats that we will test here.
    struct FormatInfo {
        Format format;
        const char *layout;
    };
    static const FormatInfo testFormats[];
    static const int nTestFormats;

    // Information on the swizzles that we will test here.
    struct SwizzleInfo {
        TextureSwizzle swizzles[4];
    };
    static const SwizzleInfo testSwizzles[];
    static const int nTestSwizzles;

    ImageHandleType m_handleType;
    Program *mkprogram(Device *device, ShaderType shaderType, const char *formatQualifier) const;
    void setCellViewportScissor(CommandBuffer &cb, int cellX, int cellY) const;
    void showTexture(CommandBuffer &cb, int cellX, int cellY, Texture *tex) const;

public:
    LWNImageSwizzleTest(ImageHandleType handleType) : m_handleType(handleType) {}
    LWNTEST_CppMethods();
};

const LWNImageSwizzleTest::FormatInfo LWNImageSwizzleTest::testFormats[] = {
    { Format::R8, "r8" },
    { Format::RGBA8, "rgba8" },
    { Format::R32F, "r32f" },
    { Format::RG32F, "rg32f" },
    { Format::RGBA32F, "rgba32f" },
};
const int LWNImageSwizzleTest::nTestFormats = __GL_ARRAYSIZE(LWNImageSwizzleTest::testFormats);

const LWNImageSwizzleTest::SwizzleInfo LWNImageSwizzleTest::testSwizzles[] = {
    {{ TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }},
    {{ TextureSwizzle::R, TextureSwizzle::R, TextureSwizzle::R, TextureSwizzle::R }},
    {{ TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::R, TextureSwizzle::A }},
    {{ TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A, TextureSwizzle::R }},
};
const int LWNImageSwizzleTest::nTestSwizzles = __GL_ARRAYSIZE(LWNImageSwizzleTest::testSwizzles);

lwString LWNImageSwizzleTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "This tests exercises the interaction of texture swizzles and image loads/stores.  "
        "The image has two halves, where the bottom half exercises texel fetches and image "
        "loads and the top half exercises image stores.  Each row tests a different format "
        "(lwrrently R8, RGBA8, R32F, RG32F, and RGBA32F from top to bottom).\n"
        "\n"
        "In the bottom half, the leftmost cell displays a 2x2 texture initialized with red, "
        "green, blue, and 'alpha' colors.  The remaining groups of 3 cells test texel fetches, "
        "image loads using EXT_shader_image_load_formatted, and basic image loads using a format "
        "layout qualifier, each with a specific swizzle pattern.  These cell images are arranged "
        "in a 4x4 grid, where the four rows of each column display the red, green, blue, and "
        "alpha components of one of the four texels.  The four swizzle patterns from left to "
        "right are RGBA (default), RRRR, BGRA, and GBAR.\n"
        "\n"
        "In the top half, each group of four cells tests the same swizzles as in the load tests.  "
        "The left pair displays a texture initialized with image stores and a shader without a "
        "format layout qualifier and the right pair does the same with a format layout qualifier. "
        "Within each pair, the left half displays the 2x2 texture directly while the right half "
        "displays the same 4x4 grid as in the bottom half.";
    return sb.str();
}

int LWNImageSwizzleTest::isSupported() const
{
    if (m_handleType == TextureHandles && !g_lwnDeviceCaps.supportsImagesUsingTextureHandles) {
        return 0;
    }
    // Test is using GL_EXT_shader_image_load_formatted that is not supported on Kepler
    return g_lwnDeviceCaps.supportsMaxwell2Features &&
           lwogCheckLWNAPIVersion(21, 5);
}

// Display the contents of a texture in cell (cellX, cellY) using a basic CopyTextureToTexture
// targeting the display texture.
void LWNImageSwizzleTest::showTexture(CommandBuffer &cb, int cellX, int cellY, Texture *tex) const
{
    int ox = cellX * cellSize + cellMargin;
    int oy = cellY * cellSize + cellMargin;
    int os = cellSize - 2 * cellMargin;
    CopyRegion srcRegion = { 0, 0, 0, testImageWidth, testImageHeight, 1 };
    CopyRegion dstRegion = { ox, oy, 0, os, os, 1 };
    cb.CopyTextureToTexture(tex, NULL, &srcRegion,
                            g_lwnWindowFramebuffer.getAcquiredTexture(), NULL, &dstRegion,
                            CopyFlags::ENGINE_2D);
}

// Generate an LWN program object containing the shader type specified by
// <shaderType>, using an optional qualifier <formatQualifier>.
lwn::Program * LWNImageSwizzleTest::mkprogram(Device *device, ShaderType shaderType, const char *formatQualifier) const
{
    Shader s;
    Program *program = device->CreateProgram();
    if (!program) {
        LWNFailTest();
        return NULL;
    }

    lwStringBuf layout;
    if (formatQualifier) {
        layout << "layout(" << formatQualifier << ") ";
    }

    switch (shaderType) {

    case ShaderTypeVertexPassthrough:
        // The passthrough vertex shader simply passes through a 3-component
        // position and a 2-component texture coordinate.
        s = VertexShader(450);
        s <<
            "layout(location=0) in vec3 position;\n"
            "out vec2 tc;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  tc = 0.5 * position.xy + 0.5;\n"
            "}\n";
        break;

    case ShaderTypeFragmentTexelFetch:
    case ShaderTypeFragmentImageLoad:
        // The texel fetch and image load shaders remap the interpolated texture
        // coordinate to a specific texel and component, do a lookup, and then
        // display the appropriate component.
        s = FragmentShader(450);
        if (!formatQualifier) {
            s.addExtension(lwShaderExtension::EXT_shader_image_load_formatted);
        }
        s <<
            "in vec2 tc;\n"
            "out vec4 color;\n";
        if (shaderType == ShaderTypeFragmentTexelFetch) {
            s << "layout(binding=0) uniform sampler2D tex;\n";
        } else {
            s << layout.str() << "layout(binding=0) uniform image2D image;\n";
        }
        s <<
            "void main() {\n"
            "  ivec2 itc = ivec2(4.0 * tc);\n"
            "  ivec2 lc = ivec2(itc.x & 1, itc.x >> 1);\n";
        if (shaderType == ShaderTypeFragmentTexelFetch) {
            s << "  vec4 texel = texelFetch(tex, lc, 0);\n";
        } else {
            s << "  vec4 texel = imageLoad(image, lc);\n";
        }
        s <<
            "  switch (itc.y & 3) {\n"
            "  case 0:  color = vec4(texel.r); break;\n"
            "  case 1:  color = vec4(texel.g); break;\n"
            "  case 2:  color = vec4(texel.b); break;\n"
            "  case 3:  color = vec4(texel.a); break;\n"
            "  }\n"
            "}\n";
        break;

    case ShaderTypeFragmentShaderOutput:
    case ShaderTypeFragmentImageStore:
        // The shader output and image store shaders compute a color to write to
        // a texture using the interpolated texture coordinate and then write it
        // out using a shader output or image store.
        s = FragmentShader(450);
        if (shaderType == ShaderTypeFragmentShaderOutput) {
            s << "out vec4 color;\n";
        } else {
            s << "writeonly " << layout.str() << "uniform image2D image;\n";
        }
        s <<
            "in vec2 tc;\n"
            "void main() {\n"
            "  vec4 value;\n"
            "  value.r = (tc.x <  0.5 && tc.y <  0.5) ? 1.0 : 0.0;\n"
            "  value.g = (tc.x >= 0.5 && tc.y <  0.5) ? 1.0 : 0.0;\n"
            "  value.b = (tc.x <  0.5 && tc.y >= 0.5) ? 1.0 : 0.0;\n"
            "  value.a = (tc.x >= 0.5 && tc.y >= 0.5) ? 1.0 : 0.0;\n";
        if (shaderType == ShaderTypeFragmentShaderOutput) {
            s << "  color = value;\n";
        } else {
            s <<
                "  imageStore(image, ivec2(gl_FragCoord.xy), value);\n"
                "  discard;\n";
        }
        s <<  "}\n";
        break;
    default:
        assert(0);
        return NULL;
    }
    if (!g_glslcHelper->CompileAndSetShaders(program, s)) {
        LWNFailTest();
        return NULL;
    }
    return program;
}

void LWNImageSwizzleTest::setCellViewportScissor(CommandBuffer &cb, int cellX, int cellY) const
{
    int ox = cellX * cellSize + cellMargin;
    int oy = cellY * cellSize + cellMargin;
    int vps = cellSize - 2 * cellMargin;
    cb.SetViewportScissor(ox, oy, vps, vps);
}

void LWNImageSwizzleTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    LWNuint id;

    // Set up programs for our basic shaders.
    Program *vspass = mkprogram(device, ShaderTypeVertexPassthrough, NULL);
    Program *fsinterp = mkprogram(device, ShaderTypeFragmentShaderOutput, NULL);
    Program *fsiload = mkprogram(device, ShaderTypeFragmentImageLoad, NULL);
    Program *fsistore = mkprogram(device, ShaderTypeFragmentImageStore, NULL);
    Program *fstexfetch = mkprogram(device, ShaderTypeFragmentTexelFetch, NULL);

    // Set up the basic vertex format and vertex buffer.
    struct Vertex {
        dt::vec3 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3(-1.0, +1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0) },
    };
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // Set a basic sampler for texture lookups.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *sampler = sb.CreateSampler();
    LWNuint samplerID = sampler->GetRegisteredID();

    // Set up our basic vertex shader that we will use for all cells.
    queueCB.BindProgram(vspass, ShaderStageBits::ALL_GRAPHICS_BITS);

    // Clear the window before we start looping over formats.
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    // Set up an allocator to hold one basic texture plus two "swizzled image
    // store" textures for each set of swizzles, with separate sets for each
    // tested format.
    const int nImagesPerFormat = 2 * nTestSwizzles + 1;
    const size_t imagePoolSize = nTestFormats * nImagesPerFormat * testImageMaxBytes;
    MemoryPoolAllocator imageAllocator(device, NULL, imagePoolSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Start rendering test cells for each format we're testing.
    for (int i = 0; i < nTestFormats; i++) {

        Format format = testFormats[i].format;
        const char *layout = testFormats[i].layout;

        // Generate image load and store programs using format-specific layout
        // qualifiers.
        Program *fsiloadWithQual = mkprogram(device, ShaderTypeFragmentImageLoad, layout);
        Program *fsistoreWithQual = mkprogram(device, ShaderTypeFragmentImageStore, layout);

        // Set up an image builder object for this texture format.
        TextureBuilder ib;
        ib.SetDevice(device).SetDefaults();
        ib.SetFlags(TextureFlags::IMAGE);
        ib.SetTarget(TextureTarget::TARGET_2D);
        ib.SetSize2D(testImageWidth, testImageHeight);
        ib.SetFormat(format);
        ib.SetLevels(1);
        if (ib.GetStorageSize() > testImageMaxBytes || ib.GetStorageAlignment() > testImageMaxBytes) {
            LWNFailTest();
            return;
        }

        // Allocate a base texture that we will use to test swizzled loads.
        // Initialize the texture with a fragment shader.
        Texture *baseTexture = imageAllocator.allocTexture(&ib);
        queueCB.SetRenderTargets(1, &baseTexture, NULL, NULL, NULL);
        queueCB.SetViewportScissor(0, 0, testImageWidth, testImageHeight);
        queueCB.BindProgram(fsinterp, ShaderStageBits::FRAGMENT);
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0, ClearColorMask::RGBA);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

        // Generate a set of "swizzled" texture handles to use for testing
        // loads.
        TextureHandle swizzledBaseTexHandle[nTestSwizzles];
        ImageHandle swizzledBaseImageHandle[nTestSwizzles];
        for (int j = 0; j < nTestSwizzles; j++) {
            const SwizzleInfo &swizInfo = testSwizzles[j];
            TextureView swizzleView;
            swizzleView.SetDefaults().SetSwizzle(swizInfo.swizzles[0], swizInfo.swizzles[1],
                                                 swizInfo.swizzles[2], swizInfo.swizzles[3]);
            id = g_lwnTexIDPool->Register(baseTexture, &swizzleView);
            swizzledBaseTexHandle[j] = device->GetTextureHandle(id, samplerID);
            if (m_handleType == ImageHandles) {
                id = g_lwnTexIDPool->RegisterImage(baseTexture, &swizzleView);
                swizzledBaseImageHandle[j] = device->GetImageHandle(id);
            } else {
                swizzledBaseImageHandle[j] = swizzledBaseTexHandle[j];
            }
        }

        // Allocate and initialize a bunch of extra textures using non-default
        // swizzles in the texture view.  We create two textures for each set of
        // swizzles.  One is initialized with imageStores() using a "formatted"
        // surface store where the image is declared with a qualifier like a
        // "layout(rgba32f)".  The other is initialized with imageStore()
        // without any layout qualifier on the image uniform.  Each texture has
        // two sets of handles -- one with swizzles disabled (for loading the
        // stored data) and one with swizzles enabled (for testing the effect on
        // stores).
        Texture *extraTex[2 * nTestSwizzles];
        TextureHandle extraTexHandles[2 * nTestSwizzles];
        for (int j = 0; j < 2 * nTestSwizzles; j++) {
            extraTex[j] = imageAllocator.allocTexture(&ib);
            id = extraTex[j]->GetRegisteredTextureID();
            extraTexHandles[j] = device->GetTextureHandle(id, samplerID);

            // Set up a swizzled texture view as well as an image handle using
            // that view.
            const SwizzleInfo &swizInfo = testSwizzles[j / 2];
            TextureView swizzleView;
            swizzleView.SetDefaults().SetSwizzle(swizInfo.swizzles[0], swizInfo.swizzles[1],
                                                 swizInfo.swizzles[2], swizInfo.swizzles[3]);
            id = g_lwnTexIDPool->RegisterImage(extraTex[j], &swizzleView);
            ImageHandle storeImageHandle = device->GetImageHandle(id);
            id = g_lwnTexIDPool->Register(extraTex[j], &swizzleView);
            TextureHandle storeTexHandle = device->GetTextureHandle(id, samplerID);

            // First, clear the texture to a constant color.
            queueCB.SetRenderTargets(1, &extraTex[j], NULL, NULL, NULL);
            queueCB.ClearColor(0, 0.2, 0.5, 0.8, 1.0, ClearColorMask::RGBA);
            queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

            // Then use an image store shader (with or without the layout
            // qualifier) to write texels into the texture, fully overwriting
            // the clear values.
            queueCB.SetRenderTargets(0, NULL, NULL, NULL, NULL);
            queueCB.SetViewportScissor(0, 0, testImageWidth, testImageHeight);
            queueCB.BindProgram((j & 1) ? fsistoreWithQual : fsistore, ShaderStageBits::FRAGMENT);
            queueCB.BindImage(ShaderStage::FRAGMENT, 0, (m_handleType == ImageHandles) ? storeImageHandle : storeTexHandle);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        }

        // Once we have all our textures of the specified initialized, prepare
        // for on-screen rendering.
        queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);
        g_lwnWindowFramebuffer.bind();

        // Render a row on the bottom half testing swizzled texel fetches and
        // image loads.
        int cx = 0;
        int cy = i;
        showTexture(queueCB, cx++, cy, baseTexture);
        for (int j = 0; j < nTestSwizzles; j++) {
            cx++;

            setCellViewportScissor(queueCB, cx++, cy);
            queueCB.BindProgram(fstexfetch, ShaderStageBits::FRAGMENT);
            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, swizzledBaseTexHandle[j]);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

            setCellViewportScissor(queueCB, cx++, cy);
            queueCB.BindProgram(fsiload, ShaderStageBits::FRAGMENT);
            queueCB.BindImage(ShaderStage::FRAGMENT, 0, swizzledBaseImageHandle[j]);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

            setCellViewportScissor(queueCB, cx++, cy);
            queueCB.BindProgram(fsiloadWithQual, ShaderStageBits::FRAGMENT);
            queueCB.BindImage(ShaderStage::FRAGMENT, 0, swizzledBaseImageHandle[j]);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        }

        // Render a row on the top half displaying the results of our swizzled
        // image loads.
        cx = 0;
        cy = i + nTestFormats + 1;
        for (int j = 0; j < 2 * nTestSwizzles; j++) {
            showTexture(queueCB, cx++, cy, extraTex[j]);
            setCellViewportScissor(queueCB, cx++, cy);
            queueCB.BindProgram(fstexfetch, ShaderStageBits::FRAGMENT);
            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, extraTexHandles[j]);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
            if (j & 1) {
                cx++;   // add spacing between each tested swizzle group
            }
        }
    }

    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNImageSwizzleTest, lwn_image_swizzle, (LWNImageSwizzleTest::ImageHandles));
OGTEST_CppTest(LWNImageSwizzleTest, lwn_image_swizzle_th, (LWNImageSwizzleTest::TextureHandles));
