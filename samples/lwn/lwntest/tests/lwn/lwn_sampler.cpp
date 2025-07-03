/*
 * Copyright (c) 2015-2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#ifndef CEIL
#define CEIL(a,b)        (((a)+(b)-1)/(b))
#endif

#ifndef ROUND_UP
#define ROUND_UP(N, S) (CEIL((N),(S)) * (S))
#endif

using namespace lwn;

class LWNSamplerTest
{
public:
    enum TestVariant {
        VariantWrapIdentical,   // Test all wrap modes with same modes on S/T/R.
        VariantWrapMix,         // Test all wrap modes with different modes on S/T/R.
        VariantWrapNearest,     // Test all wrap modes with same modes on S/T/R and nearest filtering.
        VariantLodBias,         // Test different LOD bias values.
        VariantLodClamp,        // Test different LOD clamping ranges.
        VariantQueryLod,        // Test querying automatic LOD with textureQueryLod
        VariantMinMagFilter,    // Test different min/mag filter combinations.
        VariantMinMaxReduction, // Test MIN/MAX reductions.
    };

private:
    // We set up 32x32 textures, where "checkerboard" versions have 4x4 grids
    // of 8x8-pixel checkers.  We truncate to 4 mipmap levels so all levels
    // can show a proper checker pattern (where the checkers are 8x8, 4x4,
    // 2x2, and 1x1).
    static const int texSize = 32;
    static const int log2CheckerSize = 3;
    static const int texCheckerSize = (1 << log2CheckerSize);
    static const int texLevels = 4;

    // We render 48x48 quads, where many cells have texture coordinates in
    // [-0.25,+1.25].  The center of the image will be 1:1 pixels/texels; the
    // edges show various border conditions.  Store these in 60x60 cells.
    static const int cellSize = 48;
    static const int cellSpacing = 60;
    static const int cellPadding = (cellSpacing - cellSize) / 2;

    // Sizes of memory pools used for staging/storing texture data.
    static const int texDataSize = 4 * 1024 * 1024;
    static const int texStorageSize = 4 * 1024 * 1024;

    // The test iterates over texture targets in columns.
    enum TextureType {
        Texture1D,
        Texture1DArray,
        TextureRect,
        Texture2D,
        Texture2DArray,
        TextureLwbe,
        TextureLwbeArray,
        Texture3D,              // Slice of a 3D texture with constant Z.
        Texture3DXZ,            // Slice of the same 3D texture with constant Y.
        TextureTypeCount,
    };

    Program *mkprogram(Device *device, TextureType textype) const;
    Texture *mktexture(TextureType textype,
                       TextureBuilder &tb, MemoryPoolAllocator &allocator,
                       QueueCommandBuffer &queueCB, BufferAddress texDataBufferAddr,
                       uint8_t **texDataPtr, LWNuint *texDataOffset) const;

    TestVariant m_variant;
    bool m_useSamplerObject;
public:
    LWNSamplerTest(TestVariant variant, bool useSamplerObject = true) :
        m_variant(variant), m_useSamplerObject(useSamplerObject) {}
    LWNTEST_CppMethods();
};

lwString LWNSamplerTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Functional test for various LWN sampler states.  This test exercises "
        "different sampler states with all texture targets.  Each column is a "
        "different target.  From left-to-right, we have 1D, 1D array, "
        "rectangle, 2D, lwbe map, lwbe map array, and two 3D columns.  The "
        "first 3D column slices with a constant R texture coordinate; the "
        "second slices with a constant T coordinate.  The array texture columns "
        "sample from a layer set up differently from the equivalent non-array "
        "texture; patterns are reversed and gradients are different in these columns.";
    sb << "\n\n";

    switch (m_variant) {
    case VariantWrapIdentical:
    case VariantWrapMix:
    case VariantWrapNearest:
        sb <<
            "This test exercises different texture wrap modes, where the S wrap "
            "mode is, from bottom to top:\n"
            " * CLAMP\n"
            " * REPEAT\n"
            " * MIRROR_CLAMP\n"
            " * MIRROR_CLAMP_TO_EDGE\n"
            " * MIRROR_CLAMP_TO_BORDER\n"
            " * CLAMP_TO_BORDER\n"
            " * MIRRORED_REPEAT\n"
            " * CLAMP_TO_EDGE\n"
            "\n";
        if (m_variant == VariantWrapIdentical) {
            sb <<
                "The T and R wrap modes are identical.  The border color "
                "(where applicable) is red.";
        } else if (m_variant == VariantWrapNearest) {
            sb <<
                "The T and R wrap modes are identical.  The border color "
                "(where applicable) is blue.  NEAREST filtering is used "
                "which means that the border is not accessed in CLAMP modes.";
        } else {
            sb <<
                "The T wrap mode in each row is the mode after the S wrap mode "
                "(wrapping around) and the R wrap mode is two modes after the S "
                "wrap mode.  The border color (where applicable) is green.";
        }
        sb <<
            "  The textures show a gradient checkerboard pattern with a thin "
            "grey margin at the edges.  For the rectangle texture column, REPEAT "
            "and the MIRROR* modes are unsupported and mapped to CLAMP.  For lwbe "
            "maps and lwbe map arrays, wrap modes are ignored and cross-face "
            "filtering is performed instead.";
        break;
    case VariantLodBias:
        sb <<
            "This test exercises different sampler LOD bias values from -0.5 "
            "(bottom) to +3.0 (top).  The checkers are colored by mipmap level "
            "and are red, green, blue, and cyan.  The rectangle texture "
            "column is all red because only LOD 0 is supported there.  The lwbe "
            "map columns are displaying the +Z face in the middle, and neighboring "
            "faces on the edges.  The LODs near the seams and on adjacent faces "
            "will be non-constant due to lwbe map projection math.";
        break;
    case VariantLodClamp:
        sb <<
            "This test exercises different sampler LOD clamp values.  LOD varies "
            "in each cell from 0.0 (bottom) to +3.0 (top).  The checkers are colored by "
            "mipmap level and are red, green, blue, and cyan.  Clamp values start at [0,3] "
            "(bottom), and then move inward in increments of 0.2, where the top row "
            "has clamps of [1.4, 1.6].  The rectangle texture column is all red "
            "because only LOD 0 is supported there.";
        break;
     case VariantQueryLod:
        sb <<
            "This test renders into the LOD obtained by querying textureQueryLod. "
            "The test will render each cell progressively smaller in screen size "
            "towards the top of the window.  As the cells get smaller, the "
            "value returned by textureQueryLod will change.  The colors for each mipmap "
            "level should be displayed based on the results of using this LOD with "
            "the textureLod function.";
        break;
    case VariantMinMagFilter:
        sb <<
            "This test exercises different combinations of min/mag filters. "
            "From bottom to top:\n"
            " * (min) NEAREST / (mag) LINEAR\n"
            " * (min) LINEAR / (mag) NEAREST\n"
            " * (min) NEAREST_MIPMAP_NEAREST / (mag) NEAREST\n"
            " * (min) NEAREST_MIPMAP_LINEAR / (mag) NEAREST\n"
            " * (min) LINEAR_MIPMAP_NEAREST / (mag) LINEAR\n"
            " * (min) LINEAR_MIPMAP_LINEAR / (mag) LINEAR\n"
            " * (min) LINEAR_MIPMAP_LINEAR / (mag) NEAREST\n"
            " * (min) NEAREST_MIPMAP_NEAREST / (mag) LINEAR\n"
            "LOD varies from -3 (bottom) to +3 in each cell; the bottom half "
            "should use the mag filter and the top half should use the min "
            "filter and possibly different (rainbow) mipmap levels.  Checkers "
            "are red, green, blue, and cyan for the four LODs.  This test "
            "zooms in on the center of the texture and magnifies so you can "
            "clearly tell the difference between linear and nearest filtering.";
        break;
    case VariantMinMaxReduction:
        sb <<
            "This test exercises different combinations of MIN/MAX/AVERAGE "
            "reduction modes.  Each pair of two rows from bottom to top:\n"
            " * MIN reduction\n"
            " * MAX reduction\n"
            " * AVERAGE reduction\n"
            " * default reduction state (AVERAGE)\n"
            "The bottom mode in the pair tests point sampling; the top row "
            "tests linear filtering.  All the point sampling images should "
            "be the same.  The linear filtering rows should have smaller "
            "checkers for MIN, larger overlapped checkers for MAX, and a "
            "gradient between checkers for AVERAGE.  This image is highly "
            "zoomed in on the corner between colored and black checkers.";
        break;
    default:
        assert(0);
    }

    return sb.str();
}

int LWNSamplerTest::isSupported() const
{
    if (!m_useSamplerObject && !lwogCheckLWNAPIVersion(53, 0)) {
        return 0;
    }
    if (m_variant == VariantMinMaxReduction) {
        return lwogCheckLWNAPIVersion(26, 3) && g_lwnDeviceCaps.supportsMinMaxReduction;
    }
    return lwogCheckLWNAPIVersion(26, 0);
}

Program *LWNSamplerTest::mkprogram(Device *device, TextureType textype) const
{
    VertexShader vs(440);
    vs <<
        "layout(binding=0) uniform Block {\n"
        "  int rowNum;\n"
        "};\n"
        "layout(location=0) in vec2 position;\n"
        "layout(location=1) in vec3 texcoord;\n"
        "out vec3 otc;\n"
        "void main() {\n"
        "  vec2 iPos = position;\n";
    if (m_variant == VariantQueryLod) {
        // Modify the screen size of the cell based on the exponential function.
        // The exponential function and its parameter are chosen here to give us
        // a non-linear smooth decrease in cell size as we go up in rows.
        vs << "  if (rowNum > 0) {\n"
              "    iPos /= (1.4f*exp(rowNum/3.0f));\n"
              "  }\n";
    }
    vs <<
        "  gl_Position = vec4(iPos, 0.0, 1.0);\n"
        "  otc = texcoord;\n"
        "}\n";

    FragmentShader fs(440);
    switch (textype) {
    case Texture1D:         fs << "uniform sampler1D sampler;\n"; break;
    case Texture1DArray:    fs << "uniform sampler1DArray sampler;\n"; break;
    case Texture2D:         fs << "uniform sampler2D sampler;\n"; break;
    case Texture2DArray:    fs << "uniform sampler2DArray sampler;\n"; break;
    case TextureRect:       fs << "uniform sampler2DRect sampler;\n"; break;
    case TextureLwbe:       fs << "uniform samplerLwbe sampler;\n"; break;
    case TextureLwbeArray:  fs << "uniform samplerLwbeArray sampler;\n"; break;
    case Texture3D:         fs << "uniform sampler3D sampler;\n"; break;
    case Texture3DXZ:       fs << "uniform sampler3D sampler;\n"; break;
    default:                assert(0); break;
    }
    fs <<
        "in vec3 otc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  vec3 tc = otc;\n";

    bool useTextureLod = false;
    bool useTextureQueryLod = false;
    switch (m_variant) {
    case VariantWrapIdentical:
    case VariantWrapMix:
    case VariantWrapNearest:
    case VariantLodBias:
        // For wrap and LOD bias, remap [0,1] coordinate range to [-0.25,+1.25].
        fs << "  tc.xy = 1.5 * tc.xy - 0.25;\n";
        break;
    case VariantLodClamp:
        // For LOD clamp testing, remap the [0,1] T coordinate range into an LOD
        // gradient [0,texLevels-1].
        fs << "  float lod = tc.y * " << (texLevels - 1) << ".0;\n";
        useTextureLod = (textype != TextureRect);
        break;
    case VariantQueryLod:
        useTextureQueryLod = (textype != TextureRect);

        if (useTextureQueryLod) {
            // For textureQueryLod tests, we still render with textureLod, but
            // the LOD is callwlated based on the automatic LOD value returned
            // by textureQueryLod.  For lwbe textures, we just use the first
            // coordinate with other two fixed as 1.
            fs << "  float lod = textureQueryLod(sampler, ";

            switch (textype) {
            case Texture1D:
            case Texture1DArray:
                fs << "tc.x";
                break;
            case Texture2D:
            case Texture2DArray:
                fs << "tc.xy";
                break;
            case TextureLwbe:
            case TextureLwbeArray:
                fs << "vec3(tc.x, 1.0, 1.0)";
                break;
            case Texture3D:
            case Texture3DXZ:
                fs << "tc.xyz";
                break;
            default:                assert(0); break;
            }

            // Take .y value which holds the automatic LOD value.
            fs << ").y;\n";
        }

        break;
    case VariantMinMagFilter:
        // For min/mag filter testing, remap the [0,1] T coordinate range into an LOD
        // gradient [-texLevels+1, texLevels-1].  Remap the (s,t) coordinate
        // range to zoom in on the center of the image to make nearest/linear
        // filtering obvious.
        fs << "  float lod = 2.0 * (tc.y - 0.5) * " << (texLevels - 1) << ".0;\n";
        fs << "  tc.xy = 0.5 + (tc.xy - 0.5) * 4.0 / " << texSize << ".0;\n";
        useTextureLod = (textype != TextureRect);
        break;
    case VariantMinMaxReduction:
        // Remap the (s,t) coordinate range to zoom in on the center of the
        // image to make min/mag choices obvious.
        fs << "  tc.xy = 0.5 + (tc.xy - 0.5) * 4.0 / " << texSize << ".0;\n";
        break;
    default:
        break;
    }

    if (useTextureLod || useTextureQueryLod) {
        fs << "  fcolor = textureLod(sampler, ";
    } else {
        fs << "  fcolor = texture(sampler, ";
    }
    switch (textype) {
    case Texture1D:         fs << "tc.x"; break;
    case Texture1DArray:    fs << "vec2(tc.x, " << texSize-0.5 << ")"; break;
    case Texture2D:         fs << "tc.xy"; break;
    case Texture2DArray:    fs << "vec3(tc.xy, " << texSize-0.5 << ")"; break;
    case TextureRect:       fs << "tc.xy * " << texSize << ".0"; break;
    case TextureLwbe:       fs << "vec3(2.0 * tc.xy - 1.0, +1.0)"; break;
    case TextureLwbeArray:  fs << "vec4(2.0 * tc.xy - 1.0, +1.0, " << texSize-0.5 << ")"; break;
    case Texture3D:         fs << "tc"; break;
    case Texture3DXZ:       fs << "tc.xzy"; break;
    default:                assert(0); break;
    }
    if (useTextureLod || useTextureQueryLod) {
        fs << ", lod";
    }
    fs << ");\n";
    fs << "}\n";

    Program *program = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        program->Free();
        program = NULL;
    }
    return program;
}

Texture *LWNSamplerTest::mktexture(TextureType textype,
                                   TextureBuilder &tb, MemoryPoolAllocator &allocator,
                                   QueueCommandBuffer &queueCB, BufferAddress texDataBufferAddress,
                                   uint8_t **texDataPtr, LWNuint *texDataOffset) const
{
    // Do we show a rainbow coloring for the mipmap levels to show the selected LOD?
    bool rainbow = (m_variant == VariantLodBias || m_variant == VariantLodClamp || m_variant == VariantQueryLod ||
                    m_variant == VariantMinMagFilter);

    // Do we draw a thin edge around the texture (to distinguish between CLAMP_TO_EDGE
    // and mirrored operations)?
    bool addEdge = (m_variant == VariantWrapIdentical || m_variant == VariantWrapMix ||
                    m_variant == VariantWrapNearest);

    // Set up information on the texture.
    int numLevels = texLevels;
    int w = texSize;
    int h = texSize;
    int d = texSize;
    int dim = 1;        // dimension of individual images (excludes layers/faces)
    TextureTarget target = TextureTarget::TARGET_2D;
    switch (textype) {
    case Texture1D:
        target = TextureTarget::TARGET_1D;
        dim = 1;
        h = 1;
        d = 1;
        break;
    case Texture1DArray:
        target = TextureTarget::TARGET_1D_ARRAY;
        dim = 1;
        d = 1;
        break;
    case Texture2D:
        target = TextureTarget::TARGET_2D;
        dim = 2;
        d = 1;
        break;
    case Texture2DArray:
        target = TextureTarget::TARGET_2D_ARRAY;
        dim = 2;
        break;
    case TextureRect:
        target = TextureTarget::TARGET_RECTANGLE;
        dim = 2;
        d = 1;
        numLevels = 1;
        break;
    case TextureLwbe:
        target = TextureTarget::TARGET_LWBEMAP;
        dim = 2;
        d = 6;
        break;
    case TextureLwbeArray:
        target = TextureTarget::TARGET_LWBEMAP_ARRAY;
        dim = 2;
        d = 6;
        break;
    case Texture3D:
        target = TextureTarget::TARGET_3D;
        dim = 3;
        break;
    case Texture3DXZ:
    default:
        assert(0);
        break;
    }
    tb.SetDefaults();
    tb.SetTarget(target);
    if (textype == TextureLwbe) {
        tb.SetSize3D(w, h, 6);
    } else {
        tb.SetSize3D(w, h, d);
    }
    tb.SetFormat(Format::RGBA8);
    tb.SetLevels(numLevels);
    Texture *tex = allocator.allocTexture(&tb);

    // Table of colors for rainbow mipmaps.
    static const uint8_t rainbowColors[][4] = {
        { 0xFF, 0x00, 0x00, 0xFF },
        { 0x00, 0xFF, 0x00, 0xFF },
        { 0x00, 0x00, 0xFF, 0xFF },
        { 0x00, 0xFF, 0xFF, 0xFF },
        { 0xFF, 0x00, 0xFF, 0xFF },
        { 0xFF, 0xFF, 0x00, 0xFF },
    };

    // Generate data for each of the levels.
    uint8_t *texData = *texDataPtr;
    TextureView levelView;
    levelView.SetDefaults();
    for (int level = 0; level < numLevels; level++) {
        levelView.SetLevels(level, 1);
        for (int z = 0; z < d; z++) {
            uint8_t b = uint8_t(255.0 * (0.2 + 0.6 * ((z + 0.5) / d)));
            int zchecker = z >> (log2CheckerSize - level);
            int zedge = (dim >= 3) && (z == 0 || z == d-1);
            // For lwbe maps/arrays, all six faces in a lwbe should have
            // the same pattern.
            if (textype == TextureLwbe || textype == TextureLwbeArray) {
                zchecker = z / 6;
            }
            if (rainbow) {
                b = rainbowColors[level][2];
            }
            for (int y = 0; y < h; y++) {
                uint8_t g = uint8_t(255.0 * (0.2 + 0.6 * ((y + 0.5) / h)));
                int ychecker = y >> (log2CheckerSize - level);
                int yedge = (dim >= 2) && (y == 0 || y == h-1);
                if (rainbow) {
                    g = rainbowColors[level][1];
                }
                for (int x = 0; x < w; x++) {
                    uint8_t r = uint8_t(255.0 * (0.2 + 0.6 * ((x + 0.5) / w)));
                    int xchecker = x >> (log2CheckerSize - level);
                    int xedge = (x == 0 || x == w-1);
                    if (rainbow) {
                        r = rainbowColors[level][0];
                    }

                    if (addEdge && (xedge || yedge || zedge)) {
                        // Solid grey on edges if <addEdge> is set.
                        *texData++ = 0x80;
                        *texData++ = 0x80;
                        *texData++ = 0x80;
                        *texData++ = 0xFF;
                    } else if ((xchecker ^ ychecker ^ zchecker) & 1) {
                        *texData++ = 0x00;
                        *texData++ = 0x00;
                        *texData++ = 0x00;
                        *texData++ = 0x00;
                    } else {
                        *texData++ = r;
                        *texData++ = g;
                        *texData++ = b;
                        *texData++ = 0xFF;
                    }
                }
            }
        }

        // Copy our texture data into the image/mipmap and prepare for the next one.
        CopyRegion copyRegion = { 0, 0, 0, w, h, d };
        queueCB.CopyBufferToTexture(texDataBufferAddress + *texDataOffset, tex, &levelView, &copyRegion, CopyFlags::NONE);
        *texDataOffset += (texData - *texDataPtr);
        *texDataPtr = texData;
        w /= 2;
        if (dim >= 2) h /= 2;
        if (dim >= 3) d /= 2;
    }

    return tex;
}


void LWNSamplerTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Generate programs for each texture type.
    Program *programs[TextureTypeCount];
    for (int i = 0; i < TextureTypeCount; i++) {
        programs[i] = mkprogram(device, TextureType(i));
        if (!programs[i]) {
            LWNFailTest();
            return;
        }
    }

    // Set up a pool/buffer for staging texel data.
    MemoryPoolAllocator texDataAllocator(device, NULL, texDataSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *texDataBuffer = texDataAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, texDataSize);
    uint8_t *texDataCPUAddress = (uint8_t *) texDataBuffer->Map();
    LWNuint texDataOffset = 0;

    // Set up a texture object for each texture type.
    MemoryPoolAllocator texAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    Texture *textures[TextureTypeCount];
    for (int i = 0; i < TextureTypeCount; i++) {
        if (i == Texture3DXZ) {
            textures[i] = textures[Texture3D];
            continue;
        }
        textures[i] = mktexture(TextureType(i), tb, texAllocator, queueCB,
                                texDataBuffer->GetAddress(), &texDataCPUAddress, &texDataOffset);
    }

    // Set up ubo to contain the row number (used only in textureQueryLod tests, but
    // no harm in having it around for other tests even though it won't be referenced
    // in the shader).
    struct UniformBlock {
        int rowNum;
    };

    UniformBlock dummyBlock;

    LWNuint uboSize = sizeof(UniformBlock);
    LWNint uboStorageSize = ROUND_UP(uboSize, 1024);
    MemoryPool* uboMemPool = device->CreateMemoryPool(NULL, uboStorageSize, MemoryPoolType::CPU_COHERENT);
    LWNbufferAddress uboAddr = uboMemPool->GetBufferAddress();

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec2 position;
        dt::vec3 texcoord;
    };
    float texCheckerCenter = (0.5 * texCheckerSize) / texSize;
    Vertex vertexData[] = {
        { dt::vec2(-1.0, -1.0), dt::vec3(0.0, 0.0, texCheckerCenter) },
        { dt::vec2(-1.0, +1.0), dt::vec3(0.0, 1.0, texCheckerCenter) },
        { dt::vec2(+1.0, -1.0), dt::vec3(1.0, 0.0, texCheckerCenter) },
        { dt::vec2(+1.0, +1.0), dt::vec3(1.0, 1.0, texCheckerCenter) },
    };
    MemoryPoolAllocator vboAllocator(device, NULL, 4 * sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Tables of wrap modes to test.
    static const WrapMode wrapModes[] = {
        WrapMode::CLAMP,
        WrapMode::REPEAT,
        WrapMode::MIRROR_CLAMP,
        WrapMode::MIRROR_CLAMP_TO_EDGE,
        WrapMode::MIRROR_CLAMP_TO_BORDER,
        WrapMode::CLAMP_TO_BORDER,
        WrapMode::MIRRORED_REPEAT,
        WrapMode::CLAMP_TO_EDGE,
    };

    // Table of min/mag filter combinations to test.
    static const struct {
        MinFilter minFilter;
        MagFilter magFilter;
    } minMagModes[] = {
        { MinFilter::NEAREST, MagFilter::LINEAR },
        { MinFilter::LINEAR, MagFilter::NEAREST},
        { MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST },
        { MinFilter::NEAREST_MIPMAP_LINEAR, MagFilter::NEAREST },
        { MinFilter::LINEAR_MIPMAP_NEAREST, MagFilter::LINEAR },
        { MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR },
        { MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::NEAREST },
        { MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::LINEAR },
    };

    // Set up samplers for each row (shared by all texture types).
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *samplers[8];
    uint32_t samplerIDs[8];
    LWNfloat borders[][4] = { { 0, 0, 0, 0 }, { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 } };
    for (int i = 0; i < 8; i++) {
        sb.SetDefaults();
        sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
        sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
        sb.SetBorderColor(borders[0]);
        switch (m_variant) {
        case VariantWrapIdentical:
            sb.SetBorderColor(borders[1]);
            sb.SetWrapMode(wrapModes[i], wrapModes[i], wrapModes[i]);
            break;
        case VariantWrapMix:
            sb.SetBorderColor(borders[2]);
            sb.SetWrapMode(wrapModes[i], wrapModes[(i + 1) % 8], wrapModes[(i + 2) % 8]);
            break;
        case VariantWrapNearest:
            sb.SetBorderColor(borders[3]);
            sb.SetWrapMode(wrapModes[i], wrapModes[i], wrapModes[i]);
            sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
            break;
        case VariantLodBias:
            sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
            sb.SetLodBias(0.5 * (i - 1));
            break;
        case VariantLodClamp:
            sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
            sb.SetLodClamp(0.2 * i, texLevels - 1.0 - 0.2 * i);
            break;
        case VariantQueryLod:
            sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
            break;
        case VariantMinMagFilter:
            sb.SetMinMagFilter(minMagModes[i].minFilter, minMagModes[i].magFilter);
            break;
        case VariantMinMaxReduction:
            if (i & 1) {
                sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
            } else {
                sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
            }
            switch (i/2) {
            case 0:  sb.SetReductionFilter(SamplerReduction::MIN); break;
            case 1:  sb.SetReductionFilter(SamplerReduction::MAX); break;
            case 2:  sb.SetReductionFilter(SamplerReduction::AVERAGE); break;
            case 3:  break;  // default state is AVERAGE
            default: assert(0); break;
            }
            break;
        default:
            assert(0);
            break;
        }
        if (m_useSamplerObject) {
            samplers[i] = sb.CreateSampler();
            samplerIDs[i] = samplers[i]->GetRegisteredID();
        } else {
            samplers[i] = NULL;
            samplerIDs[i] = g_lwnTexIDPool->Register(&sb);
        }
    }

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
    for (int i = 0; i < TextureTypeCount; i++) {
        queueCB.BindProgram(programs[i], ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, uboStorageSize);

        for (int j = 0; j < 8; j++) {
            TextureHandle texHandle =
                device->GetTextureHandle(textures[i]->GetRegisteredTextureID(), samplerIDs[j]);
            queueCB.SetViewport(i * cellSpacing + cellPadding, j * cellSpacing + cellPadding, cellSize, cellSize);

            dummyBlock.rowNum = j;
            queueCB.UpdateUniformBuffer(uboAddr, uboStorageSize, 0, sizeof(UniformBlock), &dummyBlock);

            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        }
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    // If we were registering sampler pool entries directly from sampler
    // builders, clean up sampler pool entries allocated since there is no
    // hook for freeing the entries along with sampler objects.
    if (!m_useSamplerObject) {
        for (int i = 0; i < 8; i++) {
            g_lwnTexIDPool->FreeSamplerID(samplerIDs[i]);
        }
    }

    uboMemPool->Free();
}

OGTEST_CppTest(LWNSamplerTest, lwn_sampler_filter, (LWNSamplerTest::VariantMinMagFilter));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_lodbias, (LWNSamplerTest::VariantLodBias));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_lodclamp, (LWNSamplerTest::VariantLodClamp));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_minmax, (LWNSamplerTest::VariantMinMaxReduction));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_querylod, (LWNSamplerTest::VariantQueryLod));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_wrap, (LWNSamplerTest::VariantWrapIdentical));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_wrapmix, (LWNSamplerTest::VariantWrapMix));
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_wrapnearest, (LWNSamplerTest::VariantWrapNearest));

// Special-case test to exercise lwnSamplerPoolRegisterSamplerBuilder.
OGTEST_CppTest(LWNSamplerTest, lwn_sampler_wrap_noobj, (LWNSamplerTest::VariantWrapIdentical, false));

class LWNSamplerBorderTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNSamplerBorderTest::getDescription() const
{
    return  "Test checks border color functionality of the sampler.\n"
            "For each of the float, int and uint types, test creates a\n"
            "texture, sampler and a program. Output should be three white\n"
            "rectangles with red, green and blue borders.";
}

int LWNSamplerBorderTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

void LWNSamplerBorderTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int texDim = 4;

    struct {
        Texture* tex;
        TextureHandle texHandle;
        Buffer* buffer;
        void* ptr;
        Sampler* smp;
        Program* pgm;
    } data[3];

    // create and compile programs
    data[0].pgm = device->CreateProgram();
    data[1].pgm = device->CreateProgram();
    data[2].pgm = device->CreateProgram();

    VertexShader vs(440);
    vs << "out vec2 uv;\n"
          "void main() {\n"

          "  vec2 position; "
          "  if (gl_VertexID == 0) position = vec2(-1.0, -1.0);\n"
          "  if (gl_VertexID == 1) position = vec2(1.0, -1.0);\n"
          "  if (gl_VertexID == 2) position = vec2(1.0, 1.0);\n"
          "  if (gl_VertexID == 3) position = vec2(-1.0, 1.0);\n"
          "  gl_Position = vec4(position, 0.0, 1.0);\n"
          "  uv = (position*0.5 + vec2(0.5,0.5));\n"
          "  uv = 3.0 * uv - 1.0;\n"
          "}\n";

    {
        FragmentShader fs(440);
        fs <<
            "in vec2 uv;\n"
            "layout (binding=0) uniform sampler2D tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = texture(tex,uv);\n"
            "}\n";
        g_glslcHelper->CompileAndSetShaders(data[0].pgm, vs, fs);
    }
    {
        FragmentShader fs(440);
        fs <<
            "in vec2 uv;\n"
            "layout (binding=0) uniform isampler2D tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  ivec4 col = texture(tex,uv)/0x7FFFFFFF;\n"
            "  fcolor = vec4(col);\n"
            "}\n";
        g_glslcHelper->CompileAndSetShaders(data[1].pgm, vs, fs);
    }
    {
        FragmentShader fs(440);
        fs <<
            "in vec2 uv;\n"
            "layout (binding=0) uniform usampler2D tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  uvec4 col = texture(tex,uv)/0xFFFFFFFF;\n"
            "  fcolor = vec4(col);\n"
            "}\n";
        g_glslcHelper->CompileAndSetShaders(data[2].pgm, vs, fs);
    }

    // create samplers
    SamplerBuilder sb;
    sb.SetDevice(device);
    sb.SetDefaults();
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    sb.SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER);
    float borderFloat[4] = {1, 0, 0, 1};
    sb.SetBorderColor(borderFloat);
    data[0].smp = sb.CreateSampler();
    int borderInt[4] = {0, 0x7FFFFFFF, 0, 0x7FFFFFFF};
    sb.SetBorderColori(borderInt);
    data[1].smp = sb.CreateSampler();
    uint32_t borderUint[4] = {0, 0, 0xFFFFFFFF, 0xFFFFFFFF};
    sb.SetBorderColorui(borderUint);
    data[2].smp = sb.CreateSampler();

    // create textures
    TextureBuilder tb;
    tb.SetDevice(device);
    tb.SetDefaults();
    tb.SetSize2D(texDim, texDim);
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetFormat(Format::RGBA32F);
    size_t texSize = tb.GetStorageSize();
    MemoryPool *cpuBufferMemPool = NULL;
    MemoryPool* gpuTexMemPool = NULL;
    gpuTexMemPool = device->CreateMemoryPool(NULL, texSize * 3, MemoryPoolType::GPU_ONLY);
    tb.SetFormat(Format::RGBA32F);
    data[0].tex = tb.CreateTextureFromPool(gpuTexMemPool, 0);
    tb.SetFormat(Format::RGBA32I);
    data[1].tex = tb.CreateTextureFromPool(gpuTexMemPool, texSize);
    tb.SetFormat(Format::RGBA32UI);
    data[2].tex = tb.CreateTextureFromPool(gpuTexMemPool, texSize * 2);

    // fill textures
    BufferBuilder bb;
    bb.SetDevice(device);
    bb.SetDefaults();
    cpuBufferMemPool = device->CreateMemoryPool(NULL, texSize * 3, MemoryPoolType::CPU_COHERENT);
    for (int i=0; i<3; i++) {
        data[i].buffer = bb.CreateBufferFromPool(cpuBufferMemPool, texSize * i, texSize);
        data[i].ptr = data[i].buffer->Map();
    }

    float* ptrFloat = static_cast<float*>(data[0].ptr);
    int* ptrInt = static_cast<int*>(data[1].ptr);
    uint32_t* ptrUint = static_cast<uint32_t*>(data[2].ptr);
    for (int i=0; i<texDim*texDim; i++) {
        ptrFloat[i*4] = 1.0;
        ptrFloat[i*4+1] = 1.0;
        ptrFloat[i*4+2] = 1.0;
        ptrFloat[i*4+3] = 1.0;

        ptrInt[i*4] = 0x7FFFFFFF;
        ptrInt[i*4+1] = 0x7FFFFFFF;
        ptrInt[i*4+2] = 0x7FFFFFFF;
        ptrInt[i*4+3] = 0x7FFFFFFF;

        ptrUint[i*4] = 0xFFFFFFFF;
        ptrUint[i*4+1] = 0xFFFFFFFF;
        ptrUint[i*4+2] = 0xFFFFFFFF;
        ptrUint[i*4+3] = 0xFFFFFFFF;
    }

    CopyRegion cr = { 0, 0, 0, texDim, texDim, 1 };
    for (int i=0; i<3; i++) {
        queueCB.CopyBufferToTexture(data[i].buffer->GetAddress(), data[i].tex, 0, &cr, CopyFlags::NONE);
        queueCB.submit();
        queue->Finish();
    }

    // get handles
    for (int i=0; i<3; i++) {
        LWNuint smpID = g_lwnTexIDPool->Register(data[i].smp);
        LWNuint texID = g_lwnTexIDPool->Register(data[i].tex);
        data[i].texHandle = device->GetTextureHandle(texID, smpID);
    }

    // draw
    int cellW = lwrrentWindowWidth / 3;
    int cellH = lwrrentWindowHeight;

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);
    for (int i=0; i<3; i++) {
        queueCB.SetViewportScissor(cellW * i, 0, cellW, cellH);
        queueCB.ClearColor(0, 0.4, 0.0, 0.0, 1.0);

        queueCB.BindProgram(data[i].pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, data[i].texHandle);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);

        queueCB.submit();
        queue->Finish();
    }

    //cleanup
    for (int i=0; i<3; i++) {
        data[i].buffer->Free();
        data[i].tex->Free();
        data[i].smp->Free();
        data[i].pgm->Free();
    }
    cpuBufferMemPool->Free();
    gpuTexMemPool->Free();
}

OGTEST_CppTest(LWNSamplerBorderTest, lwn_sampler_border, );

class LWNSamplerAnisotropyTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWNSamplerAnisotropyTest::getDescription() const
{
    return  "Test iterates over different combinations of anisotropic level\n"
            "and angle of geometry. Test should be verified manually, by\n"
            "making sure that rows sample texture differently. Expected \n"
            "output are progressively more better sampled quads. Quads are\n"
            "distorted by perspection, and Moire patterns will be visible.";
}

int LWNSamplerAnisotropyTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 19);
}

void LWNSamplerAnisotropyTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // test configurations
    float zAngles[] = {0.0, 20.0, 45.0, 70.0, 90.0};
    float anisoLevels[] = {1.0, 4.0, 8.0, 16.0};
    const int numZAngles = __GL_ARRAYSIZE(zAngles);
    const int numALevels = __GL_ARRAYSIZE(anisoLevels);

    // setup
    const float camFOV = 90.0;
    const float camNear = 0.1;
    const float camFar = 10.0;
    const int texDim = 64;
    float S = 1.0 / tan(camFOV * 0.00872664625);    // PI/360
    float modelMat[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float projMat[16] = {
        S, 0, 0, 0,
        0, S, 0, 0,
        0, 0, camFar/(camNear-camFar), -1,
        0, 0, camFar*camNear/(camNear-camFar), 0
    };
    int cellW = lwrrentWindowWidth / numZAngles;
    int cellH = lwrrentWindowHeight / numALevels;

    // program
    Program* pgm;
    VertexShader vs(440);
    FragmentShader fs(440);
    vs <<
        "layout(binding=0, std140) uniform matrices {\n"
        "  mat4 projMat;\n"
        "  mat4 modelMat;\n"
        "};\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "  vec3 pos = vec3(0.0);\n"
        "  if (gl_VertexID == 0) pos = vec3(-1.0, -1.0, -10.0);\n"
        "  if (gl_VertexID == 1) pos = vec3( 1.0, -1.0, -1.0);\n"
        "  if (gl_VertexID == 2) pos = vec3( 1.0,  1.0, -1.0);\n"
        "  if (gl_VertexID == 3) pos = vec3(-1.0,  1.0, -10.0);\n"
        "  gl_Position = projMat * modelMat * vec4(pos, 1.0);\n"
        "  uv = pos.xy * 0.5 + 0.5;\n"
        "}\n";
    fs <<
        "layout(binding=0) uniform sampler2D tex;\n"
        "in vec2 uv;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex,uv);\n"
        "}\n";
    pgm = device->CreateProgram();
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // uniform block
    struct UBOBlock_t
    {
        float projMat[16];
        float modelMat[16];
    };
    BufferRange uboRange;
    LWNint uboAlignment = 0;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    LWNint uboStorageSize = ROUND_UP(sizeof(UBOBlock_t), uboAlignment);
    MemoryPool* uboMemPool = device->CreateMemoryPool(NULL, uboStorageSize, MemoryPoolType::CPU_COHERENT);
    uboRange.address = uboMemPool->GetBufferAddress();
    uboRange.size = uboStorageSize;
    UBOBlock_t* uboPtr = reinterpret_cast<UBOBlock_t*>(uboMemPool->Map());
    memcpy(uboPtr->projMat, projMat, sizeof(projMat));
    memcpy(uboPtr->modelMat, modelMat, sizeof(modelMat));

    // sampler
    Sampler* smp[numALevels];
    LWNuint smpID[numALevels];
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR_MIPMAP_LINEAR, MagFilter::LINEAR);
    for (int i=0; i<numALevels; i++) {
        sb.SetMaxAnisotropy(anisoLevels[i]);
        smp[i] = sb.CreateSampler();
        smpID[i] = smp[i]->GetRegisteredID();
    }

    // texture
    Texture* tex;
    TextureHandle texHandle[numALevels];
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texDim, texDim);
    tb.SetFormat(Format::RGBA8);
    size_t texSize = tb.GetStorageSize();
    size_t texAlignment = tb.GetStorageAlignment();
    texSize = ROUND_UP(texSize, texAlignment);
    MemoryPool* texGpuMemPool = device->CreateMemoryPool(NULL, texSize, MemoryPoolType::GPU_ONLY);
    MemoryPool* texCpuMemPool = device->CreateMemoryPool(NULL, texSize, MemoryPoolType::CPU_COHERENT);
    tex = tb.CreateTextureFromPool(texGpuMemPool, 0);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer* texBuffer = bb.CreateBufferFromPool(texCpuMemPool, 0, texSize);
    dt::u8lwec4* ptr  = static_cast<dt::u8lwec4*>(texBuffer->Map());
    for (int i=0; i<texDim; i++) {
        for (int j=0; j<texDim; j++) {
            if ((i+j) % 2) {
                ptr[i*texDim+j] = dt::u8lwec4(0.0, 0.0, 1.0, 1.0);
            } else {
                ptr[i*texDim+j] = dt::u8lwec4(1.0, 1.0, 1.0, 1.0);
            }
        }
    }
    CopyRegion cr = {0,0,0, texDim,texDim,1};
    queueCB.CopyBufferToTexture(texBuffer->GetAddress(), tex, 0, &cr, CopyFlags::NONE);
    Sync* sync = device->CreateSync();
    queue->FenceSync(sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue->WaitSync(sync);
    queueCB.submit();
    queue->Finish();

    sync->Free();
    texBuffer->Free();
    for (int i=0; i<numALevels; i++) {
        texHandle[i] = device->GetTextureHandle(tex->GetRegisteredTextureID(), smpID[i]);
    }

    // draw
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    for (int i=0; i<numALevels; i++) {
        for (int j=0; j<numZAngles; j++) {
            queueCB.SetViewportScissor(cellW * j, cellH * i, cellW, cellH);
            float alpha = zAngles[j] * 0.01745329251; // deg to rad
            uboPtr->modelMat[0] = cos(alpha);
            uboPtr->modelMat[1] = -sin(alpha);
            uboPtr->modelMat[4] = sin(alpha);
            uboPtr->modelMat[5] = cos(alpha);

            /* bug 1765329, bullets 3 and 4:
             * lwnCommandBufferUpdateUniformBuffer doesn't lwrrently have coverage for LWNbufferRange.size=0 case
             * lwnCommandBufferBindUniformBuffers doesn't lwrrently have coverage for buffer size=0 case
            */
            queueCB.UpdateUniformBuffer(uboRange.address, 0, 0, 0, NULL);
            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboRange.address, 0);

            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboRange.address, uboRange.size);
            queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle[i]);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
            queueCB.submit();
            queue->Finish();
        }
    }

    // cleanup
    pgm->Free();
    tex->Free();
    for (int i=0; i<numALevels; i++) {
        smp[i]->Free();
    }
    texGpuMemPool->Free();
    texCpuMemPool->Free();
    uboMemPool->Free();
}

OGTEST_CppTest(LWNSamplerAnisotropyTest, lwn_sampler_anisotropy, );
