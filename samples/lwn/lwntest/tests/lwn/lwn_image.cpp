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

using namespace lwn;

class LWNImageLoadStoreTest
{
public:
    enum ImageAccessType {
        ImageAccessBound,
        ImageAccessBindless,
    };
    enum HandleType {
        ImageHandles,
        TextureHandles,
    };
private:
    static const int ctaWidth = 8;
    static const int ctaHeight = 8;
    ShaderStage m_stage;
    ImageAccessType m_accessType;
    HandleType m_handleType;
    bool DiscardTest;
    bool isCompute() const  { return m_stage == ShaderStage::COMPUTE; }
    bool isBound() const    { return m_accessType == ImageAccessBound; }
    void showTexture(QueueCommandBuffer &queueCB, Program *program, TextureHandle texHandle,
                    int imageWidth, int imageHeight) const;
public:
    LWNImageLoadStoreTest(ShaderStage stage, ImageAccessType accessType, HandleType handleType, bool Discard = false) :
        m_stage(stage), m_accessType(accessType), m_handleType(handleType), DiscardTest(Discard) {}
    LWNTEST_CppMethods();
};

lwString LWNImageLoadStoreTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic test of image load/store functionality in LWN, using " <<
        (isBound() ? "bound" : "bindless") << " images in " << (isCompute() ? "compute " : "graphics ") <<
        "shaders.  This test renders a Gouraud shaded rectangle in four quadrants, each of "
        "which has a white inner border and a black outer border.  The quadrants "
        "are produced by the following process:\n\n"
        "* LOWER_LEFT:  Render a quad, storing per-fragment colors in an image using (x,y)\n"
        "* LOWER_RIGHT:  Render another quad, loading per-fragment colors from "
        "the previous image using (x,y)\n"
        "* UPPER_LEFT:  Use the image as the source of a copy to texture, then render "
        "a texture mapped quad\n"
        "* UPPER_RIGHT:  Render a quad, where each (x,y) uses an atomic to get a unique "
        "offset in the image to store its (x,y) and color.  Then render a cloud of points "
        "each using vertex ID (and no attributes) to fetch (x,y) and color.\n\n"
        "At the end of the test, we check the atomic value against an expected value and "
        "draw a red overlay on the UPPER_RIGHT cell if it doesn't match.\n\n"
        "This test uses " << (m_handleType == ImageHandles ? "image" : "texture") <<
        " handles to access the image.";

    return sb.str();
}

int LWNImageLoadStoreTest::isSupported() const
{
    if (m_handleType == TextureHandles && !g_lwnDeviceCaps.supportsImagesUsingTextureHandles) {
        return 0;
    }
    return lwogCheckLWNAPIVersion(21, 5);
}

void LWNImageLoadStoreTest::showTexture(QueueCommandBuffer &queueCB, Program *program, TextureHandle texHandle,
                                        int imageWidth, int imageHeight) const
{
    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
}

void LWNImageLoadStoreTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Code fragment to compute a base color from the compute shader (x,y); we
    // don't have access to interpolation.
    static const char *csComputeBaseColorCode =
        "  color.x = (float(xy.x) + 0.5) / float(viewport.z);\n"
        "  color.y = (float(xy.y) + 0.5) / float(viewport.w);\n"
        "  color.z = (float(xy.x + xy.y) + 1.0) / float(viewport.z + viewport.w);\n"
        "  color.w = 1.0;\n";

    // Code fragment to compute the maximum distance from an (x,y) coordinate
    // to the edge.
    static const char *edgeComputeDistCode =
        "  ivec2 edgedist2 = min(xy, (viewport.zw - 1) - xy);\n"
        "  int edgedist = min(edgedist2.x, edgedist2.y);\n";

    // Code fragment to discard helper pixels and (x,y) coordinates outside
    // the viewport.
    static const char *edgeCheckCode =
        "  if (gl_HelperIlwocation || edgedist < 0) {\n"
        "    discard;\n"
        "  }\n";

    // Code fragment to set a black or white color on fragments near the edge
    // of the viewport, using the distance computed in fsEdgeCheck.
    static const char *edgeAdjustColorCode =
        "  if (edgedist < 8) {\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    if (edgedist < 4) {\n"
        "      color = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "    }\n"
        "  }\n";

    // Common vertex shader for drawing a big pass-through quad.  We also
    // construct a set of texture coordinates (otc) using position.
    VertexShader vs(450);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "out vec2 otc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "  otc = 0.5 * position.xy - 0.5;\n"
        "}\n";

    // Shader used to store the fragment color to an image using an index
    // computed from a viewport-relative (x,y).
    FragmentShader imageStoreFS(450);
    if (DiscardTest) {
        imageStoreFS <<
            "#pragma HoistDiscards true\n";
    }
    imageStoreFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n";
    if (isBound()) {
        imageStoreFS <<
            "layout(rgba32f, binding = 0) uniform image2D image0;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageStoreFS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageStoreFS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  layout(rgba32f) image2D image0;"
            "};\n";
    }
    imageStoreFS <<
        "void main() {\n"
        "  vec4 color = vec4(ocolor, 1.0);\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        edgeAdjustColorCode <<
        "  imageStore(image0, xy, color/2.0);\n"
        "  fcolor = color;\n"
        "}\n";

    ComputeShader imageStoreCS(450);
    imageStoreCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        imageStoreCS <<
            "layout(rgba32f, binding = 0) uniform image2D image0;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageStoreCS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageStoreCS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  layout(rgba32f) image2D image0;\n"
            "};\n";
    }
    imageStoreCS <<
        "void main() {\n"
        "  vec4 color;\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        csComputeBaseColorCode <<
        edgeComputeDistCode <<
        edgeAdjustColorCode <<
        "  imageStore(image0, xy, color/2.0);\n"
        "}\n";

    // Shader used to load a final fragment color from a image using an index
    // computed from a viewport-relative (x,y).
    FragmentShader imageLoadFS(450);
    imageLoadFS <<
        "out vec4 fcolor;\n";
    if (isBound()) {
        imageLoadFS <<
            "layout(rgba32f, binding = 0) uniform image2D image0;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageLoadFS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageLoadFS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  layout(rgba32f) image2D image0;\n"
            "};\n";
    }
    imageLoadFS <<
        "void main() {\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        "  fcolor = imageLoad(image0, xy) * 2.0;\n"
        "}\n";

    ComputeShader imageLoadCS(450);
    imageLoadCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        imageLoadCS <<
            "layout(rgba32f, binding = 0) uniform image2D image0;\n"
            "layout(rgba32f, binding = 1) uniform image2D image1;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageLoadCS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageLoadCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        imageLoadCS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint64_t imageHandle0, imageHandle1;\n"
            "};\n";
    }
    imageLoadCS <<
        "void main() {\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        edgeComputeDistCode;
    if (!isBound()) {
        // Workaround for GLSL compiler bug 1672355.
        imageLoadCS <<
            "  layout(rgba32f) image2D image0 = layout(rgba32f) image2D(imageHandle0);\n"
            "  layout(rgba32f) image2D image1 = layout(rgba32f) image2D(imageHandle1);\n";
    }
    imageLoadCS <<
        "  vec4 color = imageLoad(image0, xy);\n"
        "  imageStore(image1, xy, color * 2.0 / 3.0);\n"
        "}\n";

    // Shader used to display a texture initialized from image contents.
    FragmentShader texDisplay2FS(450);
    texDisplay2FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 2.0;\n"
        "}\n";

    FragmentShader texDisplay3FS(450);
    texDisplay3FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 3.0;\n"
        "}\n";

    FragmentShader texDisplay4FS(450);
    texDisplay4FS <<
        "in vec2 otc;\n"
        "out vec4 fcolor;\n"
        "layout(binding=0) uniform sampler2D tex;\n"
        "void main() {\n"
        "  fcolor = texture(tex, otc) * 4.0;\n"
        "}\n";

    // Shader used to store a fragment's (x,y) and color to an image using a
    // unique index derived from an atomic.
    FragmentShader imageCloudStoreFS(450);
    imageCloudStoreFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n";
    if (isBound()) {
        imageCloudStoreFS <<
            "layout(rg32i, binding = 0) uniform iimage2D image0;\n"
            "layout(rgba32f, binding = 1) uniform image2D image1;\n"
            "layout(r32ui, binding = 2) uniform uimageBuffer image2;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageCloudStoreFS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageCloudStoreFS.addExtension(lwShaderExtension::LW_gpu_shader5);
        imageCloudStoreFS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint64_t imageHandle0, imageHandle1, imageHandle2;\n"
            "};\n";
    }
    imageCloudStoreFS <<
        "void main() {\n"
        "  vec4 color = vec4(ocolor, 1.0);\n"
        "  ivec2 xy = ivec2(gl_FragCoord.xy) - viewport.xy;\n" <<
        edgeComputeDistCode <<
        edgeCheckCode <<
        edgeAdjustColorCode;
    if (!isBound()) {
        // Workaround for GLSL compiler bug 1672355.
        imageCloudStoreFS <<
            "  layout(rg32i) iimage2D image0 = layout(rg32i) iimage2D(imageHandle0);\n"
            "  layout(rgba32f) image2D image1 = layout(rgba32f) image2D(imageHandle1);\n"
            "  layout(r32ui) uimageBuffer image2 = layout(r32ui) uimageBuffer(imageHandle2);\n";
    }
    imageCloudStoreFS <<
        "  uint index = imageAtomicAdd(image2, 0, 1);\n"
        "  ivec2 ixy = ivec2(index % viewport.z, index / viewport.z);\n"
        "  imageStore(image0, ixy, ivec4(xy, 0, 0));\n"
        "  imageStore(image1, ixy, color * 4.0);\n"
        "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);  // draw red (should be overwritten)\n"
        "}\n";

    ComputeShader imageCloudStoreCS(450);
    imageCloudStoreCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        imageCloudStoreCS <<
            "layout(rg32i, binding = 0) uniform iimage2D image0;\n"
            "layout(rgba32f, binding = 1) uniform image2D image1;\n"
            "layout(r32ui, binding = 2) uniform uimageBuffer image2;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageCloudStoreCS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageCloudStoreCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        imageCloudStoreCS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint64_t imageHandle0, imageHandle1, imageHandle2;\n"
            "};\n";
    }
    imageCloudStoreCS <<
        "void main() {\n"
        "  vec4 color;\n"
        "  ivec2 xy = ivec2(gl_GlobalIlwocationID.xy);\n" <<
        csComputeBaseColorCode <<
        edgeComputeDistCode <<
        edgeAdjustColorCode;
    if (!isBound()) {
        // Workaround for GLSL compiler bug 1672355.
        imageCloudStoreCS <<
            "  layout(rg32i) iimage2D image0 = layout(rg32i) iimage2D(imageHandle0);\n"
            "  layout(rgba32f) image2D image1 = layout(rgba32f) image2D(imageHandle1);\n"
            "  layout(r32ui) uimageBuffer image2 = layout(r32ui) uimageBuffer(imageHandle2);\n";
    }
    imageCloudStoreCS <<
        "  uint index = imageAtomicAdd(image2, 0, 1);\n"
        "  ivec2 ixy = ivec2(index % viewport.z, index / viewport.z);\n"
        "  imageStore(image0, ixy, ivec4(xy, 0, 0));\n"
        "  imageStore(image1, ixy, color * 4.0);\n"
        "}\n";

    // Shader used to load a fragment's (x,y) and color from an image using
    // gl_VertexID.
    VertexShader imageCloudLoadVS(450);
    if (isBound()) {
        imageCloudLoadVS <<
            "layout(rg32i, binding = 0) uniform iimage2D image0;\n"
            "layout(rgba32f, binding = 1) uniform image2D image1;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageCloudLoadVS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageCloudLoadVS.addExtension(lwShaderExtension::LW_gpu_shader5);
        imageCloudLoadVS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint64_t imageHandle0, imageHandle1;\n"
            "};\n";
    }
    imageCloudLoadVS <<
        "out vec4 ocolor;\n"
        "void main() {\n";
    if (!isBound()) {
        // Workaround for GLSL compiler bug 1672355.
        imageCloudLoadVS <<
            "  layout(rg32i) iimage2D image0 = layout(rg32i) iimage2D(imageHandle0);\n"
            "  layout(rgba32f) image2D image1 = layout(rgba32f) image2D(imageHandle1);\n";
    }
    // writing to output while imageLoad for bug:200188343
    // Division by 4.0 introduces precision issues while compiling as fp16
    imageCloudLoadVS <<
        "  ivec2 ixy = ivec2(gl_VertexID % viewport.z, gl_VertexID / viewport.z);\n"
        "  ivec2 fragcoord = imageLoad(image0, ixy).xy;\n"
        "  ocolor = imageLoad(image1, ixy) / 4.0;\n"
        "  vec2 viewsize = vec2(viewport.zw);\n"
        "  gl_Position.xy = 2.0 * ((fragcoord + vec2(0.5)) / viewsize) - 1.0;\n"
        "  gl_Position.zw = vec2(0.0, 1.0);\n"
        "}\n";

    // Displays the color loaded from the vertex shader.
    FragmentShader imageCloudLoadFS(450);
    if (!isBound()) imageCloudLoadFS.addExtension(lwShaderExtension::LW_bindless_texture);
    imageCloudLoadFS <<
        "in vec4 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = ocolor;\n"
        "}\n";

    ComputeShader imageCloudLoadCS(450);
    imageCloudLoadCS.setCSGroupSize(ctaWidth, ctaHeight);
    if (isBound()) {
        imageCloudLoadCS <<
            "layout(rg32i, binding = 0) uniform iimage2D image0;\n"
            "layout(rgba32f, binding = 1) uniform image2D image1;\n"
            "layout(rgba32f, binding = 2) uniform image2D image2;\n"
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "};\n";
    } else {
        imageCloudLoadCS.addExtension(lwShaderExtension::LW_bindless_texture);
        imageCloudLoadCS.addExtension(lwShaderExtension::LW_gpu_shader5);
        imageCloudLoadCS <<
            "layout(std140, binding = 0) uniform UBO {\n"
            "  ivec4 viewport;\n"
            "  uint64_t imageHandle0, imageHandle1, imageHandle2;\n"
            "};\n";
    }
    imageCloudLoadCS <<
        "void main() {\n";
    if (!isBound()) {
        // Workaround for GLSL compiler bug 1672355.
        imageCloudLoadCS <<
            "  layout(rg32i) iimage2D image0 = layout(rg32i) iimage2D(imageHandle0);\n"
            "  layout(rgba32f) image2D image1 = layout(rgba32f) image2D(imageHandle1);\n"
            "  layout(rgba32f) image2D image2 = layout(rgba32f) image2D(imageHandle2);\n";
    }
    imageCloudLoadCS <<
        "  ivec2 ixy = ivec2(gl_GlobalIlwocationID.xy);\n"
        "  ivec2 fragcoord = imageLoad(image0, ixy).xy;\n"
        "  vec4 color = imageLoad(image1, ixy);\n"
        "  vec2 viewsize = vec2(viewport.zw);\n"
        "  imageStore(image2, fragcoord, color / 16.0);\n"
        "}\n";

    // Load up all the programs.
    enum TestProgramType {
        ImageStore,
        ImageLoad,
        ImageCloudStore,
        ImageCloudLoad,
        DisplayTex2,
        DisplayTex3,
        DisplayTex4,
        TestProgramCount
    };
    struct TestProgramConfig {
        VertexShader *vs;
        FragmentShader *fs;
        ComputeShader *cs;
    };
    TestProgramConfig programConfigs[TestProgramCount] = {
        { &vs, &imageStoreFS, &imageStoreCS },
        { &vs, &imageLoadFS, &imageLoadCS },
        { &vs, &imageCloudStoreFS, &imageCloudStoreCS },
        { &imageCloudLoadVS, &imageCloudLoadFS, &imageCloudLoadCS },
        { &vs, &texDisplay2FS, NULL },
        { &vs, &texDisplay3FS, NULL },
        { &vs, &texDisplay4FS, NULL },
    };
    Program *programs[TestProgramCount];
    for (int i = 0; i < TestProgramCount; i++) {
        programs[i] = device->CreateProgram();
        ComputeShader *cs = programConfigs[i].cs;
        if (isCompute() && cs) {
            if (!g_glslcHelper->CompileAndSetShaders(programs[i], *cs)) {
                LWNFailTest();
                return;
            }
        } else {
            VertexShader *vs = programConfigs[i].vs;
            FragmentShader *fs = programConfigs[i].fs;
            if (!g_glslcHelper->CompileAndSetShaders(programs[i], *vs, *fs)) {
                LWNFailTest();
                return;
            }
        }
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec3(1.0, 1.0, 1.0) },
    };
    MemoryPoolAllocator allocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device,  4, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // We program our images to hold 1/4 of the full window size rounded down
    // to multiples of compute CTA size units.  In the point cloud case, we'll
    // have separate images to store an (x,y) and (R,G,B,A) per vertex.  We
    // also have a spare RGBA image and a buffer texture used for atomics.
    int imageWidth = ctaWidth * (lwrrentWindowWidth / (2 * ctaWidth));
    int imageHeight = ctaHeight * (lwrrentWindowHeight / (2 * ctaHeight));

    TextureBuilder ib;
    ib.SetDevice(device).SetDefaults();
    ib.SetFlags(TextureFlags::IMAGE);
    ib.SetTarget(TextureTarget::TARGET_2D);
    ib.SetSize2D(imageWidth, imageHeight);
    ib.SetFormat(Format::RGBA32F);
    ib.SetLevels(1);

    size_t imageSize = ib.GetStorageSize();
    size_t imageAlignment = ib.GetStorageAlignment();
    size_t imagePoolSize = 4 * imageAlignment * ((imageSize + imageAlignment - 1) / imageAlignment);

    MemoryPoolAllocator imageAllocator(device, NULL, imagePoolSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *textures[4];
    textures[0] = imageAllocator.allocTexture(&ib);
    textures[1] = imageAllocator.allocTexture(&ib);
    ib.SetFormat(Format::RG32I);
    textures[2] = imageAllocator.allocTexture(&ib);
    ib.SetSize1D(1);
    ib.SetTarget(TextureTarget::TARGET_BUFFER);
    ib.SetFormat(Format::RG32UI);
    textures[3] = imageAllocator.allocTexture(&ib);

    // We program four separate sysmem UBO copies, one for each quadrant, and
    // fill each with a single ivec4 holding (x,y,w,h).
    int uboVersions = 4;
    LWNint uboAlignment;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    int uboSize = uboAlignment;
    if (uboSize < LWNint(sizeof(dt::ivec4))) {
        uboSize = sizeof(dt::ivec4);
    }
    MemoryPoolAllocator uboAllocator(device, NULL, uboVersions * uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboVersions * uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    char *uboMem = (char *) ubo->Map();
    struct UBOLayout {
        dt::ivec4       viewport;
        ImageHandle     image0, image1, image2;
    };
    UBOLayout *uboContents;
    for (int i = 0; i < 4; i++) {
        uboContents = (UBOLayout *) (uboMem + i * uboSize);
        uboContents->viewport = dt::ivec4((i & 1) ? imageWidth : 0, (i & 2) ? imageHeight: 0,
                                          imageWidth, imageHeight);
    }

    // Set a sampler for rendering using image contents.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    sb.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    Sampler *sampler = sb.CreateSampler();

    // Set up texture and image handles for each of the images.
    TextureHandle texHandles[4];
    ImageHandle imageHandles[4];
    for (int i = 0; i < 4; i++) {
        texHandles[i] = device->GetTextureHandle(textures[i]->GetRegisteredTextureID(), sampler->GetRegisteredID());
        if (m_handleType == ImageHandles) {
            LWNuint id = g_lwnTexIDPool->RegisterImage(textures[i]);
            imageHandles[i] = device->GetImageHandle(id);
        } else {
            imageHandles[i] = texHandles[i];
        }
    }

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    ShaderStageBits programBindMask = ShaderStageBits::ALL_GRAPHICS_BITS;
    if (isCompute()) {
        programBindMask = ShaderStageBits::COMPUTE;
    }

    // Render the store-to-image pass in the lower left quadrant.
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 0 * uboSize, uboSize);
    queueCB.SetViewportScissor(0, 0, imageWidth, imageHeight);
    queueCB.BindProgram(programs[ImageStore], programBindMask);
    if (isBound()) {
        queueCB.BindImage(m_stage, 0, imageHandles[0]);
    } else {
        uboContents = (UBOLayout *) (uboMem + 0 * uboSize);
        uboContents->image0 = imageHandles[0];
    }
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 0 * uboSize, uboSize);
        queueCB.DispatchCompute(imageWidth / ctaWidth, imageHeight / ctaHeight, 1);
        showTexture(queueCB, programs[DisplayTex2], texHandles[0], imageWidth, imageHeight);
    } else {
        queueCB.BindProgram(programs[ImageStore], ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Insert a barrier to make sure the store is fully done before the
    // subsequent load pass.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER |
                    BarrierBits::ILWALIDATE_TEXTURE);

    // Render the load-from-image pass in the lower right quadrant.
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 1 * uboSize, uboSize);
    queueCB.SetViewportScissor(imageWidth, 0, imageWidth, imageHeight);
    queueCB.BindProgram(programs[ImageLoad], programBindMask);
    if (isBound()) {
        queueCB.BindImage(m_stage, 0, imageHandles[0]);
    } else {
        uboContents = (UBOLayout *) (uboMem + 1 * uboSize);
        uboContents->image0 = imageHandles[0];
        if (isCompute()) {
            uboContents->image1 = imageHandles[1];
        }
    }
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 1 * uboSize, uboSize);
        if (isBound()) {
            queueCB.BindImage(m_stage, 1, imageHandles[1]);
        }
        queueCB.DispatchCompute(imageWidth / ctaWidth, imageHeight / ctaHeight, 1);
        showTexture(queueCB, programs[DisplayTex3], texHandles[1], imageWidth, imageHeight);
    } else {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Copy from the image to a texture and render with the texture in the
    // upper left.
    queueCB.SetViewportScissor(0, imageHeight, imageWidth, imageHeight);
    showTexture(queueCB, programs[DisplayTex2], texHandles[0], imageWidth, imageHeight);

    // Now generate data for the the point cloud pass.  CopyBuffer here
    // initializes the atomic counter to zero (by copying from the zero at the
    // beginning of our first UBO).
    CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };
    queueCB.CopyBufferToTexture(ubo->GetAddress(), textures[3], NULL, &copyRegion, CopyFlags::NONE);
    queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr + 3 * uboSize, uboSize);
    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr + 3 * uboSize, uboSize);
    queueCB.SetViewportScissor(imageWidth, imageHeight, imageWidth, imageHeight);
    queueCB.BindProgram(programs[ImageCloudStore], programBindMask);
    if (isBound()) {
        queueCB.BindImage(m_stage, 0, imageHandles[2]); // xy
        queueCB.BindImage(m_stage, 1, imageHandles[0]); // rgba
        queueCB.BindImage(m_stage, 2, imageHandles[3]); // counters
    } else {
        uboContents = (UBOLayout *) (uboMem + 3 * uboSize);
        uboContents->image0 = imageHandles[2];
        uboContents->image1 = imageHandles[0];
        uboContents->image2 = imageHandles[3];
    }
    if (isCompute()) {
        queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 3 * uboSize, uboSize);
        queueCB.DispatchCompute(imageWidth / ctaWidth, imageHeight / ctaHeight, 1);
    } else {
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    }

    // Wait for the point cloud data to be ready.
    queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES | BarrierBits::ILWALIDATE_SHADER |
                    BarrierBits::ILWALIDATE_TEXTURE);

    // For bindless, set up and bind (unused) UBO slot 2 to hold the same
    // viewport data as slot 3, but with different bindings.
    if (!isBound()) {
        uboContents = (UBOLayout *) (uboMem + 2 * uboSize);
        *uboContents = *((UBOLayout *) (uboMem + 3 * uboSize));
        uboContents->image0 = imageHandles[2];
        uboContents->image1 = imageHandles[0];
        uboContents->image2 = imageHandles[1];
        if (isCompute()) {
            queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr + 2 * uboSize, uboSize);
        } else {
            queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr + 2 * uboSize, uboSize);
        }
    }

    // Render the point cloud.
    queueCB.BindProgram(programs[ImageCloudLoad], programBindMask);
    if (isCompute()) {
        if (isBound()) {
            queueCB.BindImage(m_stage, 2, imageHandles[1]); // rgba output
        }
        queueCB.DispatchCompute(imageWidth / ctaWidth, imageHeight / ctaHeight, 1);
        showTexture(queueCB, programs[DisplayTex4], texHandles[1], imageWidth, imageHeight);
    } else {
        if (isBound()) {
            queueCB.BindImage(ShaderStage::VERTEX, 0, imageHandles[2]); // xy
            queueCB.BindImage(ShaderStage::VERTEX, 1, imageHandles[0]); // rgba
        }

        // Set up "null" vertex state for the point cloud pass, which uses no
        // attributes other than gl_VertexID.
        queueCB.BindVertexAttribState(0, NULL);
        queueCB.BindVertexStreamState(0, NULL);
        queueCB.DrawArrays(DrawPrimitive::POINTS, 0, imageWidth * imageHeight);
    }

    // When finished, copy the first word of the image (the counter) back to
    // our UBO memory and check with the CPU.  If it doesn't match, clear a
    // red rectangle on top of our upper right quadrant.
    queueCB.CopyTextureToBuffer(textures[3], NULL, &copyRegion, uboAddr, CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();
    if (*((LWNuint *) uboMem) != LWNuint(imageWidth * imageHeight)) {
#if 0
        printf("Count is %d, expected %d\n", *((LWNuint *) uboMem), imageWidth * imageHeight);
#endif
        queueCB.SetViewportScissor(3 * imageWidth / 2 - 16, 3 * imageHeight / 2 - 16, 32, 32);
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        queueCB.submit();
    }

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_basic,
               (ShaderStage::FRAGMENT, LWNImageLoadStoreTest::ImageAccessBound,
                LWNImageLoadStoreTest::ImageHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_basic_discard,
               (ShaderStage::FRAGMENT, LWNImageLoadStoreTest::ImageAccessBound,
                LWNImageLoadStoreTest::ImageHandles, true));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_compute,
               (ShaderStage::COMPUTE, LWNImageLoadStoreTest::ImageAccessBound,
                LWNImageLoadStoreTest::ImageHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_bindless_basic,
               (ShaderStage::FRAGMENT, LWNImageLoadStoreTest::ImageAccessBindless,
                LWNImageLoadStoreTest::ImageHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_bindless_compute,
               (ShaderStage::COMPUTE, LWNImageLoadStoreTest::ImageAccessBindless,
                LWNImageLoadStoreTest::ImageHandles));

OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_basic_th,
               (ShaderStage::FRAGMENT, LWNImageLoadStoreTest::ImageAccessBound,
                LWNImageLoadStoreTest::TextureHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_compute_th,
               (ShaderStage::COMPUTE, LWNImageLoadStoreTest::ImageAccessBound,
                LWNImageLoadStoreTest::TextureHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_bindless_basic_th,
               (ShaderStage::FRAGMENT, LWNImageLoadStoreTest::ImageAccessBindless,
                LWNImageLoadStoreTest::TextureHandles));
OGTEST_CppTest(LWNImageLoadStoreTest, lwn_image_bindless_compute_th,
               (ShaderStage::COMPUTE, LWNImageLoadStoreTest::ImageAccessBindless,
                LWNImageLoadStoreTest::TextureHandles));
