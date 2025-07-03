/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
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

class LWNBindingTest
{
public:
    enum TestType {
        TEST_TEXTURES = 0,
        TEST_IMAGES,
        TEST_UBOS,
        TEST_SSBOS,
        TEST_VBOS
    };
private:
    static const int numTextures = 32;          //lwn::DeviceInfo::TEXTURE_BINDINGS_PER_STAGE
    static const int numImages = 8;             //lwn::DeviceInfo::IMAGE_BINDINGS_PER_STAGE
    static const int numUniformBuffers = 14;    //lwn::DeviceInfo::UNIFORM_BUFFER_BINDINGS_PER_STAGE
    static const int numStorageBuffers = 16;    //lwn::DeviceInfo::SHADER_STORAGE_BUFFER_BINDINGS_PER_STAGE
    static const int numVertexBuffers = 16;     //lwn::DeviceInfo::VERTEX_BUFFER_BINDINGS
    static const int maxBindings = 32;  //max of the above constants

    static const int numVerticesPerQuad = 6;

    static const int texSize = 1;

    static const int cellSizeX = maxBindings * 8;
    static const int cellSizeY = 64;
    static const int cellMargin = 16;

    int      m_nBindings;
    TestType m_testType;

    void drawRow(Device *device, QueueCommandBuffer &cb,
                 const uint64_t* bindables, int row, int column) const;

public:
    LWNBindingTest(TestType testType) : m_testType(testType)
    {
        static const int numBindings[] = {
            numTextures, numImages, numUniformBuffers, numStorageBuffers, numVertexBuffers };
        m_nBindings = numBindings[testType];
        assert(m_nBindings <= maxBindings);
    }
    LWNTEST_CppMethods();
};

lwString LWNBindingTest::getDescription() const
{
    lwStringBuf sb;
    switch (m_testType) {
    case TEST_TEXTURES:
        sb <<
            "Test exercising texture binding API.\n\n"
            "There are two columns of test cases: the left column uses BindTextures API, the right one "
            "BindTexture API. The images in each column should match. The test allocates 32 textures, "
            "one for each texture binding point. In each cell, each constant color column is drawn with "
            "a separate texture. The textures are filled with increasing levels of gray. The rows of cells "
            "from bottom to top are: 1. bind all 32 textures in order 2. reverse order of textures 0-7 "
            "3. reverse order of textures 8-15 4. reverse order of textures 16-23 5. reverse order of "
            "textures 24-31. The result is that the top row has textures in the opposite order compared "
            "to the bottom row.";
        break;
    case TEST_IMAGES:
        sb <<
            "Test exercising image binding API.\n\n"
            "There are two columns of test cases: the left column uses BindImages API, the right one "
            "BindImage API. The images in each column should match. The test allocates 8 images, "
            "one for each image binding point. In each cell, each constant color column is drawn with "
            "a separate image. The images are filled with increasing levels of gray. The rows of cells "
            "from bottom to top are: 1. bind all 8 images in order 2. reverse order of images 0-1 "
            "3. reverse order of images 2-3 4. reverse order of images 4-5 5. reverse order of "
            "images 6-7. The result is that the top row has images in the opposite order compared "
            "to the bottom row.";
        break;
    case TEST_UBOS:
        sb <<
            "Test exercising uniform buffer binding API.\n\n"
            "There are two columns of test cases: the left column uses BindUniformBuffers API, the right one "
            "BindUniformBuffer API. In each cell, each constant color column is drawn with a separate uniform "
            "block. The test consists of allocating a uniform buffer, filling it with increasing levels of gray, "
            "and then binding ranges of the buffer to uniform buffer binding points. The resulting images in the "
            "left and right column should match, indicating that the two APIs produce the same result. "
            "The rows of cells from bottom to top are: 1. bind all 14 uniform blocks in order 2. reverse order of "
            "blocks 0-2 3. reverse order of blocks 3-7 4. reverse order of blocks 8-10 5. reverse order of "
            "blocks 11-14. The result is that the top row has colors in the opposite order compared "
            "to the bottom row.";
        break;
    case TEST_SSBOS:
        sb <<
            "Test exercising shader storage buffer binding API.\n\n"
            "There are two columns of test cases: the left column uses BindStorageBuffers API, the right one "
            "BindStorageBuffer API. In each cell, each constant color column is drawn with a separate storage "
            "block. The test consists of allocating a storage buffer, filling it with increasing levels of gray, "
            "and then binding ranges of the buffer to storage buffer binding points. The resulting images in the "
            "left and right column should match, indicating that the two APIs produce the same result. "
            "The rows of cells from bottom to top are: 1. bind all 16 storage blocks in order 2. reverse order of "
            "blocks 0-3 3. reverse order of blocks 4-7 4. reverse order of blocks 8-11 5. reverse order of "
            "blocks 12-15. The result is that the top row has colors in the opposite order compared "
            "to the bottom row.";
        break;
    case TEST_VBOS:
        sb <<
            "Test exercising vertex buffer binding API.\n\n"
            "There are two columns of test cases: the left column uses BindVertexBuffers API, the right one "
            "BindVertexBuffer API. In each cell, each constant color column is drawn with vertices from a separate "
            "buffer. The test allocates 16 vertex buffers, each containing vertices and colors for 16 quads "
            "in a row. In each buffer, vertex colors are a constant shade of grey, with brightness increasing "
            "for each buffer. The test binds various sets of vertex buffers to vertex buffer binding "
            "points. The resulting images in the left and right column should match, indicating that the two "
            "APIs produce the same result. "
            "The rows of cells from bottom to top are: 1. bind all 16 vertex buffers in order 2. reverse order of "
            "buffers 0-3 3. reverse order of buffers 4-7 4. reverse order of buffers 8-11 5. reverse order of "
            "buffers 12-15. The result is that the top row has colors in the opposite order compared "
            "to the bottom row.";
        break;
    default: assert(0); break;
    }
    return sb.str();
}

int LWNBindingTest::isSupported() const
{
    if (m_testType == TEST_TEXTURES)
        return lwogCheckLWNAPIVersion(40, 3);
    else
        return lwogCheckLWNAPIVersion(40, 14);
}

void LWNBindingTest::drawRow(Device *device, QueueCommandBuffer &cb, const uint64_t * bindables, int row, int column) const
{
    uint64_t addresses[maxBindings];
    BufferRange buffers[maxBindings];
    for (int i = 0; i < m_nBindings; ++i) {
        int id = i;
        if (row)
            id = m_nBindings - 1 - i;
        addresses[i] = buffers[i].address = bindables[id];
        switch (m_testType) {
        case TEST_UBOS: buffers[i].size = 4 * sizeof(LWNuint); break;
        case TEST_SSBOS: buffers[i].size = sizeof(LWNuint); break;
        case TEST_VBOS: buffers[i].size = numVertexBuffers * numVerticesPerQuad * sizeof(dt::vec4); break;
        default: buffers[i].size = 0; break;
        }
    }
    int first = 0;
    int count = m_nBindings;
    if (row > 0) {
        first = ((row-1)*m_nBindings) >> 2;
        count = ((row*m_nBindings) >> 2) - first;
    }
    if (column) {
        for (int i = first; i < first + count; ++i) {
            switch (m_testType) {
            case TEST_TEXTURES:
                cb.BindTexture(ShaderStage::FRAGMENT, i, addresses[i]);
                break;
            case TEST_IMAGES:
                cb.BindImage(ShaderStage::FRAGMENT, i, addresses[i]);
                break;
            case TEST_UBOS:
                cb.BindUniformBuffer(ShaderStage::FRAGMENT, i, buffers[i].address, buffers[i].size);
                break;
            case TEST_SSBOS:
                cb.BindStorageBuffer(ShaderStage::FRAGMENT, i, buffers[i].address, buffers[i].size);
                break;
            case TEST_VBOS:
                cb.BindVertexBuffer(i, buffers[i].address, buffers[i].size);
                break;
            default: assert(0); break;
            }
        }
    } else {
        switch (m_testType) {
        case TEST_TEXTURES:
            cb.BindTextures(ShaderStage::FRAGMENT, first, count, addresses + first);
            break;
        case TEST_IMAGES:
            cb.BindImages(ShaderStage::FRAGMENT, first, count, addresses + first);
            break;
        case TEST_UBOS:
            cb.BindUniformBuffers(ShaderStage::FRAGMENT, first, count, buffers + first);
            break;
        case TEST_SSBOS:
            cb.BindStorageBuffers(ShaderStage::FRAGMENT, first, count, buffers + first);
            break;
        case TEST_VBOS:
            cb.BindVertexBuffers(first, count, buffers + first);
            break;
        default: assert(0); break;
        }
    }
    cb.SetViewport(column*(cellSizeX + cellMargin), row * (cellSizeY + cellMargin), cellSizeX, cellSizeY);
    if (m_testType == TEST_VBOS)
        cb.DrawArrays(DrawPrimitive::TRIANGLES, 0, numVertexBuffers * numVerticesPerQuad);
    else
        cb.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
}

void LWNBindingTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    MemoryPoolAllocator gpu_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator coherent_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexShader vs(440);
    if (m_testType != TEST_VBOS) {
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec2 texcoord;\n"
            "out vec2 otc;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  otc = texcoord;\n"
            "}\n";
    } else {
        vs <<
            "const int numVertexBuffers = 16;\n"
            "const int numVerticesPerQuad = 6;\n"
            "layout(location=0) in vec4 positionColor[numVertexBuffers];\n"
            "out vec4 color;\n"
            "void main() {\n"
            "  int t = gl_VertexID / numVerticesPerQuad;\n"
            "  vec4 p = positionColor[t];\n"
            "  gl_Position = vec4(p.xyz, 1.0f);\n"
            "  color = vec4(p.w);\n"
            "}\n";
    }

    FragmentShader fs(440);
    switch (m_testType) {
    case TEST_TEXTURES:
        fs <<
            "const int numTextures = 32;\n"
            "layout(binding = 0) uniform sampler2D tex[numTextures];\n"
            "in vec2 otc;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  int t = int(floor(otc.x*float(numTextures))) % numTextures;\n"
            "  fcolor = texture(tex[t], otc);\n"
            "}\n";
        break;
    case TEST_IMAGES:
        fs <<
            "const int numImages = 8;\n"
            "layout(rgba8, binding = 0) uniform image2D img[numImages];\n"
            "in vec2 otc;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  int t = int(floor(otc.x*float(numImages))) % numImages;\n"
            "  fcolor = imageLoad(img[t], ivec2(otc));\n"
            "}\n";
        break;
    case TEST_UBOS:
        fs <<
            "const int numUniformBuffers = 14;\n"
            "layout(binding = 0, std140) uniform Block { uvec4 color; } ubos[numUniformBuffers];\n"
            "in vec2 otc;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  int t = int(floor(otc.x*float(numUniformBuffers))) % numUniformBuffers;\n"
            "  fcolor = vec4(ubos[t].color)/255.0f;\n"
            "}\n";
        break;
    case TEST_SSBOS:
        fs <<
            "const int numStorageBuffers = 16;\n"
            "layout(binding = 0, std140) buffer Block { uint color; } ssbos[numStorageBuffers];\n"
            "in vec2 otc;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  int t = int(floor(otc.x*float(numStorageBuffers))) % numStorageBuffers;\n"
            "  uint c = ssbos[t].color;\n"
            "  fcolor = vec4(((c>>24u)&255u), ((c>>16u)&255u), ((c>>8u)&255u), (c&255u))/255.0f;\n"
            "}\n";
        break;
    case TEST_VBOS:
        fs <<
            "in vec4 color;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = color;\n"
            "}\n";
        break;
    default: assert(0); break;
    }

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    // Set up texture builder
    TextureBuilder tb;
    tb.SetDevice(device);
    tb.SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetFormat(Format::RGBA8);
    tb.SetFlags(m_testType == TEST_IMAGES ? TextureFlags::IMAGE : TextureFlags(0));
    tb.SetSize2D(texSize, texSize);
    tb.SetLevels(1);
    Texture *textures[maxBindings];

    // Set up UBO
    const int uboAlignment = 256;
    const int uboSize = m_nBindings * uboAlignment;
    MemoryPoolAllocator allocator(device, NULL, uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder uboBuilder;
    uboBuilder.SetDefaults();
    uboBuilder.SetDevice(device);
    Buffer *uboSysMem = allocator.allocBuffer(&uboBuilder, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    LWNuint *uboptr = (LWNuint*) uboSysMem->Map();
    uint64_t uboAddrs[maxBindings];

    // Set up SSBO
    const int texDataSize = texSize*texSize * 4;
    const int ssboSize = m_nBindings * texDataSize;
    MemoryPoolAllocator ssboAllocator(device, NULL, ssboSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Buffer *ssbo = ssboAllocator.allocBuffer(&uboBuilder, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    uint64_t ssboAddrs[maxBindings];

    // Create a staging buffer to hold texel data.
    Buffer *stagingBuffer;
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    stagingBuffer = coherent_allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, 64 * 1024);
    uint8_t *stagingMem = (uint8_t *)stagingBuffer->Map();

    // Allocate textures, upload data to textures, UBO and SSBO
    int stagingOffset = 0;
    for (int i = 0; i < m_nBindings; i++) {
        int imageBase = stagingOffset;
        textures[i] = gpu_allocator.allocTexture(&tb);
        uint8_t shade = ((uint8_t)(i+1) * (256/m_nBindings)) - 1;
        for (int y = 0; y < texSize; y++) {
            for (int x = 0; x < texSize; x++) {
                for (int c = 0; c < 4; c++) {
                    stagingMem[stagingOffset + c] = shade;
                }
                stagingOffset += 4;
            }
        }
        for (int c = 0; c < 4; c++) {
            uboptr[i*(uboAlignment / sizeof(LWNuint)) + c] = shade;
        }
        uboAddrs[i] = uboSysMem->GetAddress() + i * uboAlignment;
        ssboAddrs[i] = ssbo->GetAddress() + i * texDataSize;

        // Upload texture data
        CopyRegion region = {0, 0, 0, texSize, texSize, 1};
        queueCB.CopyBufferToTexture(stagingBuffer->GetAddress() + imageBase, textures[i], NULL, &region, LWN_COPY_FLAGS_NONE);

        // Upload SSBO data
        queueCB.CopyBufferToBuffer(stagingBuffer->GetAddress() + imageBase, ssboAddrs[i], texDataSize, LWN_COPY_FLAGS_NONE);
    }

    // Set up a basic sampler object.
    Sampler *sampler;
    SamplerBuilder sb;
    sb.SetDevice(device);
    sb.SetDefaults();
    sb.SetWrapMode(WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE);
    sb.SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    sampler = sb.CreateSampler();

    // Set up texture handles for all our textures.
    TextureHandle textureHandles[maxBindings];
    ImageHandle imageHandles[maxBindings];
    for (int i = 0; i < m_nBindings; ++i) {
        textureHandles[i] = device->GetTextureHandle(textures[i]->GetRegisteredTextureID(),
                                                     sampler->GetRegisteredID());
        if (m_testType == TEST_IMAGES) {
            LWNuint id = g_lwnTexIDPool->RegisterImage(textures[i]);
            imageHandles[i] = device->GetImageHandle(id);
        }
    }

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    if (m_testType != TEST_VBOS) {
        // Set up the vertex format and buffer.
        struct Vertex {
            dt::vec3 position;
            dt::vec2 texcoord;
        };
        static const Vertex vertexData[] = {
            { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0, 0.0) },
            { dt::vec3(-1.0, +1.0, 0.0), dt::vec2(0.0, 1.0) },
            { dt::vec3(+1.0, -1.0, 0.0), dt::vec2(1.0, 0.0) },
            { dt::vec3(+1.0, +1.0, 0.0), dt::vec2(1.0, 1.0) },
        };

        VertexStream stream(sizeof(Vertex));
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, texcoord);
        VertexArrayState vertex = stream.CreateVertexArrayState();
        Buffer *vbo = stream.AllocateVertexBuffer(device, 4, coherent_allocator, vertexData);
        BufferAddress vboAddr = vbo->GetAddress();
        queueCB.BindVertexArrayState(vertex);
        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

        for (int column = 0; column < 2; ++column) {
            for (int row = 0; row < 5; ++row) {
                switch (m_testType) {
                case TEST_TEXTURES:
                    drawRow(device, queueCB, textureHandles, row, column);
                    break;
                case TEST_IMAGES:
                    drawRow(device, queueCB, imageHandles, row, column);
                    break;
                case TEST_UBOS:
                    drawRow(device, queueCB, uboAddrs, row, column);
                    break;
                case TEST_SSBOS:
                    drawRow(device, queueCB, ssboAddrs, row, column);
                    break;
                default: assert(0); break;
                }
            }
        }
    } else {
        // Set up the vertex format and buffer.
        struct Vertex {
            dt::vec4 positionColor;
        };

        // Create 16 vertex buffers, each with identical vertices but
        // different colors
        VertexStreamSet streamSet;
        VertexStream streams[numVertexBuffers];
        Buffer *vbo[numVertexBuffers];
        uint64_t vboAddrs[numVertexBuffers];
        for (int b = 0; b < numVertexBuffers; b++) {
            dt::vec4 vertexData[numVertexBuffers * numVerticesPerQuad];
            for (int i = 0; i < numVertexBuffers; i++) {
                float x0 = i / float(numVertexBuffers)*2.0f - 1.0f;
                float x1 = (i + 1) / float(numVertexBuffers)*2.0f - 1.0f;
                float c = (b + 1) / float(numVertexBuffers);
                vertexData[i * numVerticesPerQuad + 0] = dt::vec4(x0, -1.0f, 0.0f, c);
                vertexData[i * numVerticesPerQuad + 1] = dt::vec4(x1, -1.0f, 0.0f, c);
                vertexData[i * numVerticesPerQuad + 2] = dt::vec4(x1, +1.0f, 0.0f, c);
                vertexData[i * numVerticesPerQuad + 3] = dt::vec4(x0, -1.0f, 0.0f, c);
                vertexData[i * numVerticesPerQuad + 4] = dt::vec4(x1, +1.0f, 0.0f, c);
                vertexData[i * numVerticesPerQuad + 5] = dt::vec4(x0, +1.0f, 0.0f, c);
            }
            VertexStream stream(sizeof(Vertex));
            LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, positionColor);
            streams[b] = stream;
            streamSet.addStream(streams[b]);
            vbo[b] = stream.AllocateVertexBuffer(device, numVertexBuffers * numVerticesPerQuad, coherent_allocator, vertexData);
            vboAddrs[b] = vbo[b]->GetAddress();
        }
        VertexArrayState vertex = streamSet.CreateVertexArrayState();
        queueCB.BindVertexArrayState(vertex);

        for (int column = 0; column < 2; ++column) {
            for (int row = 0; row < 5; ++row) {
                drawRow(device, queueCB, vboAddrs, row, column);
            }
        }
    }

    // Reset to the system texture pool.
    g_lwnTexIDPool->Bind(queueCB);

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNBindingTest, lwn_texture_binding, (LWNBindingTest::TEST_TEXTURES));
OGTEST_CppTest(LWNBindingTest, lwn_image_binding, (LWNBindingTest::TEST_IMAGES));
OGTEST_CppTest(LWNBindingTest, lwn_ubo_binding, (LWNBindingTest::TEST_UBOS));
OGTEST_CppTest(LWNBindingTest, lwn_ssbo_binding, (LWNBindingTest::TEST_SSBOS));
OGTEST_CppTest(LWNBindingTest, lwn_vertexbuffer_binding, (LWNBindingTest::TEST_VBOS));

class LWNTransformFeedbackBindingTest
{
    static const int numTransformFeedbackBuffers = 4;   //lwn::DeviceInfo::TRANSFORM_FEEDBACK_BUFFER_BINDINGS
    static const int numVerticesPerQuad = 6;

    static const int cellSizeX = 64;
    static const int cellSizeY = 64;
    static const int cellMargin = 16;

public:
    LWNTransformFeedbackBindingTest() {}
    LWNTEST_CppMethods();
};

lwString LWNTransformFeedbackBindingTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test exercising transform feedback buffer binding API.\n\n"
        "There are two columns of test cases: the left column uses BindTransformFeedbackBuffers API, the right one "
        "BindTransformFeedbackBuffer API. The images in each column should match. The test allocates 4 transform feedback buffers, "
        "one for each binding point. In each cell, each constant color column is drawn with "
        "a separate buffer. The buffers are filled with increasing levels of gray. The rows of cells "
        "from bottom to top are: 1. bind all 4 buffers in order 2. reverse order of buffers 2-3 "
        "3. change order of buffers 1-3 4. change order of buffers 0-4 5. change order of "
        "buffers 0-1. The result is that the top row has colors in the opposite order compared "
        "to the bottom row.";
    return sb.str();
}

int LWNTransformFeedbackBindingTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 14);
}

void LWNTransformFeedbackBindingTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    MemoryPoolAllocator gpu_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator coherent_allocator(device, NULL, 1024 * 1024, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Transform feedback shader writes a different constant into each buffer
    VertexShader tfvs(440);
    tfvs <<
        "layout(xfb_buffer = 0, xfb_offset = 0) out float out0;\n"
        "layout(xfb_buffer = 1, xfb_offset = 0) out float out1;\n"
        "layout(xfb_buffer = 2, xfb_offset = 0) out float out2;\n"
        "layout(xfb_buffer = 3, xfb_offset = 0) out float out3;\n"
        "void main() {\n"
        "  out0 = 0.25f;\n"
        "  out1 = 0.5f;\n"
        "  out2 = 0.75f;\n"
        "  out3 = 1.0f;\n"
        "}\n";

    Program *tfpgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(tfpgm, tfvs)) {
        LWNFailTest();
        return;
    }

    const int numVertices = numTransformFeedbackBuffers * numVerticesPerQuad;
    const int vertexDataSize = numVertices*sizeof(float);

    // Allocate the transform feedback buffers
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    BufferAlignBits xfbAlign = BufferAlignBits(BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT |
                                               BUFFER_ALIGN_COPY_READ_BIT |
                                               BUFFER_ALIGN_VERTEX_BIT);
    Buffer *xfbResult[numTransformFeedbackBuffers];
    for (int i = 0; i < numTransformFeedbackBuffers; i++) {
        xfbResult[i] = coherent_allocator.allocBuffer(&bb, xfbAlign, vertexDataSize);
    }
    Buffer *xfbControl = coherent_allocator.allocBuffer(&bb, BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT, 32);
    BufferAddress xfbControlAddr = xfbControl->GetAddress();

    // Visualization shader reads color from 4 transform feedback buffers.
    // Each quad fetches color from a different buffer.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in float colors[4];\n"
        "out float color;\n"
        "const int numVerticesPerQuad = 6;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  color = colors[gl_VertexID / numVerticesPerQuad];\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "in float color;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(color);\n"
        "}\n";

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        LWNFailTest();
        return;
    }

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    // Set up the vertex format and buffer for the visualization shader.
    VertexStreamSet streamSet;
    VertexStream streams[numTransformFeedbackBuffers + 1];
    Buffer *vbo;
    BufferRange xfbBuffers[numTransformFeedbackBuffers];
    dt::vec3 vertexData[numTransformFeedbackBuffers * numVerticesPerQuad];
    for (int i = 0; i < numTransformFeedbackBuffers; i++) {
        float x0 = i / float(numTransformFeedbackBuffers)*2.0f - 1.0f;
        float x1 = (i + 1) / float(numTransformFeedbackBuffers)*2.0f - 1.0f;
        vertexData[i * numVerticesPerQuad + 0] = dt::vec3(x0, -1.0f, 0.0f);
        vertexData[i * numVerticesPerQuad + 1] = dt::vec3(x1, -1.0f, 0.0f);
        vertexData[i * numVerticesPerQuad + 2] = dt::vec3(x1, +1.0f, 0.0f);
        vertexData[i * numVerticesPerQuad + 3] = dt::vec3(x0, -1.0f, 0.0f);
        vertexData[i * numVerticesPerQuad + 4] = dt::vec3(x1, +1.0f, 0.0f);
        vertexData[i * numVerticesPerQuad + 5] = dt::vec3(x0, +1.0f, 0.0f);
    }
    // Create a stream for vertex position data
    {
        struct Vertex {
            dt::vec3 position;
        };
        VertexStream stream(sizeof(Vertex));
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
        streams[0] = stream;
        streamSet.addStream(streams[0]);
        vbo = stream.AllocateVertexBuffer(device, numTransformFeedbackBuffers * numVerticesPerQuad, coherent_allocator, vertexData);
    }
    // Create streams for the transform feedback buffers
    for (int b = 0; b < numTransformFeedbackBuffers; b++) {
        struct VertexColor {
            dt::vec1 color;
        };
        VertexStream stream(sizeof(VertexColor));
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, VertexColor, color);
        streams[b + 1] = stream;
        streamSet.addStream(streams[b + 1]);
        xfbBuffers[b].address = xfbResult[b]->GetAddress();
        xfbBuffers[b].size = numTransformFeedbackBuffers * numVerticesPerQuad * sizeof(dt::vec1);
    }
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    for (int column = 0; column < 2; ++column) {
        for (int row = 0; row < 5; ++row) {

            // Shuffle transform feedback buffers to test the multibind API
            BufferRange buffers[numTransformFeedbackBuffers];
            int order[5][numTransformFeedbackBuffers] = {
                { 0, 1, 2, 3 },
                { 0, 1, 3, 2 },
                { 0, 3, 2, 1 },
                { 3, 2, 0, 1 },
                { 3, 2, 1, 0 },
            };
            for (int i = 0; i < numTransformFeedbackBuffers; ++i) {
                int id = order[row][i];
                buffers[id].address = xfbBuffers[i].address;
                buffers[id].size = xfbBuffers[i].size;
            }
            int first = 0;
            int count = numTransformFeedbackBuffers;
            switch (row) {
            case 0: break;
            case 1: first = 2; count = 2; break;
            case 2: first = 1; count = 3; break;
            case 3: first = 0; count = 4; break;
            case 4: first = 0; count = 2; break;
            default: assert(0); break;
            }

            if (column) {
                for (int i = first; i < first + count; ++i) {
                    queueCB.BindTransformFeedbackBuffer(i, buffers[i].address, buffers[i].size);
                }
            } else {
                queueCB.BindTransformFeedbackBuffers(first, count, buffers + first);
            }

            // Draw with transform feedback
            queueCB.BindProgram(tfpgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.BeginTransformFeedback(xfbControlAddr);
            queueCB.DrawArrays(DrawPrimitive::POINTS, 0, numVertices);
            queueCB.EndTransformFeedback(xfbControlAddr);
            queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES);

            // Visualize
            vertex.bind(queueCB);
            queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB.BindVertexBuffer(0, vbo->GetAddress(), numTransformFeedbackBuffers * numVerticesPerQuad * sizeof(dt::vec3));
            queueCB.BindVertexBuffers(1, numTransformFeedbackBuffers, xfbBuffers);
            queueCB.SetViewport(column*(cellSizeX + cellMargin), row * (cellSizeY + cellMargin), cellSizeX, cellSizeY);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, numTransformFeedbackBuffers * numVerticesPerQuad);
            queueCB.Barrier(BarrierBits::ORDER_PRIMITIVES);
        }
    }
    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNTransformFeedbackBindingTest, lwn_xfb_binding, );
