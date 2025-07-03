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

#define LWN_TEXBIND_LOG_OUTPUT      0

#if LWN_TEXBIND_LOG_OUTPUT >= 2
#define SHOULD_LOG(_result)     true
#define LOG(x)                  printf x
#define LOG_INFO(x)             printf x
#elif LWN_TEXBIND_LOG_OUTPUT >= 1
#define SHOULD_LOG(_result)     _result
#define LOG(x)                  printf x
#define LOG_INFO(x)
#else
#define SHOULD_LOG(_result)     false
#define LOG(x)
#define LOG_INFO(x)
#endif

using namespace lwn;

class LWNTextureBindTest
{
    // The test draws points into a framebuffer object.  The size is fairly
    // irrelevant, but is chosen to fit as 4x4 cells in a 640x480 window in
    // case we cared to display the results.
    static const int fboWidth = 160;
    static const int fboHeight = 120;

    // The test generates 32 texture and sampler objects with random data.
    static const int nTextures = 32;

    // Each texture is a 2x2 RGBA image.
    static const int texelCount = 2 * 2;
    static const int textureBytes = texelCount * sizeof(dt::u8lwec4);
    static const int texFormat = Format::RGBA8;

    // The test generates a number of random variants.

    // We have shaders accessing either a single selected texture unit or an
    // array of units with a run-time index.
    enum ShaderVariant {
        Texture0,               // test only unit 0
        Texture7,               // test only unit 7
        TextureAll,             // test all units in an array
        ShaderVariantCount,
    };

    // Commands are submitted either through queue or API command buffers at
    // random.
    enum CommandVariant {
        UseQueue,
        UseCommandBuffer,
        UseCommandBufferTransient,
        CommandVariantCount,
    };

    // Each state change affects a subset of texture units chosen at random.
    enum ChangeUnits {
        Unit0,
        Unit7,
        Unit0to3,
        UnitRand,
        UnitRand4,
        ChangeUnitCount,
    };

    // Data for each allocated texture and sampler.
    struct TextureInfo {
        Texture        *texture;
        LWNuint        id;
        dt::u8lwec4    texels[4];       // random texel data
        dt::vec4       filtered;        // filtered texel data centered at (0.5, 0.5)
    };
    struct SamplerInfo {
        Sampler        *sampler;
        LWNuint        id;
        dt::vec4       borderColor;     // random border color
    };

    // Current state of a texture unit (index of lwrrently bound texture and
    // sampler).
    struct TexUnitInfo {
        int             texture;
        int             sampler;
    };

    // Each pixel in the FBO is a separate cell, and we keep track of how the
    // pixel should have been rendered.
    struct TestCellInfo {
        ShaderVariant   shader;             // shader to use for rendering
        bool            isTexture;          // should we sample the texture or (sampler) border
        int             item;               // which texture or sampler should we access?
    };

    // Control information for our full test run.
    struct TestControl {
        TestCellInfo    *cells;                     // information on all the cells
        TextureInfo     textures[nTextures];        // textures to use
        SamplerInfo     samplers[nTextures];        // samplers to use
        TexUnitInfo     bindings[nTextures];        // current texture binding state
        Program         *programs[ShaderVariantCount]; // generated programs
        dt::ivec3       *vboMem;                    // vertex buffer holding cell info

        int             lastDrawCount;              // number of previously submitted draws
        int             drawCount;                  // number of queued up draws

        ShaderVariant   lastShaderVariant;          // last shader variant used

        TestControl()
        {
            cells = new TestCellInfo[fboWidth*fboHeight];
            vboMem = NULL;
            lastDrawCount = 0;
            drawCount = 0;
            lastShaderVariant = ShaderVariantCount /* invalid  */;
        }

        ~TestControl()
        {
            delete[] cells;
        }

        // Add a new pair of entries to be rendered from texture unit <unit>,
        // one using the texture and the other using the sampler border color.
        void add(int unit, ShaderVariant shader)
        {
            if (drawCount >= fboWidth * fboHeight) {
                return;
            }
            cells[drawCount].shader = shader;
            cells[drawCount].isTexture = true;
            cells[drawCount].item = unit;
            drawCount++;
            cells[drawCount].shader = shader;
            cells[drawCount].isTexture = false;
            cells[drawCount].item = unit;
            drawCount++;
        }

        void bindTextureSampler(Device *device, CommandBuffer *cmdBuf, LWNuint unit, int texAllocIndex, int smpAllocIndex);
        void setProgram(CommandBuffer *cmdBuf, const TestCellInfo * info, int cell);
        void sendPrimitives(CommandBuffer *cmdBuf, int newDrawCount);
        void drawCells(CommandBuffer *cmdBuf);
    };

public:
    LWNTEST_CppMethods();
};

lwString LWNTextureBindTest::getDescription() const
{
    lwStringBuf sb;
    sb << 
        "Random texture binding test for LWN.  The test generates a collection "
        "of textures with random data and samplers with random border colors, "
        "binds them randomly, and renders points with randomly selected shaders.  "
        "At the end of the test, we check the results of rendering against "
        "expected values and display green if results match exactly or red "
        "otherwise.";
    return sb.str();    
}

int LWNTextureBindTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(8, 2);
}

void LWNTextureBindTest::TestControl::bindTextureSampler(Device *device, CommandBuffer *cmdBuf, LWNuint unit, int texAllocIndex, int smpAllocIndex)
{
    LOG_INFO(("%d:  bind texture/sampler in unit %d to %d / %d\n", drawCount, unit, texAllocIndex, smpAllocIndex));
    TextureHandle handle = device->GetTextureHandle(textures[texAllocIndex].id, samplers[smpAllocIndex].id);
    cmdBuf->BindTexture(ShaderStage::FRAGMENT, unit, handle);
    bindings[unit].texture = texAllocIndex;
    bindings[unit].sampler = smpAllocIndex;
}

void LWNTextureBindTest::TestControl::setProgram(CommandBuffer *cmdBuf, const TestCellInfo * info, int cell)
{
    if (info->shader != lastShaderVariant) {
        sendPrimitives(cmdBuf, cell);
        LOG_INFO(("%d:  switch to program %d\n", lastDrawCount, info->shader));
        cmdBuf->BindProgram(programs[info->shader], ShaderStageBits::ALL_GRAPHICS_BITS);
        lastShaderVariant = info->shader;
    }
}

// Send a batch of primitives to bring the total draw count up to <newDrawCount>.
void LWNTextureBindTest::TestControl::sendPrimitives(CommandBuffer *cmdBuf, int newDrawCount)
{
    if (newDrawCount == lastDrawCount) {
        return;
    }
    LOG_INFO(("%d:  draw %d\n", lastDrawCount, newDrawCount - lastDrawCount));
    cmdBuf->DrawArrays(DrawPrimitive::POINTS, lastDrawCount, newDrawCount - lastDrawCount);
    lastDrawCount = newDrawCount;
}

void LWNTextureBindTest::TestControl::drawCells(CommandBuffer *cmdBuf)
{
    for (int i = lastDrawCount; i < drawCount; i++) {
        TestCellInfo *info = &cells[i];
        setProgram(cmdBuf, info, i);

        // Generate vertex data for the cell with the point number, texture unit
        // to test, and the texture/border color to test.
        vboMem[i] = dt::ivec3(i, info->item, info->isTexture ? 0 : 1);

        // Update the cell data, changing <item> to hold the actual texture or 
        // sampler object number instead of the selected texture unit.
        if (info->isTexture) {
            info->item = bindings[info->item].texture;
        } else {
            info->item = bindings[info->item].sampler;
        }
    }
    sendPrimitives(cmdBuf, drawCount);
}

void LWNTextureBindTest::doGraphics() const
{
    TestControl test;
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    BufferBuilder bufferBuilder;
    TextureBuilder textureBuilder;
    bufferBuilder.SetDevice(device).SetDefaults();
    textureBuilder.SetDevice(device).SetDefaults();

    // allocator will create pool at first allocation
    // make coherent pool same size as texture pool (texture copies during texture allocation)
    const LWNsizeiptr coherent_poolsize = 0x100000UL;
    // safe guess, if the textures don't fit (in a future modification of this test) we'd notice allocation failures
    const LWNsizeiptr tex_poolsize = 0x100000UL;
    MemoryPoolAllocator allocator(device, NULL, coherent_poolsize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator tex_allocator(device, NULL, tex_poolsize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    // Set up a buffer object used for downloading texture data.
    bufferBuilder.SetDefaults();
    Buffer *downloadBuffer = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_READ_BIT, nTextures*textureBytes);
    BufferAddress downloadBufferAddr = downloadBuffer->GetAddress();
    char *downloadBufPtr = (char *) downloadBuffer->Map();
    if (!downloadBufPtr) {
        LWNFailTest();
        return;
    }

    // Generate a collection of RGBA8 textures with random texel data.
    CopyRegion downloadRegion = { 0, 0, 0, 2, 2, 1 };
    for (int i = 0; i < nTextures; i++) {
        TextureInfo *info = &test.textures[i];
        for (int j = 0; j < 4; j++) {
            info->texels[j] = dt::u8lwec4(lwFloatRand(0, 1), lwFloatRand(0, 1),
                                          lwFloatRand(0, 1), lwFloatRand(0, 1));
        }
        info->filtered = (dt::vec4(info->texels[0]) + dt::vec4(info->texels[1]) +
                          dt::vec4(info->texels[2]) + dt::vec4(info->texels[3])) * 0.25;

        textureBuilder.SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).SetFormat(Format::RGBA8).
            SetSize2D(2, 2);
        info->texture = tex_allocator.allocTexture(&textureBuilder);
        info->id = info->texture->GetRegisteredTextureID();
        memcpy(downloadBufPtr + i * textureBytes, info->texels, textureBytes);
        queueCB.CopyBufferToTexture(downloadBufferAddr + i *textureBytes, info->texture, NULL, &downloadRegion, CopyFlags::NONE);
    }

    // Generate a collection of sampler objects with random border colors.
    SamplerBuilder samplerBuilder;
    samplerBuilder.SetDevice(device).SetDefaults();
    samplerBuilder.SetMinMagFilter(MinFilter::LINEAR, MagFilter::LINEAR);
    samplerBuilder.SetWrapMode(WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP_TO_BORDER,
                               WrapMode::CLAMP_TO_BORDER);
    for (int i = 0; i < nTextures; i++) {
        SamplerInfo *info = &test.samplers[i];
        for (int j = 0; j < 4; j++) {
            info->borderColor = dt::vec4(lwFloatRand(0,1), lwFloatRand(0,1),
                                         lwFloatRand(0,1), lwFloatRand(0,1));
        }
        samplerBuilder.SetBorderColor(info->borderColor.ptr());
        info->sampler = samplerBuilder.CreateSampler();
        info->id = info->sampler->GetRegisteredID();
    }

    // Set up a vertex buffer for the test, where we allocate a separate ivec3
    // point for each pixel in the FBO we use for testing.  The input data is
    // laid out as:
    //    data.x = vertex number
    //    data.y = texture unit to reference (with texture array shaders)
    //    data.z = 0 to access the middle of the texture, !=0 to access the border color
    struct Vertex {
        dt::ivec3 data;
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, data);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, fboWidth * fboHeight, allocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();
    test.vboMem = (dt::ivec3 *) vbo->Map();

    // Bind vertex state
    queueCB.BindVertexArrayState(vertex);

    // Set up programs.
    bool programFailed = false;
    VertexShader vs(440);
    vs <<
        "layout(location=0) in ivec3 data;\n"
        "const float xscale = 2.0 / " << fboWidth << ";\n"
        "const float xbias = -1.0 + 0.5 / " << fboWidth << ";\n"
        "const float yscale = 2.0 / " << fboHeight << ";\n"
        "const float ybias = -1.0 + 0.5 / " << fboHeight << ";\n"
        "flat out int which;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  float x = float(data.x % " << fboWidth << ") + 0.5;\n"
        "  float y = float(data.x / " << fboWidth << ") + 0.5;\n"
        "  gl_Position.x = 2.0 * x / " << fboWidth  << ".0 - 1.0;\n"
        "  gl_Position.y = 2.0 * y / " << fboHeight << ".0 - 1.0;\n"
        "  gl_Position.zw = vec2(0.0, 1.0);\n"
        "  which = data.y;\n"
        "  tc = vec2((data.z == 0) ? 0.5 : 3.0, 0.5);\n"
        "}\n";

    for (int i = 0; i < ShaderVariantCount; i++) {
        FragmentShader fs(440);
        fs <<
            "flat in int which;\n"
            "in vec2 tc;\n"
            "out vec4 color;\n";

        const char *texaccess = "tex";
        switch (i) {
        case Texture0:
            fs << "layout(binding=0) uniform sampler2D tex;\n";
            break;
        case Texture7:
            fs << "layout(binding=7) uniform sampler2D tex;\n";
            break;
        case TextureAll:
            fs << "layout(binding=0) uniform sampler2D tex["<<nTextures<<"];\n";
            texaccess = "tex[which]";
            break;
        default:
            assert(0);
        }
        fs <<
            "void main() {\n"
            "  color = texture(" << texaccess << ", tc);\n"
            "}\n";
        test.programs[i] = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(test.programs[i], vs, fs)) {
            programFailed = true;
            continue;
        }
    }
    if (programFailed) {
        LWNFailTest();
        return;
    }

    // Set up a framebuffer for temporary results.
    Framebuffer fbo(fboWidth, fboHeight);
    fbo.setColorFormat(0, Format::RGBA8);
    fbo.alloc(device);
    fbo.bind(queueCB);
    fbo.setViewportScissor();
    queueCB.ClearColor();
    queueCB.BindVertexBuffer(0, vboAddr, fboWidth * fboHeight * sizeof(Vertex));

    LOG_INFO(("\n"));

    // Initialize the state by binding textures and samplers as pairs in the
    // order initially generated and testing each pair.
    for (int i = 0; i < nTextures; i++) {
        test.bindTextureSampler(device, queueCB, i, i, i);
        test.add(i, TextureAll);
    }
    test.drawCells(queueCB);

    CommandBuffer *apiCmdBuf = device->CreateCommandBuffer();
    g_lwnCommandMem.populateCommandBuffer(apiCmdBuf, CommandBufferMemoryManager::Coherent);

    // Then keep running loops with random texture and sampler bindings,
    // rendering cells as we go.
    for (int loop = 0; loop < 1000; loop++) {

        LOG_INFO(("%d: starting pass %d\n", test.drawCount, loop));

        // Every few runs, send some command buffers that only update state
        // without drawing.  Those state changes should affect subsequent draws.
        bool skipDraws = (lwFloatRand(0,1) < 0.15);
        if (skipDraws) {
            LOG_INFO(("%d: skipping draws for pass %d\n", test.drawCount, loop));
        }

        // Decide what method we should use for sending new binds and/or draws.
        CommandVariant commandVariant = CommandVariant(lwRand() % CommandVariantCount);
        CommandBuffer *cmdBuf;
        if (commandVariant == UseCommandBuffer || commandVariant == UseCommandBufferTransient) {
            LOG_INFO(("%d: starting command buffer\n", test.drawCount));
            cmdBuf = apiCmdBuf;
            cmdBuf->BeginRecording();
        } else {
            cmdBuf = queueCB;
        }

        // Perform a random number of state changes.
        int nChanges = lwRand() % 6 + 1;
        for (int change = 0; change < nChanges; change++) {

            // Most of the time we use a program that indexes into an array of
            // textures, but every now and then, we use a variant that folwses
            // on a single unit.
            ShaderVariant shaderVariant = TextureAll;
            
            // Decide on a set of units to update.
            int nUnits = 0;
            int units[4];
            ChangeUnits changeUnits = ChangeUnits(lwRand() % ChangeUnitCount);
            switch (changeUnits) {
            case Unit0:
                nUnits = 1;
                units[0] = 0;
                if (lwRand() & 1) shaderVariant = Texture0;
                break;
            case Unit7:
                nUnits = 1;
                units[0] = 7;
                if (lwRand() & 1) shaderVariant = Texture7;
                break;
            case Unit0to3:
                nUnits = 4;
                for (int i = 0; i < nUnits; i++) {
                    units[i] = i;
                }
                break;
            case UnitRand:
                nUnits = 1;
                units[0] = lwRand() % nTextures;
                break;
            case UnitRand4:
                nUnits = 4;
                for (int i = 0; i < nUnits; i++) {
                    units[i] = lwRand() % nTextures;
                }
                break;
            default:
                nUnits = 0;
                assert(0);
            }

            // For each unit we update, we modify the texture and sampler
            // bindings at random.
            for (int i = 0; i < nUnits; i++) {
                int unit = units[i];
                int texnum = lwRand() % nTextures;
                int smpnum = lwRand() % nTextures;
                test.bindTextureSampler(device, cmdBuf, unit, texnum, smpnum);
                break;
                test.add(unit, shaderVariant);
            }

            if (!skipDraws) {
                test.drawCells(cmdBuf);
            }
        }

        // If we're using an API command buffer, finish it off and submit it.
        if (commandVariant == UseCommandBuffer || commandVariant == UseCommandBufferTransient) {
            queueCB.submit();       // send anything in the queue command buffer first
            CommandHandle handle = cmdBuf->EndRecording();
            queue->SubmitCommands(1, &handle);
            LOG_INFO(("%d: ending command buffer\n", test.drawCount));
        }
        
        LOG_INFO(("%d: ending pass %d\n", test.drawCount, loop));
    }

    // At the end, do a rendering pass exercising all units.
    for (int i = 0; i < nTextures; i++) {
        test.add(i, TextureAll);
    }
    test.drawCells(queueCB);
    queueCB.submit();

    // Clean up the API command buffer used for test loops.
    apiCmdBuf->Free();

    LOG_INFO(("%d points drawn out of %d\n", test.drawCount, fboWidth * fboHeight));

    // Allocate a buffer used to read back the contents of the framebuffer.
    bufferBuilder.SetDefaults();
    Buffer *readbackBuffer = allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_COPY_WRITE_BIT,
                                                   fboWidth * fboHeight * sizeof(dt::u8lwec4));
    dt::u8lwec4 *readback = (dt::u8lwec4 *) readbackBuffer->Map();
    Texture *fbtex = fbo.getColorTexture(0);
    queue->Finish();
    CopyRegion copyRegion = { 0, 0, 0, fboWidth, fboHeight, 1 };
    queueCB.CopyTextureToBuffer(fbtex, NULL, &copyRegion, readbackBuffer->GetAddress(), CopyFlags::NONE);
    queueCB.submit();
    queue->Finish();

    // Loop over all the pixels that should be touched by our test and compare
    // their RGBA values to expected values from the appropriate texture or
    // sampler object.
    bool passed = true;
    queueCB.ClearColor();
    for (int i = 0; i < test.drawCount; i++) {
        TestCellInfo &info = test.cells[i];
        dt::vec4 value = readback[i];
        dt::vec4 expected;
        if (info.isTexture) {
            expected = test.textures[info.item].filtered;
        } else {
            expected = test.samplers[info.item].borderColor;
        }
        bool comparison = all(abs(value - expected) < 1.0 / 255);
        if (SHOULD_LOG(!comparison)) {
            LOG(("\n%d %s:  expected %s %d\n", i, comparison ? "PASS" : "FAIL",
                   info.isTexture ? "texture" : "sampler", info.item));
            LOG(("  got:  %10.8f %10.8f %10.8f %10.8f\n", value[0], value[1], value[2], value[3]));
            LOG(("  exp:  %10.8f %10.8f %10.8f %10.8f\n", expected[0], expected[1], expected[2], expected[3]));
        }
        if (!comparison) {
            passed = false;
        }
    }

    // Clear the window to red or green based on the result of the comparisons
    // above.
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    if (passed) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }

    queueCB.submit();

    queue->Finish();
    fbo.destroy();
}

OGTEST_CppTest(LWNTextureBindTest, lwn_texbind_random, );
