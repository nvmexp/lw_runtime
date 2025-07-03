/*
 * Copyright (c) 2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNForceSampleShading
{
public:
    static const uint32_t texWidth  = 64;
    static const uint32_t texHeight = 64;
    static const uint32_t cellWidth  = 64;
    static const uint32_t cellHeight = 64;

    bool checkBuffer(uint32_t *buffer, uint32_t samples) const;

    LWNTEST_CppMethods();
};

lwString LWNForceSampleShading::getDescription() const
{
    lwStringBuf sb;
    sb << "Basic test to verify the lwnProgramSetSampleShading function. "
          "The test creates programs with fragment shaders that would either be "
          "evaluated per sample or per fragment. For programs that should be evaluated "
          "per sample, the shader is forced to be evaluated per fragment and vice versa. "
          "If the number of times that the shader is called per fragment matches the "
          "number of samples respectively one, a green quad is drawn.\n";

    return sb.str();
}

int LWNForceSampleShading::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 310);
}

bool LWNForceSampleShading::checkBuffer(uint32_t *buffer, uint32_t samples) const
{
    assert(buffer);

    uint32_t idx = 0;

    for (uint32_t y = 0; y < texHeight; ++y) {
        for (uint32_t x = 0; x < texHeight; ++x) {
            idx = y * texWidth + x;
            if (buffer[idx] != samples) {
                return false;
            }
        }
    }

    return true;
}

void LWNForceSampleShading::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    struct Vertex {
        dt::vec3 position;
    };

    // Define a triangle that covers the entire viepowrt
    static const Vertex vertexData[] = {
        { dt::vec3(-1.0, -1.0, 0.0) },
        { dt::vec3( 3.0, -1.0, 0.0) },
        { dt::vec3(-1.0,  3.0, 0.0) },
    };

    const uint32_t numVertices = __GL_ARRAYSIZE(vertexData);
    const uint32_t vboSize = numVertices * sizeof(vertexData);
    const uint32_t ssboSize = texWidth * texHeight * sizeof(int);

    MemoryPoolAllocator allocator(device, NULL, vboSize + ssboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, numVertices, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Create SSBO that will store the number of times the shader get called
    // per fragment.
    BufferBuilder bb;
    bb.SetDefaults().SetDevice(device);

    Buffer *ssbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();
    uint32_t *ssboPtr = (uint32_t*)ssbo->Map();

    int cellX = 0;
    int cellY = 0;

    queueCB.ClearColor(0, 0.0, 0.0, 0.4, 0.0);

    struct TestConfig
    {
        SampleShadingMode mode;
        bool perSampleShader;
    };

    std::vector<TestConfig> testModes =
        { { SampleShadingMode::FORCE_SAMPLE_SHADING_ON,  false }, // Force a per-fragment shader to be evaluated per-sample
          { SampleShadingMode::FORCE_SAMPLE_SHADING_OFF, true  }, // Force a per-sample shader to be evaluated per-fragment
          { SampleShadingMode::DEFAULT_FRAGMENT,         true  }, // Default, make sure a per-sample shader is evaluated per-sample.
          { SampleShadingMode::DEFAULT_FRAGMENT,         false }  // Default, make sure a per-fragment shader is evaluated per-fragment.
        };

    for (std::vector<TestConfig>::iterator itr = testModes.begin(); itr != testModes.end(); ++itr) {

        VertexShader vs(450);
        vs <<
            "layout(location=0) in vec3 position;\n"
            "out vec3 ocolor;\n"
            "void main() {\n"
            "    gl_Position = vec4(position, 1.0);\n"
            "}\n";

        FragmentShader fs(450);
        fs <<
            "in vec3 ocolor;\n"
            "out vec4 fcolor;\n"
            "layout(binding = 0) buffer sampleCount\n"
            "{\n"
            "    uint samples[" << (texWidth * texHeight) << "];\n"
            "};\n"
            "const uint width = " << texWidth << ";\n"
            "void main() {\n"
            "    uint idx = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);\n"
            "    atomicAdd(samples[idx], 1);\n";
        if (itr->perSampleShader) {
            // Just add gl_SampleID to the shader to make sure it will request to be
            // exelwted per sample.
            fs <<
                "    if (gl_SampleID > 1) {\n"
                "        fcolor = vec4(0.0f, 1.0f, 0.0f, 1.0);\n"
                "    } else {\n"
                "        fcolor = vec4(0.0f, 0.0f, 1.0f, 1.0);\n"
                "    }\n";
        } else {
            fs <<
                "    fcolor = vec4(0.0f, 1.0f, 0.0f, 1.0);\n";
        }
        fs <<
            "}\n";

        Program *pgm = device->CreateProgram();

        // Override sample shading mode
        pgm->SetSampleShading(itr->mode);

        // Compile and call lwnProgramSetShaders.
        if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
            LWNFailTest();
            return;
        }

        const uint32_t samples[] = { 1, 2, 4, 8 };

        MultisampleState msState;
        msState.SetDefaults();

        queueCB.BindStorageBuffer(ShaderStage::FRAGMENT, 0, ssboAddr, ssboSize);

        queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindVertexArrayState(vertex);
        queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));

        for (uint32_t i = 0; i < __GL_ARRAYSIZE(samples); ++i) {
            memset(ssboPtr, 0, texWidth * texHeight * sizeof(uint32_t));

            Framebuffer msFb(texWidth, texHeight);
            msFb.setFlags(TextureFlags::COMPRESSIBLE);
            msFb.setColorFormat(0, Format::RGBA8);

            if (samples[i] > 1) {
                msFb.setSamples(samples[i]);
                msState.SetSamples(samples[i]);
            }
            msFb.alloc(device);

            msFb.bind(queueCB);
            queueCB.SetViewportScissor(0, 0, texWidth, texHeight);
            queueCB.BindMultisampleState(&msState);
            queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, numVertices);

            if (samples[i] > 1) {
                msFb.downsample(queueCB);
            }

            queueCB.submit();
            queue->Finish();

            msFb.destroy();

            g_lwnWindowFramebuffer.bind();
            queueCB.SetViewportScissor(cellX + 1, cellY + 1, cellWidth - 1, cellHeight - 1);

            msState.SetDefaults();
            queueCB.BindMultisampleState(&msState);

            uint32_t expectedSamples = 1;

            if ((itr->mode == SampleShadingMode::FORCE_SAMPLE_SHADING_ON) ||
               ((itr->mode == SampleShadingMode::DEFAULT_FRAGMENT) && itr->perSampleShader)) {
                expectedSamples = samples[i];
            }

            if (checkBuffer(ssboPtr, expectedSamples)) {
                queueCB.ClearColor(0, 0.0, 1.0, 0.0, 0.0);
            } else {
                queueCB.ClearColor(0, 1.0, 0.0, 0.0, 0.0);
            }

            queueCB.submit();
            queue->Finish();

            cellX += cellWidth;
            if (cellX >= lwrrentWindowWidth) {
                cellX = 0;
                cellY += cellHeight;
            }
        } // for (uint32_t i = 0

        pgm->Free();
    } // for (std::vector<TestConfig>::iterator itr
}

OGTEST_CppTest(LWNForceSampleShading, lwn_force_sample_shading, );