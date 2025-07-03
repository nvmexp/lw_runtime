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

#define UBO_BINDING 1

using namespace lwn;

class LWNGlslDerivativeControl
{
public:
    LWNTEST_CppMethods();
};

lwString LWNGlslDerivativeControl::getDescription() const
{
    lwStringBuf sb;
    sb << "Testing coverage for ARB_derivative_control GLSL extension. "
          "A shape is printed and colored based on position derivative "
          "values. This is repeated once on each quarter of the screen. "
          "Expected output is four gray and red gradients.\n"
          "* Method 0 - Bottom left  - dFdxFine & dFdyFine\n"
          "* Method 1 - Bottom right - dFdxCoarse & dFdyCoarse\n"
          "* Method 2 - Top left     - fwidthFine\n"
          "* Method 3 - Top right    - fwidthCoarse\n";
    return sb.str();
}

int LWNGlslDerivativeControl::isSupported() const
{
    return lwogCheckLWNAPIVersion(31, 1);
}

void LWNGlslDerivativeControl::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer *queueCB = &deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    queueCB->ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec4 position;\n"
        "layout(binding=" << UBO_BINDING << ", std140) uniform ubo {\n"
        // calcMethod specifies the method used to callwlate the derivative for a pixel,
        // which is used to determine color in the fragment shader.
        "    int calcMethod;\n"
        "};\n"
        "out vec4 pos;\n"
        "flat out int method;\n"
        "void main () {\n"
        "    gl_Position = position;\n"
        "    pos = position;\n"
        "    method = calcMethod;\n"
        "};\n";

    // Fragment shader using coarse and fine derivative callwaltion
    FragmentShader fs(440);
    fs.addExtension(lwShaderExtension::ARB_derivative_control);
    fs <<
        "in vec4 pos;\n"
        "out vec4 fcolor;\n"
        "flat in int method;\n"
        // Gradient color array
        "vec3 colors[20] = vec3[20](\n"
        "    vec3(0.1, 0.1, 0.1),\n"
        "    vec3(0.2, 0.2, 0.2),\n"
        "    vec3(0.3, 0.3, 0.3),\n"
        "    vec3(0.4, 0.4, 0.4),\n"
        "    vec3(0.5, 0.5, 0.5),\n"
        "    vec3(0.6, 0.6, 0.6),\n"
        "    vec3(0.7, 0.7, 0.7),\n"
        "    vec3(0.8, 0.8, 0.8),\n"
        "    vec3(0.9, 0.9, 0.9),\n"
        "    vec3(1.0, 1.0, 1.0),\n"
        "    vec3(1.0, 0.0, 0.0),\n"
        "    vec3(0.9, 0.0, 0.0),\n"
        "    vec3(0.8, 0.0, 0.0),\n"
        "    vec3(0.7, 0.0, 0.0),\n"
        "    vec3(0.6, 0.0, 0.0),\n"
        "    vec3(0.5, 0.0, 0.0),\n"
        "    vec3(0.4, 0.0, 0.0),\n"
        "    vec3(0.3, 0.0, 0.0),\n"
        "    vec3(0.2, 0.0, 0.0),\n"
        "    vec3(0.1, 0.0, 0.0)\n"
        ");\n"
        "int level;\n"
        // Returns an integer representing a derivative value range.
        "int getIndex (float derivative) { return int(log2(pow(64.0 * derivative, 6))); }"
        "void main() {\n"
        // Pixel color is chosen using an integer value callwlated using the position derivative.
        // If the derivative value changs enough, the color will change to the next value in the
        // gradient. This clearly shows the difference between derivative callwlation resolutions.
        "    if (method == 0) {\n"
        "        level = getIndex(max(length(dFdxFine(pos)), length(dFdyFine(pos))));\n"
        "    } else if (method == 1) {\n"
        "        level = getIndex(max(length(dFdxCoarse(pos)), length(dFdyCoarse(pos))));\n"
        "    } else if (method == 2) {\n"
        "        level = getIndex(length(fwidthFine(pos)));\n"
        "    } else if (method == 3) {\n"
        "        level = getIndex(length(fwidthCoarse(pos)));\n"
        "    }\n"        "    if(level < 0 || level > 19) {\n"
        "        fcolor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "    } else {\n"
        "        fcolor = vec4(colors[level], 1.0);\n"
        "    }\n"
        "};\n";

    Program *pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        queueCB->ClearColor(0, 1.0, 0.0, 0.0, 0.0);
        LWNFailTest();
        return;
    }

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };

    // Pre translated and rotated shape.
    static const Vertex vertexData[] = {
        { dt::vec4(-13.677, -16.869, -12.586, -9.623) },
        { dt::vec4( 23.179,   0.432,  -6.835, -4.024) },
        { dt::vec4(-23.179,  11.568,  26.078, 28.024) },
        { dt::vec4( 13.677,  28.869,  31.829, 33.623) },
    };

    // Allocator will create pool at first allocation
    MemoryPoolAllocator vboAllocator(device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 4, vboAllocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB->BindVertexArrayState(vertex);
    queueCB->BindVertexBuffer(0, vboAddr, sizeof(vertexData));

    // UBO to select derivative callwlation resolution.
    struct CalcMethodSelector {
        int method;
    };

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    MemoryPoolAllocator allocator(device, NULL, sizeof(CalcMethodSelector), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = allocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, sizeof(CalcMethodSelector));
    char * uboMem = (char *) ubo->Map();
    BufferAddress uboAddr = ubo->GetAddress();
    CalcMethodSelector * selector = (CalcMethodSelector *) uboMem;
    queueCB->BindUniformBuffer(ShaderStage::VERTEX, UBO_BINDING, uboAddr, sizeof(CalcMethodSelector));
    queueCB->BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // Method 0 - Bottom left  - dFdxFine & dFdyFine
    selector->method = 0;
    queueCB->SetViewportScissor(0, 0, lwrrentWindowWidth / 2.0, lwrrentWindowHeight);
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    queueCB->submit();
    queue->Finish();

    // Method 1 - Bottom right - dFdxCoarse & dFdyCoarse
    selector->method = 1;
    queueCB->SetViewportScissor(lwrrentWindowWidth / 2.0, 0, lwrrentWindowWidth / 2.0, lwrrentWindowHeight);
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    queueCB->submit();
    queue->Finish();

    // Method 2 - Top left     - fwidthFine
    selector->method = 2;
    queueCB->SetViewportScissor(0, lwrrentWindowHeight / 2.0, lwrrentWindowWidth / 2.0, lwrrentWindowHeight);
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    queueCB->submit();
    queue->Finish();

    // Method 3 - Top right    - fwidthCoarse
    selector->method = 3;
    queueCB->SetViewportScissor(lwrrentWindowWidth / 2.0, lwrrentWindowHeight / 2.0, lwrrentWindowWidth / 2.0, lwrrentWindowHeight);
    queueCB->DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

    // We need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory.
    queueCB->submit();
    queue->Finish();
}

OGTEST_CppTest(LWNGlslDerivativeControl, lwn_glsl_derivative_control, );
