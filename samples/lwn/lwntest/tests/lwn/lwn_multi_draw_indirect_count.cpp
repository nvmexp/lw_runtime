/*
* Copyright(c) 2016 LWPU Corporation.All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define _USE_MATH_DEFINES 1
#include <math.h>


using namespace lwn;

class LWNMultiDrawIndirectCount
{
public:

    LWNTEST_CppMethods();

    void createCircle(LWNuint numTriangles, dt::vec3* v, uint16_t* i) const;
};


lwString LWNMultiDrawIndirectCount::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test to verify MultiDrawElementsCount and MultiDrawArraysCount. The test uses these\n"
        "functions to draw a circle or parts of a circle. The draw count parameter is stored\n"
        "in a buffer object. The color used for each of the draws is selected based on the\n"
        "gl_DrawID. The following tests are exelwted.\n"
        "1: Draw a circle with 8 draws.\n"
        "2: Draw half of a circle using 4 draws.\n"
        "3: Make sure the value in the parameter buffer will not exceed the maximum draw count.\n"
        "4: Draw circle using instancing and making sure the draw parameters gl_BaseVertex and\n"
        "   gl_BaseInstanceARB are passed correctly to the shader.\n"
        "Each test draws two circles, one by calling MultiDrawElementsCount and the other by calling\n"
        "MultiDrawArraysCount. The circles are segmented based on the number of draws that are used.\n"
        "Each segment is drawn using its own color. Tests 1 and 2 are rendered on the bottom half\n"
        "of the windows; tests 3 and 4 in the top half.";

    return sb.str();
}

int LWNMultiDrawIndirectCount::isSupported() const
{
    return lwogCheckLWNAPIVersion(45, 1);
}


void LWNMultiDrawIndirectCount::createCircle(LWNuint numTriangles, dt::vec3* const vec, uint16_t* const idx) const
{
    assert(numTriangles >= 4);

    const float a = (2.0f * M_PI) / numTriangles;

    uint16_t* iptr = idx;
    dt::vec3* vptr = vec;

    for (LWNuint i = 0; i < numTriangles; ++i) {

        *(vptr + 0) = dt::vec3(0.0f, 0.0f, 0.0f);
        *(vptr + 1) = dt::vec3(sinf((i + 1)*a), cosf((i + 1)*a), 0.0f);
        *(vptr + 2) = dt::vec3(sinf(i*a), cosf(i*a), 0.0f);

        *(iptr + 0) = 0;
        *(iptr + 1) = (i*3) + 1;
        *(iptr + 2) = (i*3) + 2;

        vptr += 3;
        iptr += 3;
    }
    *(iptr - 2) = 2;
}

void LWNMultiDrawIndirectCount::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    struct IndirectCmdData {
        DrawElementsIndirectData    element;
        DrawArraysIndirectData      array;
    };

    static const LWNuint numRows = 2;
    static const LWNuint numCols = 4;
    static const LWNuint drawCount = 8;
    static const LWNuint numTriangles = 64;
    static const LWNuint numVertices = numTriangles * 3;
    static const LWNuint vboSize = numVertices * sizeof(dt::vec3);
    static const LWNuint iboSize = numVertices * sizeof(uint16_t);
    static const LWNuint indSize = (drawCount + 2) * sizeof(IndirectCmdData);
    static const LWNuint uboSize = drawCount * sizeof(dt::vec4) + sizeof(dt::vec4);
    // Increase pool size by 512 bytes for additional space requirements like alignment
    static const LWNuint poolSize = vboSize + iboSize + indSize + uboSize + 512;

    LWNint indirectAlignment;
    device->GetInteger(DeviceInfo::INDIRECT_DRAW_ALIGNMENT, &indirectAlignment);

    assert(numTriangles > drawCount);
    assert(sizeof(DrawElementsIndirectData) % indirectAlignment == 0);

    VertexShader vs(440);
    vs <<
        "#extension GL_ARB_shader_draw_parameters:enable\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(binding = 0) uniform Block {\n"
        "    vec4 scale;\n"
        "    vec4 color[" << drawCount << "];\n"
        "};\n"
        "out vec4 col;\n"
        "void main() {\n"
        "  float alpha = 1.0f - (float(gl_BaseVertexARB) / float(" << numVertices << "));\n"
        "  gl_Position = vec4(position, 1.0f) * scale;\n"
        "  gl_Position.y *= (1.0f - 2.0 * float(gl_InstanceID));\n"
        "  int idx = gl_DrawIDARB + gl_BaseInstanceARB;\n"
        "  col = vec4(color[idx].r, color[idx].g, color[idx].b, alpha);\n"
        "}\n";

    FragmentShader fs(440);
    fs <<
        "in vec4 col;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = col;\n"
        "}\n";

    Program *program = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(program, vs, fs)) {
        printf("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog());

        LWNFailTest();
    }

    const int cellWidth = lwrrentWindowWidth / numCols;
    const int cellHeight = lwrrentWindowHeight / numRows;

    MemoryPoolAllocator bufferAllocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream vertexStream(sizeof(dt::vec3));
    vertexStream.addAttribute<dt::vec3>(0);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, numVertices, bufferAllocator, NULL);
    BufferAddress vboAddr = vbo->GetAddress();

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    // Allocate UBO to store the scale vactor and the color array
    Buffer *ubo = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress uboAddr = ubo->GetAddress();

    dt::vec4* scale = (dt::vec4*)ubo->Map(); // Scale is the first vec4 in the ubo
    dt::vec4* colorArray = scale + 1;       // The color array starts with the first vec4 after scale

    const float ar = (float)cellWidth / (float)cellHeight;

    *scale = (ar >= 1.0f) ? dt::vec4(1.0f / ar *0.5, .5f, .5f, 1.0f) : dt::vec4(0.5f, ar *0.5, 0.5f, 0.5f);

    // Allocate index buffer
    Buffer *ibo = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_INDEX_BIT, iboSize);
    BufferAddress iboAddr = ibo->GetAddress();

    // Allocate indirect command buffer
    Buffer *indBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_INDIRECT_BIT, indSize);
    BufferAddress indBufferAddr = indBuffer->GetAddress();

    // Allocate parameter buffer to store draw counts
    static const LWNuint numDrawCntParams = 4;
    const LWNuint drawCountArray[numDrawCntParams] = { drawCount, drawCount / 2, 2 * drawCount, 2 };

    Buffer *paramBuffer = bufferAllocator.allocBuffer(&bb, BUFFER_ALIGN_INDIRECT_BIT, numDrawCntParams * sizeof(LWNuint));
    BufferAddress paramBufferAddr = paramBuffer->GetAddress();

    LWNuint* draws = (LWNuint*)paramBuffer->Map();
    memcpy(draws, drawCountArray, numDrawCntParams * sizeof(LWNuint));

    dt::vec3* vertices = (dt::vec3*)vbo->Map();
    uint16_t* indices = (uint16_t*)ibo->Map();

    createCircle(numTriangles, vertices, indices);

    IndirectCmdData* indData = (IndirectCmdData*)indBuffer->Map();

    // Generate indirect draw commands and color information. Each draw command will draw a
    // portion of a circle. If all draw commands get exelwted the full circle is drawn. The
    // number of commands that get exelwted depends on the draw count that is stored in the
    // parameter buffer. The color used for each draw is selected by the shader out of the
    // color array using the gl_DrawID.
    for (LWNuint i = 0; i < drawCount; ++i) {
        indData[i].element.count = numVertices / drawCount;
        indData[i].element.instanceCount = 1;
        indData[i].element.firstIndex = (numVertices / drawCount) * i;
        indData[i].element.baseVertex = 0;
        indData[i].element.baseInstance = 0;

        indData[i].array.count = numVertices / drawCount;
        indData[i].array.instanceCount = 1;
        indData[i].array.first = (numVertices / drawCount) * i;
        indData[i].array.baseInstance = 0;

        colorArray[i] = dt::vec4(1.0f - (float)i / (float)drawCount, 1.0f - (float)i / (float)drawCount, 1.0f, 1.0f);
    }

    // Generate draw commands to draw a circle using instancing. Each draw command will draw 2 instances
    // of a quarter of the circle. The shader will mirror one of the quarters based on the gl_instanceID.
    // The color is selected by the shader based on gl_DrawID and gl_BaseInstanceID. In addition an alpha
    // value is generated for the MultiDrawElementsIndirectCount which is based on the gl_BaseVertexID.
    indData[drawCount].element.count = numVertices / 4;
    indData[drawCount].element.instanceCount = 2;
    indData[drawCount].element.firstIndex = 0;
    indData[drawCount].element.baseVertex = numVertices / 4;
    indData[drawCount].element.baseInstance = 3;

    indData[drawCount + 1].element.count = numVertices / 4;
    indData[drawCount + 1].element.instanceCount = 2;
    indData[drawCount + 1].element.firstIndex = 0;
    indData[drawCount + 1].element.baseVertex = numVertices / 2;
    indData[drawCount + 1].element.baseInstance = 4;

    indData[drawCount].array.count = numVertices / 4;
    indData[drawCount].array.instanceCount = 2;
    indData[drawCount].array.first = numVertices / 4;
    indData[drawCount].array.baseInstance = 3;

    indData[drawCount + 1].array.count = numVertices / 4;
    indData[drawCount + 1].array.instanceCount = 2;
    indData[drawCount + 1].array.first = numVertices / 2;
    indData[drawCount + 1].array.baseInstance = 4;

    struct TestParam {
        LWNuint     indBufferOffset;
        LWNuint     drawCountOffset;
        LWNuint     maxDrawCount;
    };

    TestParam tests[] = {
        { 0, 0, drawCount },                                            // Draw whole circle
        { 0, sizeof(LWNuint), drawCount },                              // Draw half of a circle
        { 0, 2 * sizeof(LWNuint), drawCount },                          // Make sure to fallback to use maxDrawCount if draw buffer value is too big
        { drawCount*sizeof(IndirectCmdData), 3 * sizeof(LWNuint), 2 }   // Draw circle with 2 colors using instancing
    };

    BlendState bs;
    bs.SetDefaults();
    bs.SetBlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA, BlendFunc::ONE, BlendFunc::ONE);
    bs.SetBlendEquation(BlendEquation::ADD, BlendEquation::ADD);

    ColorState cs;
    cs.SetDefaults();
    cs.SetBlendEnable(0, true);

    queueCB.BindBlendState(&bs);
    queueCB.BindColorState(&cs);

    queueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);

    queueCB.ClearColor(0, 0.0f, 0.0f, 0.2f, 1.0f);

    queueCB.BindProgram(program, ShaderStageBits::ALL_GRAPHICS_BITS);

    queueCB.BindUniformBuffer(ShaderStage::VERTEX, 0, uboAddr, uboSize);

    queueCB.BindVertexArrayState(vertexState);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);

    LWNint cellX = 0;
    LWNint cellY = 0;

    for (LWNuint i = 0; i < sizeof(tests) / sizeof(TestParam); ++i) {
        queueCB.SetViewport(cellX, cellY, cellWidth, cellHeight);
        queueCB.SetScissor(cellX, cellY, cellWidth, cellHeight);

        queueCB.MultiDrawElementsIndirectCount(DrawPrimitive::TRIANGLES, IndexType::UNSIGNED_SHORT, iboAddr, indBufferAddr + tests[i].indBufferOffset,
            paramBufferAddr + tests[i].drawCountOffset, tests[i].maxDrawCount, sizeof(IndirectCmdData));

        cellX += cellWidth;
        queueCB.SetViewport(cellX, cellY, cellWidth, cellHeight);
        queueCB.SetScissor(cellX, cellY, cellWidth, cellHeight);

        queueCB.MultiDrawArraysIndirectCount(DrawPrimitive::TRIANGLES, indBufferAddr + tests[i].indBufferOffset + sizeof(DrawElementsIndirectData),
            paramBufferAddr + tests[i].drawCountOffset, tests[i].maxDrawCount, sizeof(IndirectCmdData));

        if (2*(i + 1) % numCols == 0) {
            cellX = 0;
            cellY += cellHeight;
        } else {
            cellX += cellWidth;
        }
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNMultiDrawIndirectCount, lwn_multi_draw_indirect_count, );