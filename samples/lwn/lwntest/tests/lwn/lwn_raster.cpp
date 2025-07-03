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
#include "string.h"

/**********************************************************************/

using namespace lwn;
using namespace lwn::dt;

// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_BASIC_DO_PRINTF is set to 1.
#define LWN_BASIC_DO_PRINTF     1

#if LWN_BASIC_DO_PRINTF
#define log_output printf
#else
static void log_output(const char *fmt, ...) {}
#endif

/**********************************************************************/

// Rasterstate test class
class LwnRasterTest
{
public:
    enum LwnRasterTestType {
        LWNR_POINTS,
        LWNR_LINES,
        LWNR_TRIANGLES
    };

    LWNTEST_CppMethods();

    LwnRasterTest(const LwnRasterTestType testType) : m_testType(testType) {}

private:
    bool compileShader(Device *device, Program *&pgm, VertexShader vs, FragmentShader fs) const;
    void begin(QueueCommandBuffer &cmd, VertexArrayState &vertexState, Buffer *vertexBuffer, int sizeVBO) const;

    LwnRasterTestType m_testType;
};

int LwnRasterTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(12,0);
}

lwString LwnRasterTest::getDescription() const
{
    lwString ret;
    switch (m_testType) {
        case LWNR_POINTS: {
            ret = "Draws two groups of points.\n"
                  "  1. Draw points using lwnCommandBufferSetPointSize from 0.0 to 8191.96875 in increments of 1/32.\n"
                  "  2. Draw points using gl_PointSize from 0.0 to 2047.9875 in increments of 1/16.\n"
                  "\n"
                  "For each draw, only one pixel is displayed, all others are discarded.\n"
                  "The color corresponds to the value of the point size.\n";
        } break;
        case LWNR_LINES: {
            ret = "Draws 24 rows of 20 lines each (480 in total).\n"
                  "Each line is drawn using lwnCommandBufferSetLineWidth from 32.0 down to 1.0 in increments of 1/16.\n";
        } break;
        case LWNR_TRIANGLES: {
            ret = "Draws 6 rows of 8 triangles each.\n"
            "Across a row, triangles are drawn with varying polygon state:\n"
            "  1. LWN_FRONT_FACE_CW  LWN_FACE_NONE\n"
            "  2. LWN_FRONT_FACE_CW  LWN_FACE_FRONT\n"
            "  3. LWN_FRONT_FACE_CW  LWN_FACE_BACK\n"
            "  4. LWN_FRONT_FACE_CW  LWN_FACE_FRONT_AND_BACK\n"
            "  5. LWN_FRONT_FACE_CCW LWN_FACE_NONE\n"
            "  6. LWN_FRONT_FACE_CCW LWN_FACE_FRONT\n"
            "  7. LWN_FRONT_FACE_CCW LWN_FACE_BACK\n"
            "  8. LWN_FRONT_FACE_CCW LWN_FACE_FRONT_AND_BACK\n"
            "\n"
            "When rendering:\n"
            "* Rows 0, 2, and 4 are wound CW, where rows 1, 3, and 5 are wound CCW.\n"
            "* Rows 0 and 1 are drawn with LWN_POLYGON_MODE_POINT.\n"
            "* Rows 2 and 3 are drawn with LWN_POLYGON_MODE_LINE.\n"
            "* Rows 4 and 5 are drawn with LWN_POLYGON_MODE_FILL.\n"
            "* Front facing triangles are green.\n"
            "* Back facing triangles are blue.\n";
        } break;
    }
    return ret;
}

bool LwnRasterTest::compileShader(Device *device, Program *&pgm, VertexShader vs, FragmentShader fs) const
{
    pgm = device->CreateProgram();

    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
        log_output("Shader compile error.\lwertex Source:\n%s\n\nFragment Source:\n%s\n\nInfoLog:\n%s\n", vs.source().c_str(), fs.source().c_str(), g_glslcHelper->GetInfoLog());
        LWNFailTest();
        return false;
    }

    return true;
}

void LwnRasterTest::begin(QueueCommandBuffer &cmd, VertexArrayState &vertexState, Buffer *vertexBuffer, int sizeVBO) const
{
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    cmd.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);
    cmd.BindVertexArrayState(vertexState);
    cmd.BindVertexBuffer(0, vertexBuffer->GetAddress(), sizeVBO);
}

void LwnRasterTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    MemoryPoolAllocator *pAllocator;

    switch (m_testType) {
        case LWNR_POINTS: {
            struct VertexPoints {
                vec3 pos;
            };
            const int numVertices = lwrrentWindowWidth * lwrrentWindowHeight;
            const int numStatic = 8192 * 32;
            const int numShader = 2048 * 16;
            const int shaderOffset = numVertices - numShader;
            const int sizeVBO = numVertices * sizeof(VertexPoints);

            // Shaders and Programs
            Program *pgm[2];
            FragmentShader fs(440);
            fs << "layout(location = 0) out vec4 color;\n"
                  "flat in uvec2 idAndPointSize;\n"
                  "void main() {\n";
            fs << "  uint fragmentID = (uint(gl_FragCoord.y) * " << lwrrentWindowWidth << ") + uint(gl_FragCoord.x);\n";
            fs << "  if (idAndPointSize.x == fragmentID) {\n"
                  "    color = unpackUnorm4x8(idAndPointSize.y);\n"
                  "  }\n"
                  "  else {\n"
                  "    discard;\n"
                  "  }\n"
                  "}\n";
            for (int i = 0; i < 2; i++) {
                VertexShader vs(440);
                vs << "layout(location = 0) in vec3 pos;\n"
                      "out uvec2 idAndPointSize;\n"
                      "void main() {\n"
                      "  uvec3 rgb;\n";
                if (i) {
                    vs << "  uint pointSize = floatBitsToUint(pos.z);\n"
                          "  rgb.r = ((pointSize>>7)&0xf7);\n"
                          "  rgb.g = ((pointSize>>2)&0xf7);\n"
                          "  rgb.b = ((pointSize<<3)&0xf7);\n"
                          "  gl_PointSize = pointSize/16.0f;\n";
                }
                else {
                    vs << "  rgb.r = ((gl_VertexID>>10)&0xfc);\n"
                          "  rgb.g = ((gl_VertexID>> 4)&0xfc);\n"
                          "  rgb.b = ((gl_VertexID<< 2)&0xfc);\n";
                }
                vs << "  idAndPointSize = uvec2(\n"
                      "    gl_VertexID,\n"
                      "    packUnorm4x8(vec4(rgb/255.0f, 1.0f)));\n"
                      "  gl_Position = vec4(pos.xy, 0.0f, 1.0f);\n"
                      "}\n";
                if (!compileShader(device, pgm[i], vs, fs)) {
                    return;
                }
            }

            // Vertex state
            VertexStream stream = VertexStream(sizeof(VertexPoints));
            LWN_VERTEX_STREAM_ADD_MEMBER(stream, VertexPoints, pos);
            VertexArrayState vertexState = stream.CreateVertexArrayState();

            // Vertex data
            VertexPoints *vertexData = (VertexPoints *)__LWOG_MALLOC(sizeVBO);
            if (!vertexData) {
                LWNFailTest();
                return;
            }
            else {
                // Create a vertex in the center of each pixel from (-1.0, -1.0) to (1.0, 1.0)
                // The first 262144 points are statically sized from 0.0 to 8191.96875 in increments of 1/32
                // The last 32768 points are sized in the shader from 0.0 to 2147.9875 in increments of 1/16
                int halfWidth = lwrrentWindowWidth<<3, halfHeight = lwrrentWindowHeight<<3;
                float fHalfWidth = float(halfWidth), fHalfHeight = float(halfHeight);
                int maxX = halfWidth - 8, maxY = halfHeight - 8;
                int32_t n = 0, p = 0;
                for (int j = -maxY; j <= maxY; j += 16) {
                    for (int i = -maxX; i <= maxX; i += 16) {
                        vertexData[n].pos = vec3(i/fHalfWidth, j/fHalfHeight, 0.0f);
                        if (n >= shaderOffset) {
                            float fp;
                            ct_assert(sizeof(p) == sizeof(fp));
                            memcpy(&fp, &p, sizeof(p));
                            vertexData[n].pos.setZ(fp);
                            p++;
                        }
                        n++;
                    }
                }
            }

            // Vertex buffer
            pAllocator = new MemoryPoolAllocator(device, NULL, sizeVBO, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            Buffer *vertexBuffer = stream.AllocateVertexBuffer(device, numVertices, *pAllocator, vertexData);
            __LWOG_FREE(vertexData);
            vertexData = NULL;

            // Draw static point size
            begin(cmd, vertexState, vertexBuffer, sizeVBO);
            cmd.BindProgram(pgm[0], ShaderStageBits::ALL_GRAPHICS_BITS);
            for (int i = 0; i < numStatic; i++) {
                cmd.SetPointSize(i/32.0f);
                cmd.DrawArrays(DrawPrimitive::POINTS, i, 1);

                // Submit the command buffer every submitRate iterations to avoid a too high submit frequency.
                // See Bug 200271499. The submitRate needs to be small enough to make sure that we do not run
                // out of command memory before submitting and that we have enough control memory available when
                // submitting. Note that the debug layer requires a significant amount of control memory for DTV.
                const uint32_t submitRate = 2048;
                if ((i > 0) && (i % submitRate) == 0) {
                    cmd.submit();
                    g_lwnTracker->insertFence(queue); 
                }
            }

            // Draw shader points size
            cmd.SetPointSize(0.0f);
            cmd.BindProgram(pgm[1], ShaderStageBits::ALL_GRAPHICS_BITS);
            cmd.DrawArrays(DrawPrimitive::POINTS, shaderOffset, numShader);

        } break;
        case LWNR_LINES: {
            struct VertexLines {
                vec2 pos;
            };
            const int numVertices = 1000;
            const int sizeVBO = numVertices * sizeof(VertexLines);
            const int numCols = 20, numRows = 24;

            // Shaders and Program
            Program *pgm;
            VertexShader vs(440);
            FragmentShader fs(440);
            vs << "layout(location = 0) in vec2 position;\n"
                  "void main() {\n"
                  "  gl_Position = vec4(position, 0.0f, 1.0f);\n"
                  "}\n";
            fs << "layout(location = 0) out vec4 color;\n"
                  "void main() {\n";
            fs << "  vec4 b = vec2(gl_FragCoord.x/" << lwrrentWindowWidth << ".0f, gl_FragCoord.y/" << lwrrentWindowHeight << ".0f).xyxy;\n";
            fs << "  b.zw = 1.0f - b.zw;\n"
                  "  b = b.zxzx * b.wwyy;\n"
                  "  color = vec4(b.xyz + b.www, 1.0f);\n"
                  "}\n";
            if (!compileShader(device, pgm, vs, fs)) {
                return;
            }

            // Vertex state
            VertexStream stream = VertexStream(sizeof(VertexLines));
            LWN_VERTEX_STREAM_ADD_MEMBER(stream, VertexLines, pos);
            VertexArrayState vertexState = stream.CreateVertexArrayState();

            // Vertex data
            VertexLines *vertexData = (VertexLines *)__LWOG_MALLOC(sizeVBO);
            if (!vertexData) {
                LWNFailTest();
                return;
            }
            else {
                // Create a static layout of verticals lines
                // 24 rows of 20 lines each
                int halfWidth = lwrrentWindowWidth<<3, halfHeight = lwrrentWindowHeight<<3;
                float fHalfWidth = float(halfWidth), fHalfHeight = float(halfHeight);
                int col = halfWidth/(numCols/2), halfCol = col/2, row = halfHeight/(numRows/2);
                float y = (row - halfHeight)/fHalfHeight;
                for (int i = 0; i < numCols; i++) {
                    int idx = 2*i;
                    float x = (((col * i) - halfWidth) + halfCol)/fHalfWidth;
                    vertexData[idx++].pos = vec2(x, -1.0f);
                    vertexData[idx  ].pos = vec2(x, y);
                }
                for (int j = 1; j <= numRows; j++) {
                    y = ((row * j) - halfHeight)/fHalfHeight;
                    for (int i = 0; i < numCols; i++) {
                        int idx = 2*((j * numCols) + i), idx2 = 2*(((j - 1) * numCols) + i) + 1;
                        vertexData[idx++].pos = vertexData[idx2].pos;
                        vertexData[idx  ].pos = vec2(vertexData[idx2].pos.x(), y);
                    }
                }
            }

            // Vertex buffer
            pAllocator = new MemoryPoolAllocator(device, NULL, sizeVBO, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            Buffer *vertexBuffer = stream.AllocateVertexBuffer(device, numVertices, *pAllocator, vertexData);
            __LWOG_FREE(vertexData);
            vertexData = NULL;

            // Draw Lines from width 32.0 down to 1.0
            begin(cmd, vertexState, vertexBuffer, sizeVBO);
            cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            cmd.SetLineWidth(32.0f);
            cmd.DrawArrays(DrawPrimitive::LINES, 0, 6);
            int lineWidth = 32*16;
            for (int i = 6; i < numVertices; i += 2) {
                cmd.SetLineWidth(lineWidth--/16.0f);
                cmd.DrawArrays(DrawPrimitive::LINES, i, 2);
            }

        } break;
        case LWNR_TRIANGLES: {
            struct VertexTriangles {
                vec2 pos;
            };
            const int numVertices = 144;
            const int sizeVBO = numVertices * sizeof(VertexTriangles);
            const int numRows = 6, numCols = 8;

            // Shaders and Program
            Program *pgm;
            VertexShader vs(440);
            FragmentShader fs(440);
            vs << "layout(location = 0) in vec2 position;\n"
                  "void main() {\n"
                  "  gl_Position = vec4(position, 0.0f, 1.0f);\n"
                  "}\n";
            fs << "layout(location = 0) out vec4 color;\n"
                  "void main() {\n"
                  "    color = (gl_FrontFacing) ? vec4(0.0f, 1.0f, 0.0f, 1.0f) : vec4(0.0f, 0.0f, 1.0f, 1.0f);\n"
                  "}\n";
            if (!compileShader(device, pgm, vs, fs)) {
                return;
            }

            // Vertex state
            VertexStream stream = VertexStream(sizeof(VertexTriangles));
            LWN_VERTEX_STREAM_ADD_MEMBER(stream, VertexTriangles, pos);
            VertexArrayState vertexState = stream.CreateVertexArrayState();

            // Vertex data
            VertexTriangles *vertexData = (VertexTriangles *)__LWOG_MALLOC(sizeVBO);
            if (!vertexData) {
                LWNFailTest();
                return;
            }
            else {
                // Create 6 rows of 8 triangles each
                // Even rows are wound CCW.
                // Odd rows are CW
                int halfWidth = lwrrentWindowWidth<<3, halfHeight = lwrrentWindowHeight<<3;
                float fHalfWidth = float(halfWidth), fHalfHeight = float(halfHeight);
                int col = halfWidth/(numCols/2), row = halfHeight/(numRows/2);
                int halfCol = col/2;
                int minDim = (col <= row) ? col : row;
                int colOffset = (col > row) ? col - row : 0;
                int rowOffset = (row > col) ? row - col : 0;
                int minDim16 = minDim/16, minDim15_16 = 15*minDim16;
                int yHeight = int(12.124355652982140*minDim16);
                int yOffset = (minDim - yHeight)/2;
                yHeight += yOffset;
                int idx = 0;
                int y = -halfHeight + rowOffset;
                for (int i = 0; i < numRows; i++) {
                    int x = -halfWidth + colOffset;
                    for (int j = 0; j < numCols; j++) {
                        vertexData[idx++].pos = vec2(((x + halfCol)|8)/fHalfWidth, ((y + yHeight)|8)/fHalfHeight);
                        float xValue1 = ((x + minDim15_16)|8)/fHalfWidth;
                        float xValue2 = ((x + minDim16)|8)/fHalfWidth;
                        if (i&1) {
                            float t = xValue1;
                            xValue1 = xValue2;
                            xValue2 = t;
                        }
                        float yValue = ((y + yOffset)|8)/fHalfHeight;
                        vertexData[idx++].pos = vec2(xValue1, yValue);
                        vertexData[idx++].pos = vec2(xValue2, yValue);
                        x += col;
                    }
                    y += row;
                }
            }

            // Vertex buffer
            pAllocator = new MemoryPoolAllocator(device, NULL, sizeVBO, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
            Buffer *vertexBuffer = stream.AllocateVertexBuffer(device, numVertices, *pAllocator, vertexData);
            __LWOG_FREE(vertexData);
            vertexData = NULL;

            // Begin draw
            begin(cmd, vertexState, vertexBuffer, sizeVBO);
            cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            cmd.SetPointSize(2.0f);
            cmd.SetLineWidth(2.0f);

            // Create polygon state
            PolygonState polygonState;
            polygonState.SetDefaults();

            // Loop through all PolygonState options, drawing a front and back facing triangle for each mode
            ct_assert(sizeof(LWNpolygonMode) == sizeof(int));
            ct_assert(sizeof(LWNfrontFace)   == sizeof(int));
            ct_assert(sizeof(LWNface)        == sizeof(int));
            int i = 0;
            for (int polygonMode = PolygonMode::POINT; polygonMode <= PolygonMode::FILL; polygonMode++) {
                polygonState.SetPolygonMode(PolygonMode::Enum(polygonMode));
                for (uint32_t j = 0; j < 2; j++) {
                    for (int frontFace = FrontFace::CW; frontFace <= FrontFace::CCW; frontFace++) {
                        polygonState.SetFrontFace(FrontFace::Enum(frontFace));
                        for (int lwllFace = Face::NONE; lwllFace <= Face::FRONT_AND_BACK; lwllFace++) {
                            polygonState.SetLwllFace(Face::Enum(lwllFace));
                            cmd.BindPolygonState(&polygonState);
                            cmd.DrawArrays(DrawPrimitive::TRIANGLES, i, 3);
                            i += 3;
                        }
                    }
                }
            }

        } break;
        default: return;
    }

    cmd.submit();
    queue->Finish();

    delete pAllocator;
    pAllocator = NULL;
}

OGTEST_CppTest(LwnRasterTest, lwn_raster_points,    (LwnRasterTest::LWNR_POINTS));
OGTEST_CppTest(LwnRasterTest, lwn_raster_lines,     (LwnRasterTest::LWNR_LINES));
OGTEST_CppTest(LwnRasterTest, lwn_raster_triangles, (LwnRasterTest::LWNR_TRIANGLES));
