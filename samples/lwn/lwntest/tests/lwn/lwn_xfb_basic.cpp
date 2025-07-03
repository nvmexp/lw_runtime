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

#define LWN_XFB_BASIC_DEBUG_MEMORY              0

class LWNXFBBasicTest
{
    static const int cellsX = 8;
    static const int cellsY = 6;

    // Type of drawing operation performed during the "capture" phase of
    // transform feedback testing.
    enum CaptureType {

        // Render four triangles normally.
        CaptureXFBNormal,

        // Render four triangles with XFB paused (and pause results saved to a
        // buffer) for the middle triangle.
        CaptureXFBPauseResume,

        // Render four triangles with XFB paused (and pause results not saved
        // to a buffer) for the middle triangle.
        CaptureXFBPauseResumeNoSave,

        // Render four triangles with the transform feedback buffer sized to
        // hold only three.
        CaptureXFBOverflow,

        CaptureTotalCount
    };

    // Type of drawing operation performed during the "playback" phase of
    // testing.
    enum PlaybackType {

        // Render with a fixed vertex count, based on the expected amount of
        // capture data.
        PlaybackFixedCount,

        // Render with DrawTransformFeedback (DrawAuto).
        PlaybackAuto,

        // Render red/green based on whether query results for generated
        // primitives and XFB captured primitives are correct.
        PlaybackTestQuery,

        PlaybackTotalCount
    };

public:
    OGTEST_CppMethods();
};

lwString LWNXFBBasicTest::getDescription()
{
    lwStringBuf sb;
    sb <<
        "Basic LWN transform feedback test.\n"
        "\n"
        "This test displays a grid of cells.  Each column corresponds to a specific "
        "type of capture operations (normal, pause/resume with save-to-memory, "
        "pause/resume without save-to-memory, overflow) with or without raster "
        "discard enabled.  Each pair of rows show capture results and then "
        "playback results.  The first two playbacks do different types of "
        "draws (vertex counts from app expectations plus DrawAuto); "
        "the third checks query results and displays red/green.\n"
        "\n"
        "The normal cells are four triangles.  Playback cells (rows 1 and 3) "
        "are slightly smaller than capture cells.  Pause/resume cells "
        "(columns 1, 2, 5, and 6) should miss the triangle in the middle on "
        "playback rows (middle triangle drawn while paused).  Overflow "
        "cells (columns 3, 7) should miss the triangle on top on playback "
        "rows (last triangle drawn after overflow).  Capture cells (rows 0, 2, "
        "and 4) with discard enabled (columns 4-7) should be solid blue.\n"
        "\n"
        "The vbo layout and xfb buffer layout are the same: vec4 SkipComponents4, "
        "vec4 position, vec3 SkipComponents3, vec2 SkipComponents2, vec3 color, "
        "vec1 SkipComponents1. ";
    return sb.str();    
}

int LWNXFBBasicTest::isSupported()
{
    return lwogCheckLWNAPIVersion(12, 0);
}

void LWNXFBBasicTest::initGraphics(void)
{
    cellTestInit(cellsX, cellsY);
    lwnDefaultInitGraphics();
}

void LWNXFBBasicTest::exitGraphics(void)
{
    lwnDefaultExitGraphics();
}

void LWNXFBBasicTest::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // allocator will create pool at first allocation
    // 64k is more than for this test but anything below 64k always allocates one page in the sparse pool
    const LWNsizeiptr poolSize = 0x10000UL;
    MemoryPoolAllocator allocator(device, NULL, poolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Set up a program that mostly passes through color and position, scaling
    // primitives by 80% in X/Y.  Program transform feedback varyings to capture
    // both position and color for playback operations.
    VertexShader vs(440);
    vs <<
        "layout(location=0) in vec4 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(0.8 * position.xyz, position.w);\n"
        "  ocolor = color;\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "}\n";
    const char *varyings[] = { "gl_SkipComponents4", "gl_Position", "gl_SkipComponents3", "gl_SkipComponents2", "ocolor", "gl_SkipComponents1" };
    Program *pgm = device->CreateProgram();
    g_glslcHelper->SetTransformFeedbackVaryings(6, varyings);

    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer, holding four triangles arranged
    // in a right triangle pattern with gaps in between.
    struct Vertex {
        dt::vec4 skipComponent4;
        dt::vec4 position;
        dt::vec3 skipComponent3;
        dt::vec2 skipComponent2;
        dt::vec3 color;
        dt::vec1 skipComponent1;
    };
    static const Vertex vertexData[] = {
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-1.0, -1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.0, 0.0, 1.0), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-1.0, -0.1, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.0, 0.5, 0.5), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-0.1, -1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.0, 0.5), dt::vec1(0.0) },

        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-1.0, +0.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.0, 0.5, 0.5), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(+0.0, +0.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.5, 0.0), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(+0.0, -1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.0, 0.5), dt::vec1(0.0) },

        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(+0.1, -1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.0, 0.5), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(+0.1, -0.1, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.5, 0.0), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(+1.0, -1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(1.0, 0.0, 0.0), dt::vec1(0.0) },

        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-1.0, +0.1, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.0, 0.5, 0.5), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-1.0, +1.0, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.0, 1.0, 0.0), dt::vec1(0.0) },
        { dt::vec4(0.0, 0.0, 0.0, 0.0), dt::vec4(-0.1, +0.1, 0.0, 1.0), dt::vec3(0.0, 0.0, 0.0), dt::vec2(0.0, 0.0), dt::vec3(0.5, 0.5, 0.0), dt::vec1(0.0) },
    };
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 12, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();

    // Set up buffers to hold transform feedback vertex data, control data, and
    // query results.
    BufferAlignBits xfbAlignBits = BufferAlignBits(BUFFER_ALIGN_VERTEX_BIT |
                                                   BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *xfbBuffer = allocator.allocBuffer(&bb, xfbAlignBits, 1024);
    BufferAddress xfbAddr = xfbBuffer->GetAddress();
    LWNfloat *xfbBufferMap = (LWNfloat *) xfbBuffer->Map();
    memset(xfbBufferMap, 0, 1024*sizeof(LWNfloat));

    Buffer *xfbControlBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT, 32);
    BufferAddress xfbControlAddr = xfbControlBuffer->GetAddress();
#if LWN_XFB_BASIC_DEBUG_MEMORY
    LWNuint *xfbControlMap = (LWNuint *) xfbControlBuffer->Map();
    xfbControlMap;      // inhibit unused variable warnings - only used for debugging
#endif

    Buffer *xfbQueryBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COUNTER_BIT, 64);
    BufferAddress xfbQueryAddr = xfbQueryBuffer->GetAddress();
    LWNcounterData *xfbQueryMap = (LWNcounterData *) xfbQueryBuffer->Map();

    // Bind our vertex and program objects.
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);

    // Loop over different capture and playback modes.
    int col = 0;
    for (int captureDiscard = 0; captureDiscard < 2; captureDiscard++)
    for (int captureType = 0; captureType < CaptureTotalCount; captureType++) {
        int row = 0;
        for (int playbackType = 0; playbackType < PlaybackTotalCount; playbackType++) {

            // Skip if neither the capture nor playback cells are allowed.  We
            // test in pairs because the capture cell is required for the
            // playback cell to work.
            if (!cellAllowed(col, row) && !cellAllowed(col, row + 1)) {
                row += 2;
                continue;
            }

            // Reset XFB-related counters to zero before each capture.
            queueCB.ResetCounter(CounterType::PRIMITIVES_GENERATED);
            queueCB.ResetCounter(CounterType::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

            // Compute the number of vertices we should capture in XFB mode,
            // plus a size to use for the transform feedback buffer (limited
            // in "overflow" mode).
            int capturedVertices = (captureType == CaptureXFBNormal) ? 12 : 9;
            int xfbSize = (captureType == CaptureXFBOverflow) ? capturedVertices * sizeof(Vertex) : 1024;
            
            // First, render a cell where we are capturing data via XFB.  Use a
            // blue backdrop in rasterizer discard mode.
            SetCellViewportScissorPadded(queueCB, col, row, 2);
            if (captureDiscard) {
                queueCB.SetRasterizerDiscard(LWN_TRUE);
                queueCB.ClearColor(0, 0.0, 0.0, 1.0, 1.0);
            }
            queueCB.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            queueCB.BindTransformFeedbackBuffer(0, xfbAddr, xfbSize);
            queueCB.BeginTransformFeedback(xfbControlAddr);
            switch (captureType) {
            case CaptureXFBNormal:
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 12);
                break;
            case CaptureXFBPauseResume:
            case CaptureXFBPauseResumeNoSave:
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
                queueCB.PauseTransformFeedback((captureType == CaptureXFBPauseResume) ? xfbControlAddr : 0);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 3, 3);
                queueCB.ResumeTransformFeedback((captureType == CaptureXFBPauseResume) ? xfbControlAddr : 0);
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 6, 6);
                break;
            case CaptureXFBOverflow:
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, 12);
                break;
            default:
                assert(0);
                break;
            }
            queueCB.EndTransformFeedback(xfbControlAddr);
            if (captureDiscard) {
                queueCB.SetRasterizerDiscard(LWN_FALSE);
            }
            row++;

            // Report out counters to the query buffer.
            queueCB.ReportCounter(CounterType::PRIMITIVES_GENERATED, xfbQueryAddr);
            queueCB.ReportCounter(CounterType::TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, xfbQueryAddr + sizeof(LWNcounterData));
            queueCB.submit();

            // Use a query object to wait for the results of the capture
            // operation.  Wait on the CPU or GPU, depending on whether we
            // read back from the query buffer or not.
            Sync *sync = device->CreateSync();
            queue->FenceSync(sync, SyncCondition::/*GRAPHICS_WORLD_SPACE_COMPLETE*/ALL_GPU_COMMANDS_COMPLETE, 0);
            queue->Flush();
            if (playbackType == PlaybackTestQuery) {
                sync->Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
            } else {
                queue->WaitSync(sync);
                queue->Finish();
            }
            sync->Free();

            struct Vertex* ptr = (Vertex*) xfbBufferMap;
            for (int i = 0; i < capturedVertices; i++) {
                if (ptr->skipComponent4.x() || ptr->skipComponent4.y() || ptr->skipComponent4.z() || ptr->skipComponent4.w() ||
                    ptr->skipComponent3.x() || ptr->skipComponent3.y() || ptr->skipComponent3.z() ||
                    ptr->skipComponent2.x() || ptr->skipComponent2.y() ||
                    ptr->skipComponent1.x()) {
                    // All the skipComponent* should be zero, otherwise, stop drawing and test failed.
                    return;
                }
                ptr++;
            }
#if LWN_XFB_BASIC_DEBUG_MEMORY
            for (int i = 0; i < 1024; i++) {
                printf("xfbBufferMap[%d] = %f.\n", i, xfbBufferMap[i]);
            }
#endif

            // Now render a cell where we are displaying transform feedback data.
            SetCellViewportScissorPadded(queueCB, col, row, 2);
            queueCB.BindVertexBuffer(0, xfbAddr, 1024);
            switch (playbackType) {
            case PlaybackFixedCount:
                queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 0, capturedVertices);
                break;
            case PlaybackAuto:
                queueCB.DrawTransformFeedback(DrawPrimitive::TRIANGLES, xfbControlAddr);
                break;
            case PlaybackTestQuery:
                if (xfbQueryMap[0].counter == 4 &&
                    xfbQueryMap[1].counter == uint64_t(capturedVertices/3)) {
                    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
                } else {
                    queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
                }
                break;
            default:
                assert(0);
                break;
            }
            queueCB.submit();
            row++;
        }
        col++;
    }
}

OGTEST_CppTest(LWNXFBBasicTest, lwn_xfb_basic, );
