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
#include "lwos.h"

#include "lwnUtil/lwnUtil_Interface.h"
#include "lwn/lwn.h"

#if defined(LW_HOS)
#include <nn/os.h>
#endif

#include <inttypes.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace lwn;

#define TEST_TIMEOUT_MS 15000
#define ILWALID_VBO_ADDRESS 0xfedcba9876

#define ILWALID_VBO_ADDRESS_FOR_CLASS_EXCEPTION 0xfffffffedcba9876

#if 0
#define PRINT_INFO printf
#else
#define PRINT_INFO(...)
#endif

class LWNRobustnessTest
{
public:
    enum Variant {
        ContextSwitchTimeout,           // Single heavy draw call, ctx sw triggered by initialized queue
        SubmitTimeoutSingleDrawCall,    // Single heavy draw call timeout, no context switch
        SubmitTimeoutMultipleDrawCall,  // Multiple draw calls to trigger submit timeout
        MmuFault,                       // Invalid memory access
        PreemptTimeout,                 // Single heavy draw call, create new queue to trigger preeempt timeout
        PbdmaException,                 // Pbdma exception test
        GrException                     // GrEngine exception test
    };
protected:
    Variant m_variant;
    mutable bool m_errorDetected;
    void runDrawArrayQueue(Queue *mainQueue,
            QueueCommandBuffer &mainQueueCB,
            bool shader_timeout,
            bool mmu_fault,
            bool extra_queue,
            bool multi_drawcall,
            bool switch_context,
            bool drawResultTriangle,
            bool pbdmaException,
            bool grException) const;
    void runExtraClearQueue(void) const;
public:
    LWNRobustnessTest(Variant variant) : m_variant(variant), m_errorDetected(0) {}
    LWNTEST_CppMethods();
};

lwString LWNRobustnessTest::getDescription() const
{
    lwStringBuf sb;

    switch(m_variant) {
    case PreemptTimeout:
    {
        sb << "Forced preemption timeout test. This test trigger wait-for-idle timeout in lwhost driver "
                "by creating a new channel while other channel is stuck in very long draw call.\n";
    }
    break;
    case ContextSwitchTimeout:
    {
        sb << "Context timeout test. This test trigger HW context switch by submitting a small draw call "
                "in one channel while other channel is stuck in very long draw call.\n";
    }
    break;
    case SubmitTimeoutSingleDrawCall:
    {
        sb << "Check if single draw call timeout is detected and recovered.\n";
    }
    break;
    case SubmitTimeoutMultipleDrawCall:
    {
        sb << "Check if multi-draw submit timeout is detected and recovered.\n";
    }
    break;
    case MmuFault:
    {
        sb << "Trigger MMU fault by illegal memory access, verify recovery\n";
    }
    break;
    case PbdmaException:
    {
        sb << "Trigger PBDMA exception by invalid gpfifo entry, verify recovery\n";
    }
    break;
    case GrException:
    {
        sb << "Trigger Graphics engine exception by invalid command, verify recovery\n";
    }
    break;
    }
    return sb.str();
}

int LWNRobustnessTest::isSupported() const
{
#if defined(LW_TEGRA)
    if (!lwogCheckLWNAPIVersion(52, 19)) {
        return false;
    }

    DeviceState *deviceState = DeviceState::GetActive();
    lwn::DeviceFlagBits deviceFlags = deviceState->getDeviceFlags();
    if (deviceFlags & lwn::DeviceFlagBits::DEBUG_ENABLE_LEVEL_4) {
        switch(m_variant) {
        default:
            break;
        case MmuFault:
        case GrException:
            return false;
            break;
        }
    }
    return true;
#else
    return false;
#endif
}


void LWNRobustnessTest::runExtraClearQueue(void) const {

    DeviceState *deviceState = DeviceState::GetActive();
    lwn::Device *device = deviceState->getDevice();
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);

    Queue *queue2 = device->CreateQueue();
    lwnUtil::CompletionTracker *tracker2 = new lwnUtil::CompletionTracker(cdevice, 32);
    lwnUtil::QueueCommandBufferBase queueCB2Base;
    queueCB2Base.init(cdevice, reinterpret_cast<LWNqueue *>(queue2), tracker2);

    QueueCommandBuffer &queueCB2 = queueCB2Base;
    queueCB2.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    queueCB2.ClearColor(0, 0.4, 0.4, 0.2, 0.2);
    queueCB2.submit();

    queue2->Finish();

    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(queue2);
    if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR) {
        printf("FAIL: lwnQueueGetError() returned error in queue which is not supposed to be broken\n");
        m_errorDetected = true;
    }
    queueCB2Base.destroy();
    queue2->Free();
    delete tracker2;
}

void LWNRobustnessTest::runDrawArrayQueue(
        Queue *mainQueue,
        QueueCommandBuffer &mainQueueCB,
        bool shader_timeout,
        bool mmu_fault,
        bool extra_queue,
        bool multi_drawcall,
        bool switch_context,
        bool drawResultTriangle,
        bool pbdmaException,
        bool grException) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);

    Queue *queue2 = device->CreateQueue();

    lwnUtil::CompletionTracker *tracker2 = new lwnUtil::CompletionTracker(cdevice, 32);
    lwnUtil::QueueCommandBufferBase queueCB2Base;
    queueCB2Base.init(cdevice, reinterpret_cast<LWNqueue *>(queue2), tracker2);

    QueueCommandBuffer &queueCB2 = queueCB2Base;


    const lwn::Texture *renderTexture = g_lwnWindowFramebuffer.getAcquiredTexture();
    const LWNtexture *crenderTexture = reinterpret_cast<const LWNtexture *>(renderTexture);

    LWNcommandBuffer *cb = reinterpret_cast<LWNcommandBuffer *>(&queueCB2Base);

    lwnCommandBufferSetRenderTargets(cb, 1, &crenderTexture, NULL, NULL, NULL);
    lwnCommandBufferSetScissor(cb, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    lwnCommandBufferSetViewport(cb, 0, 0, lwrrentWindowWidth, lwrrentWindowHeight);


    VertexShader vs(440);
    vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec3 ocolor;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  ocolor = color;\n"
            "}\n";

    FragmentShader fs(440);
    if (shader_timeout) {
        // Very long draw call
        fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "void main() {\n"
                "   float a = ocolor.x;\n"
                "   for (int j = 0 ; j < 1000000; j++)\n"
                "   for (int i = 0; i < 100000 ; i++) {\n"
                "        a = cos(a * ocolor.y) + 0.5 + sin(a + ocolor.x);\n"
                "        a += float(j);\n"
                "   } \n"
                "   fcolor = vec4(a, ocolor.y, ocolor.z, 1.0);\n"
                "}\n";
    } else if (multi_drawcall) {
        // long, but less than a second draw call
        fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "void main() {\n"
                "   float a = ocolor.x;\n"
                "   for (int j = 0 ; j < 1; j++)\n"
                "   for (int i = 0; i < 10000 ; i++) {\n"
                "        a = cos(a * ocolor.y) + 0.5 + sin(a + ocolor.x);\n"
                "        a += j;\n"
                "   } \n"
                "   fcolor = vec4(a, ocolor.y, ocolor.z, 1.0);\n"
                "}\n";
    } else if (drawResultTriangle) {
        fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "void main() {\n"
                "  fcolor = vec4(ocolor, 1.0);\n"
                "}\n";
    } else {
        fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "void main() {\n"
                "  fcolor = vec4(ocolor / 2.0, 1.0);\n"
                "}\n";
    }

    Program *pgm = device->CreateProgram();

    // Compile and call lwnProgramSetShaders.
    g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const Vertex vertexData[] = {
            { dt::vec3(-0.375, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
            { dt::vec3(-0.375, +0.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
            { dt::vec3(+0.375, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
    };

    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, 3 * sizeof(vertexData),
            LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, 3, allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();
    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(queue2);

    if (mmu_fault) {
        vboAddr = ILWALID_VBO_ADDRESS;
    }

    if (grException) {
        vboAddr = ILWALID_VBO_ADDRESS_FOR_CLASS_EXCEPTION;
    }

    int count = 0;
    do {
        queueCB2.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

        queueCB2.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB2.BindVertexArrayState(vertex);

        queueCB2.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
        queueCB2.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);

        count++;
    } while (count < 1000);

    if (pbdmaException) {
        // Submit malformed gpfifo entry
        LWNcommandHandle handle;
        uint64_t gpfifoCount = 1;
        char commandInfo[64];
        memset(commandInfo, 0xff, 64);
        uint64_t addr = (uint64_t)&commandInfo;
        handle = gpfifoCount << 48 | addr | 1ULL;
        lwnQueueSubmitCommands(cqueue, 1, &handle);
    }

    queueCB2.submit();
    queue2->Flush();

    if (mmu_fault) {
        // Verify that dependent waits in other queue
        // are released after fault. This test can affect time
        // of timeout tests, so we run it for mmu_fault case only
        Sync syncObj;
        syncObj.Initialize(device);
        queue2->FenceSync(&syncObj, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0);
        queue2->Flush();

        mainQueue->WaitSync(&syncObj);
        mainQueue->Finish(); // Will hang if fence recovery in LWN doesn't work
    }

    if (switch_context) {
        mainQueueCB.ClearColor(0, 0.4, 0.1, 0.0, 0.0);
        mainQueueCB.ClearColor(0, 0.1, 0.4, 0.2, 0.2);
        mainQueueCB.submit();
        mainQueue->Flush();
    }

    if (extra_queue) {
        runExtraClearQueue();
    }

    queue2->Finish();


    // lwnQueueGetError perf test
    // This test verifies that fast codepath of lwnQueueGetError() is not disabled accidentally.
    // Observed average time of fast case (no error) is 7us, slow case is 266us.
    // Use 20us as a threshold for recognition of fast path.
    {
        LWNqueueGetErrorResult error_result = lwnQueueGetError(cqueue, NULL);
        LwU32 testTimeMs = LwOsGetTimeMS();

        int loopMult;
        if (error_result == LWN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR) {
            // Use more loops for fast path to get better precision
            loopMult = 5;
        } else {
            // No extra loops needed for slow path,
            // it may lead to TM disconnect
            loopMult = 1;
        }

        // This loop shouldn't take longer than 30ms in slow case scenario
        for (int i = 0; i < 100 * loopMult; i++) {
            if (error_result != lwnQueueGetError(cqueue, NULL)) {
                printf("FAIL: lwnQueueGetError() returns inconsistent results %d %d\n",
                        error_result, lwnQueueGetError(cqueue, NULL));
                m_errorDetected = true;
            }
        }
        LwU32 testTimeUs = (LwOsGetTimeMS() - testTimeMs) * 10 / loopMult;
        PRINT_INFO("Test results is %d us per lwnQueueGetError(,NULL) call with error_result %d\n",
                testTimeUs, error_result);


        if (error_result == LWN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR &&
                testTimeUs > 20) {
            printf("FAIL: lwnQueueGetError() for non-faulted queue is expected to take less than 20 us,"
                    "reported time is %d\n", testTimeUs);
            m_errorDetected = true;
        }
    }

    if (grException) {
        if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_ENGINE_EXCEPTION) {
            printf("FAIL: lwnQueueGetError() didn't return expected gr exception error.\n");
            m_errorDetected = true;
        }
    } else if (pbdmaException) {
            if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_PBDMA_EXCEPTION) {
                printf("FAIL: lwnQueueGetError() didn't return expected pbdma exception error.\n");
                m_errorDetected = true;
            }
    } else if (mmu_fault) {
        LWNqueueErrorInfo errorInfo;
        LWNqueueGetErrorResult error_result = lwnQueueGetError(cqueue, &errorInfo);

        if (error_result != LWN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_MMU_FAULT) {
            printf("FAIL: lwnQueueGetError() didn't recognize MMU FAULT! Error code returned: %d\n",
                    error_result);
            m_errorDetected = true;
        } else if ((errorInfo.mmuFault.faultAddress >> 16) != (ILWALID_VBO_ADDRESS >> 16)) {
            printf("FAIL: lwnQueueGetError() didn't retrieve correct MMU fault address! Address retrieved: 0x%" PRIx64 "\n",
                    errorInfo.mmuFault.faultAddress);
            m_errorDetected = true;
        }
    } else if (shader_timeout || multi_drawcall) {
        if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_ERROR_TIMEOUT) {
            printf("FAIL: lwnQueueGetError() didn't return expected timeout error.\n");
            m_errorDetected = true;
        }
    } else {
        if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR) {
            printf("FAIL: lwnQueueGetError() returned error in queue which is not supposed to be broken\n");
            m_errorDetected = true;
        }
    }

    if ( shader_timeout || multi_drawcall || mmu_fault) {
        // Faulted queue stability verification: no hangs nor crashes expected
        int count = 0;
        do {
            queueCB2.ClearColor(0, 0.4, 0.1, 0.2, 0.3);

            queueCB2.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
            queueCB2.BindVertexArrayState(vertex);

            queueCB2.BindVertexBuffer(0, vboAddr, sizeof(vertexData));
            queueCB2.DrawArrays(DrawPrimitive::TRIANGLES, 0, 3);
            count++;
            queueCB2.submit();
            // We can run out of control memory here.
            // lwntest with enabled debug layer lwntest expects non-empty tracker
            // when out-of-control-memory is encountered, so we have to insert fence here (bug 200268745)
            if ((count & 0xff) == 0) {
                tracker2->insertFence(queue2);
            }
            queue2->Flush();

            // Flip texture to verify we don't accidentally pass invalid fence from faulted queue
            if ((count & 63) == 0) {
                g_lwnWindowFramebuffer.present(cqueue);
                g_lwnWindowFramebuffer.bind();
            }

        } while (count < 1000);
        queue2->Finish();
    }

    queueCB2Base.destroy();
    delete tracker2;
    queue2->Free();

}

void LWNRobustnessTest::doGraphics() const
{
    printf("\n*** Following error prints are expected! ***\n");

    // Clean up code for main test thread.
    DeviceState *deviceState = DeviceState::GetActive();
    Queue *mainQueue = deviceState->getQueue();
    QueueCommandBuffer &mainQueueCB = deviceState->getQueueCB();

    LwU32 startTimeMS = LwOsGetTimeMS();

    mainQueueCB.submit();
    mainQueue->Finish();

    mainQueueCB.ClearColor(0, 0.4, 0.1, 0.0, 0.0);
    mainQueueCB.ClearColor(0, 0.1, 0.4, 0.2, 0.2);
    mainQueueCB.submit();
    mainQueue->Flush();

    switch(m_variant) {

    case PreemptTimeout:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                true,   // Shader timeout
                false,  // MMU fault
                true,   // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
    break;

    case ContextSwitchTimeout:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                true,   // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger timeout during queue init
                false,  // multiple draw calls
                true,   // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
    break;

    case SubmitTimeoutSingleDrawCall:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                true,   // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
    break;

    case SubmitTimeoutMultipleDrawCall:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                false,  // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                true,   // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
    break;

    case MmuFault:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                false,  // Shader timeout
                true,   // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
    break;

    case PbdmaException:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                false,  // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                true,   // pbdma
                false   // gr
        );
    }
    break;

    case GrException:
    {
        runDrawArrayQueue( mainQueue, mainQueueCB,
                false,  // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                false,  // draw results triangle
                false,  // pbdma
                true    // gr
        );
    }
    break;
    }

    // Extra submit to verify state of innocent queue
    mainQueueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    mainQueueCB.ClearColor(0, 0.4, 0.4, 0.2, 0.2);
    mainQueueCB.submit();
    mainQueue->Flush();

    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(mainQueue);
    if (lwnQueueGetError(cqueue, NULL) != LWN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR) {
        printf("FAIL! Unexpected GPU error\n");
        m_errorDetected = true;
    }

    LwU32 endTimeMS = LwOsGetTimeMS();
    // Timeouts must be detected and signaled as error
    if (endTimeMS - startTimeMS > TEST_TIMEOUT_MS) {
        m_errorDetected = true;
    }

    if (!m_errorDetected) {
        // Draw result triangle if no errors detected
        runDrawArrayQueue( mainQueue, mainQueueCB,
                false,  // Shader timeout
                false,  // MMU fault
                false,  // extra thread to trigger preempt timeout during queue init
                false,  // multiple draw calls
                false,  // use main queue to trigger context switch
                true,   // draw results triangle
                false,  // pbdma
                false   // gr
        );
    }
}

OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_ctxsw_timeout, (LWNRobustnessTest::ContextSwitchTimeout));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_mmu_fault, (LWNRobustnessTest::MmuFault));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_preempt_timeout, (LWNRobustnessTest::PreemptTimeout));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_submit_timeout_single_draw, (LWNRobustnessTest::SubmitTimeoutSingleDrawCall));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_submit_timeout_multiple_draw, (LWNRobustnessTest::SubmitTimeoutMultipleDrawCall));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_gr_exception, (LWNRobustnessTest::GrException));
OGTEST_CppTest(LWNRobustnessTest, lwn_robustness_pbdma_exception, (LWNRobustnessTest::PbdmaException));



