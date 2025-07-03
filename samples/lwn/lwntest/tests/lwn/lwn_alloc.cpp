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

#define LOG_PERF        1

// Base class for allocation tests for objects of different types.
class LWNAllocationTest
{
private:
    int m_allocCount;           // number of objects to allocate
    virtual const char *getObjectName() const = 0;
    virtual bool render() const = 0;
public:

    // Timer class for recording CPU timestamps and deltas.
    class Timer {
        double m_lastTime;
    public:
        static double getTime()
        {
            return (double) lwogGetTimerValue() / (double) lwogGetTimerFrequency();
        }
        double getInterval()
        {
            double newTime = getTime();
            double delta = newTime - m_lastTime;
            m_lastTime = newTime;
            return delta;
        }
        Timer()
        {
            m_lastTime = getTime();
        }
    };

    // When performance logging is enabled, report time deltas for the
    // specified operation using a previously recorded start time in <timer>.
    void reportDelta(Timer &timer, const char *operation) const
    {
#if LOG_PERF
        double delta = timer.getInterval();
        printf("%d %s objects, %10.6f seconds to %s\n", m_allocCount, getObjectName(), delta, operation);
#endif
    }

    int getAllocCount() const { return m_allocCount; }
    LWNAllocationTest(int count) : m_allocCount(count) {}
    virtual ~LWNAllocationTest() {}     // NOP virtual destructor to appease GCC
    LWNTEST_CppMethods();
};

// Specialized allocation test for memory pools.
class LWNMemoryPoolAllocationTest : public LWNAllocationTest
{
    const char *getObjectName() const  { return "memory pool"; }
    bool render() const
    {
        DeviceState *deviceState = DeviceState::GetActive();
        Device *device = deviceState->getDevice();
        QueueCommandBuffer &queueCB = deviceState->getQueueCB();
        Queue *queue = deviceState->getQueue();
        bool result = true;

        // Set up a memory pool builder, pool memory, and an array of memory
        // pool objects for the required number of memory pools.
        int nPools = getAllocCount();
        static const int poolSize = 4 * 1024;
        char *poolStorage = (char *) PoolStorageAlloc(nPools * poolSize);
        if (!poolStorage) {
            return false;
        }
        MemoryPoolBuilder mpb;
        MemoryPool *pools = new MemoryPool[nPools];
        if (!pools) {
            PoolStorageFree(poolStorage);
            return false;
        }
        mpb.SetDefaults().SetDevice(device);

        // Allocate the pools in a loop, sending and flushing trivial commands
        // on each iteration to stress WDDM allocation list logic.
        LWNAllocationTest::Timer timer;
        for (int i = 0; i < nPools; i++) {
            mpb.SetStorage(poolStorage + i * poolSize, poolSize);
            if (!pools[i].Initialize(&mpb)) {
                nPools = i;
                result = false;
                break;
            }

            // If we are running a variant that flushes on every allocation,
            // send a dummy NOP command and flush.
            if (m_flush) {
                queueCB.SetLineWidth(1.0);
                queueCB.submit();
                queue->Flush();
                if (0 == (i % 128)) {
                    // Throw in periodic fences to make sure we don't overflow
                    // command/control memory.
                    deviceState->insertFence();
                }
            }
        }

        // Send another dummy NOP command at the end to force us to do *some*
        // flush after allocating, even for the no-flush variant.
        queueCB.SetLineWidth(1.0);
        queueCB.submit();
        queue->Finish();
        reportDelta(timer, "initialize");

        // Clean up all the pools, and send another trivial command.
        for (int i = 0; i < nPools; i++) {
            pools[i].Finalize();
        }
        queueCB.SetLineWidth(1.0);
        queueCB.submit();
        queue->Finish();
        reportDelta(timer, "finalize");

        PoolStorageFree(poolStorage);
        delete[] pools;
        return result;
    }
    bool m_flush;
public:
    LWNMemoryPoolAllocationTest(int count, bool flush) :
        LWNAllocationTest(count), m_flush(flush) {}
};

// Specialized allocation test for program objects.
class LWNProgramAllocationTest : public LWNAllocationTest
{
    int m_cases;        // number of "case" statements in our switch
    const char *getObjectName() const  { return "program"; }
    bool render() const
    {
        DeviceState *deviceState = DeviceState::GetActive();
        Device *device = deviceState->getDevice();
        bool result = true;

        // Helper with custom pool size to fit all our programs
        // without requiring resizing the memory pool or running
        // the risk of having shaders placed in the last 1024 bytes
        // in the memory.
        lwnTest::GLSLCHelper glslcHelper(device,
                                         0x1000000, // 16 MB
                                         g_glslcLibraryHelper,
                                         g_glslcHelperCache);

        // Compile a dummy vertex and fragment shader to use in our program.
        // Include a switch statement (with a configurable number of cases) to
        // make our program not trivially small.
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
        fs <<
            "in vec3 ocolor;\n"
            "out vec4 fcolor;\n"
            "uniform Block {\n"
            "  int selector;\n"
            "};\n"
            "void main() {\n"
            "  vec3 rgb;\n"
            "  switch (selector) {\n"
            "  default: rgb = vec3(1.0); break;\n";
        for (int i = 0; i < m_cases; i++) {
            fs << "case " << i << ": rgb = vec3(" << i*i << ".0 / 65536.0); break;\n";
        }
        fs <<
            "  }\n"
            "  fcolor = vec4(ocolor * rgb, 1.0);\n"
            "}\n";

        if (!glslcHelper.CompileShaders(vs, fs)) {
            printf("%s\n", glslcHelper.GetInfoLog());
            return false;
        }
        const GLSLCoutput *output = glslcHelper.GetGlslcOutput();
        if (!output) {
            return false;
        }

        // Set up a collection of program objects using our canned shader.
        LWNAllocationTest::Timer timer;
        int nPrograms = getAllocCount();
        Program *programs = new Program[nPrograms];
        if (!programs) {
            return false;
        }
        for (int i = 0; i < nPrograms; i++) {
            if (!programs[i].Initialize(device) ||
                !glslcHelper.SetShaders(&programs[i], output))
            {
                nPrograms = i;
                result = false;
                break;
            }
        }
        reportDelta(timer, "initialize");

        // Tear down the program objects.
        for (int i = 0; i < nPrograms; i++) {
            programs[i].Finalize();
        }
        reportDelta(timer, "finalize");

        delete [] programs;
        return result;
    }
public:
    LWNProgramAllocationTest(int count, int cases) :
        LWNAllocationTest(count), m_cases(cases) {}
};

// Specialized allocation test for queues.
class LWNQueueAllocationTest : public LWNAllocationTest
{
    const char *getObjectName() const  { return "queue"; }
    bool render() const
    {
        DeviceState *deviceState = DeviceState::GetActive();
        Device *device = deviceState->getDevice();
        bool result = true;

        // Set up a command buffer object and record a stupid little command
        // set that can be sent to each queue.
        CommandBuffer cb;
        cb.Initialize(device);
        g_lwnCommandMem.populateCommandBuffer(&cb, CommandBufferMemoryManager::Coherent);
        cb.BeginRecording();
        cb.SetLineWidth(1.0);
        CommandHandle cmdset = cb.EndRecording();

        // Allocate a batch of queues, and send our command set to each.
        LWNAllocationTest::Timer timer;
        int nQueues = getAllocCount();
        Queue *queues = new Queue[nQueues];
        if (!queues) {
            return false;
        }
        QueueBuilder qb;
        qb.SetDefaults().SetDevice(device);
        for (int i = 0; i < nQueues; i++) {
            if (!queues[i].Initialize(&qb)) {
                nQueues = i;
                result = false;
                break;
            }
            queues[i].SubmitCommands(1, &cmdset);
            queues[i].Finish();
        }
        reportDelta(timer, "initialize");

        // Tear down the batch of queues.
        for (int i = 0; i < nQueues; i++) {
            queues[i].Finalize();
        }
        reportDelta(timer, "finalize");

        cb.Finalize();
        delete [] queues;
        return result;
    }
public:
    LWNQueueAllocationTest(int count) : LWNAllocationTest(count) {}
};

lwString LWNAllocationTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic LWN test to exercise initializing and finalizing " <<
        getAllocCount() << " " << getObjectName() << " objects.";
    return sb.str();    
}

int LWNAllocationTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 8);
}

void LWNAllocationTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Run the appropriate test from the derived class and then display
    // red/green based on the result.
#if LOG_PERF
    printf("\n");
#endif
    bool result = render();
    if (result) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNMemoryPoolAllocationTest, lwn_alloc_pools, (8192, false));
OGTEST_CppTest(LWNMemoryPoolAllocationTest, lwn_alloc_pools_flush, (8192, true));
OGTEST_CppTest(LWNProgramAllocationTest, lwn_alloc_programs, (2048, 64));
OGTEST_CppTest(LWNQueueAllocationTest, lwn_alloc_queues_4, (4));
OGTEST_CppTest(LWNQueueAllocationTest, lwn_alloc_queues_8, (8));
OGTEST_CppTest(LWNQueueAllocationTest, lwn_alloc_queues_32, (32));
