// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <gtest/gtest.h>

#include <ThreadPool/Job.h>
#include <ThreadPool/ThreadPool.h>

#include <chrono>
#include <exception>
#include <thread>

using namespace optix;

/// A simple test job with configurable loads and delay.
class Test_job : public Job
{
  public:
    Test_job( ThreadPool* thread_pool, size_t id, float cpuLoad, float gpuLoad, float delay )
        : m_threadPool( thread_pool )
        , m_id( id )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
    {
    }
    virtual ~Test_job() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "    Started test job %zu (CPU load %.1f, GPU load %.1f)\n", m_id, m_cpuLoad, m_gpuLoad );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay ) );

        printf( "    Finished test job %zu (CPU load %.1f, GPU load %.1f)\n", m_id, m_cpuLoad, m_gpuLoad );
    }

  private:
    ThreadPool* m_threadPool;
    size_t      m_id;
    float       m_cpuLoad;
    float       m_gpuLoad;
    float       m_delay;
};

/// A simple test job that blocks its exelwtion until continue_job() is called.
class Block_job : public Job
{
  public:
    Block_job( ThreadPool* thread_pool, unsigned int id, float cpuLoad, float gpuLoad )
        : m_threadPool( thread_pool )
        , m_id( id )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
    {
    }
    virtual ~Block_job() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "    Started block job %u (CPU load %.1f, GPU load %.1f)\n", m_id, m_cpuLoad, m_gpuLoad );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        m_condition.wait();

        printf( "    Finished block job %u (CPU load %.1f, GPU load %.1f)\n", m_id, m_cpuLoad, m_gpuLoad );
    }

    void continue_job() { m_condition.signal(); }

  private:
    ThreadPool*  m_threadPool;
    unsigned int m_id;
    float        m_cpuLoad;
    float        m_gpuLoad;
    Condition    m_condition;
};

/// A job that relwrsively submits one child job up to a certain nesting level.
class Relwrsive_job : public Job
{
  public:
    Relwrsive_job( ThreadPool* thread_pool, unsigned int level, unsigned int maxLevels, float cpuLoad, float gpuLoad, float delay )
        : m_threadPool( thread_pool )
        , m_level( level )
        , m_maxLevels( maxLevels )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
    {
        for( unsigned int i = 0; i < level + 1; ++i )
            m_prefix += "    ";
    }
    virtual ~Relwrsive_job() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "%sStarted relwrsive job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level, m_cpuLoad, m_gpuLoad );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );
        if( m_level + 1 < m_maxLevels )
        {
            std::shared_ptr<Relwrsive_job> child_job(
                std::make_shared<Relwrsive_job>( m_threadPool, m_level + 1, m_maxLevels, m_cpuLoad, m_gpuLoad, m_delay ) );
            printf(
                "%sSubmitting relwrsive job level %u (CPU load %.1f, GPU load %.1f), waiting for "
                "completion\n",
                m_prefix.c_str(), m_level + 1, m_cpuLoad, m_gpuLoad );
            m_threadPool->submitJobAndWait( child_job );
        }
        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );

        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();
        printf( "%sFinished relwrsive job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level, m_cpuLoad, m_gpuLoad );
    }

  private:
    ThreadPool*  m_threadPool;
    unsigned int m_level;
    unsigned int m_maxLevels;
    float        m_cpuLoad;
    float        m_gpuLoad;
    float        m_delay;
    std::string  m_prefix;
};

/// A job that relwrsively submits two child jobs up to a certain nesting level.
class Tree_job : public Job
{
  public:
    Tree_job( ThreadPool* thread_pool, unsigned int level, unsigned int maxLevels, float cpuLoad, float gpuLoad, float delay, Condition* condition = 0 )
        : m_threadPool( thread_pool )
        , m_level( level )
        , m_maxLevels( maxLevels )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
        , m_condition( condition )
    {
        for( unsigned int i = 0; i < level + 1; ++i )
            m_prefix += "    ";
    }
    virtual ~Tree_job() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "%sStarted tree job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level, m_cpuLoad, m_gpuLoad );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );
        if( m_level + 1 < m_maxLevels )
        {

            Condition                 m_condition0;
            Condition                 m_condition1;
            std::shared_ptr<Tree_job> job0( std::make_shared<Tree_job>( m_threadPool, m_level + 1, m_maxLevels,
                                                                        m_cpuLoad, m_gpuLoad, m_delay, &m_condition0 ) );
            std::shared_ptr<Tree_job> job1( std::make_shared<Tree_job>( m_threadPool, m_level + 1, m_maxLevels,
                                                                        m_cpuLoad, m_gpuLoad, m_delay, &m_condition1 ) );

            printf( "%sSubmitting tree job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level + 1,
                    m_cpuLoad, m_gpuLoad );
            m_threadPool->submitJob( job0 );
            printf( "%sSubmitting tree job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level + 1,
                    m_cpuLoad, m_gpuLoad );
            m_threadPool->submitJob( job1 );

            m_threadPool->suspendLwrrentJob();
            m_condition0.wait();
            m_condition1.wait();
            m_threadPool->resumeLwrrentJob();
        }
        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );

        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();
        printf( "%sFinished tree job level %u (CPU load %.1f, GPU load %.1f)\n", m_prefix.c_str(), m_level, m_cpuLoad, m_gpuLoad );

        if( m_condition )
            m_condition->signal();
    }

  private:
    ThreadPool*  m_threadPool;
    unsigned int m_level;
    unsigned int m_maxLevels;
    float        m_cpuLoad;
    float        m_gpuLoad;
    float        m_delay;
    Condition*   m_condition;
    std::string  m_prefix;
};

/// A test for fragmented jobs not based on the mixin optix::FragmentedJob, but directly
/// implementing similar functionality.
class FragmentedJobWithoutMixin : public Job
{
  public:
    FragmentedJobWithoutMixin( ThreadPool* thread_pool, float cpuLoad, float gpuLoad, float delay, unsigned int count, unsigned int threadLimit )
        : m_threadPool( thread_pool )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
        , m_count( count )
        , m_nextFragment( 0 )
        , m_threads( 0 )
        , m_threadLimit( threadLimit )
    {
    }
    virtual ~FragmentedJobWithoutMixin() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    bool     isRemainingWorkSplittable()
    {
        size_t threadLimit = getThreadLimit();
        return ( m_nextFragment + 1 < m_count ) && ( threadLimit == 0 || m_threads < threadLimit );
    }
    size_t getThreadLimit() const { return m_threadLimit; }
    void   preExelwte() { ++m_threads; }

    void execute()
    {
        unsigned int index = m_nextFragment++;
        while( index < m_count )
        {
            printf( "    Started fragment %u (CPU load %.1f, GPU load %.1f)\n", index, m_cpuLoad, m_gpuLoad );
            m_threadPool->dumpLoad();
            m_threadPool->dumpThreadStateCounters();
            std::this_thread::sleep_for( std::chrono::duration<double>( m_delay ) );
            printf( "    Finished fragment %u (CPU load %.1f, GPU load %.1f)\n", index, m_cpuLoad, m_gpuLoad );
            index = m_nextFragment++;
        }
    }

  private:
    ThreadPool*               m_threadPool;
    float                     m_cpuLoad;
    float                     m_gpuLoad;
    float                     m_delay;
    unsigned int              m_count;
    std::atomic<unsigned int> m_nextFragment;
    std::atomic<unsigned int> m_threads;
    unsigned int              m_threadLimit;
};

/// A test for fragmented jobs based on the mixin optix::FragmentedJob.
class FragmentedJobUsingMixin : public FragmentedJob
{
  public:
    FragmentedJobUsingMixin( ThreadPool* thread_pool, float cpuLoad, float gpuLoad, float delay, unsigned int count, unsigned int threadLimit )
        : FragmentedJob( count )
        , m_threadPool( thread_pool )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
        , m_threadLimit( threadLimit )
    {
    }

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }
    size_t   getThreadLimit() const { return m_threadLimit; }

    void exelwteFragment( size_t index, size_t count ) noexcept
    {
        printf( "    Started fragment %zu (CPU load %.1f, GPU load %.1f)\n", index, m_cpuLoad, m_gpuLoad );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();
        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay ) );
        printf( "    Finished fragment %zu (CPU load %.1f, GPU load %.1f)\n", index, m_cpuLoad, m_gpuLoad );
    }

  private:
    ThreadPool*  m_threadPool;
    float        m_cpuLoad;
    float        m_gpuLoad;
    float        m_delay;
    unsigned int m_threadLimit;
};

/// A simple test job with configurable priority. Compares its priority during exelwtion against
/// s_expectedPriority. Increments s_expectedPriority at the end of its exelwtion.
class PriorityJob : public Job
{
  public:
    PriorityJob( ThreadPool* thread_pool, size_t id, float cpuLoad, float gpuLoad, float delay, Priority priority )
        : m_threadPool( thread_pool )
        , m_id( id )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
        , m_priority( priority )
    {
    }
    virtual ~PriorityJob() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return m_priority; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "    Started priority job %zu (CPU load %.1f, GPU load %.1f, priority %d)\n", m_id, m_cpuLoad,
                m_gpuLoad, static_cast<int>( m_priority ) );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        RT_ASSERT( m_priority == s_expectedPriority || m_priority == 127 );
        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );
        RT_ASSERT( m_priority == s_expectedPriority || m_priority == 127 );
        m_threadPool->yield();
        RT_ASSERT( m_priority == s_expectedPriority || m_priority == 127 );
        std::this_thread::sleep_for( std::chrono::duration<double>( m_delay / 2 ) );
        RT_ASSERT( m_priority == s_expectedPriority || m_priority == 127 );

        printf( "    Finished priority job %zu (CPU load %.1f, GPU load %.1f, priority %d)\n", m_id, m_cpuLoad,
                m_gpuLoad, static_cast<int>( m_priority ) );

        ++s_expectedPriority;
    }

  private:
    ThreadPool* m_threadPool;
    size_t      m_id;
    float       m_cpuLoad;
    float       m_gpuLoad;
    float       m_delay;
    Priority    m_priority;

  public:
    static Priority s_expectedPriority;
};

Priority PriorityJob::s_expectedPriority = 0;

/// A test job for yield() that can act as parent or child job. Parent jobs launch the child jobs
/// with higher/same/lower priority and check some flag after calling yield(). Child jobs just set
/// the flag during exelwtiong.
class YieldJob : public Job
{
  public:
    YieldJob( ThreadPool* thread_pool, unsigned int id, float cpuLoad, float gpuLoad, float delay, Priority priority, bool parent )
        : m_threadPool( thread_pool )
        , m_id( id )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
        , m_priority( priority )
        , m_parent( parent )
    {
    }
    virtual ~YieldJob() {}

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return m_priority; }
    bool     isRemainingWorkSplittable() { return false; }
    void     preExelwte() {}

    void execute()
    {
        printf( "    Started yield %s job %u (CPU load %.1f, GPU load %.1f, priority %d)\n",
                m_parent ? "parent" : "child", m_id, m_cpuLoad, m_gpuLoad, static_cast<int>( m_priority ) );
        m_threadPool->dumpLoad();
        m_threadPool->dumpThreadStateCounters();

        if( !m_parent )
        {
            std::this_thread::sleep_for( std::chrono::duration<double>( m_delay ) );
            s_completedYieldChildJob = true;
        }
        else
        {
            std::shared_ptr<YieldJob> job;

            // check that yielding for higher priority job works
            s_completedYieldChildJob = false;
            job.reset( new YieldJob( m_threadPool, m_id + 1, m_cpuLoad, m_gpuLoad, m_delay, m_priority - 1, false ) );
            printf( "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)\n", m_id + 1, m_cpuLoad,
                    m_gpuLoad, static_cast<int>( m_priority - 1 ) );
            m_threadPool->submitJob( job );
            m_threadPool->yield();
            RT_ASSERT( s_completedYieldChildJob );

            // check that yielding for same priority job has no effect
            s_completedYieldChildJob = false;
            job.reset( new YieldJob( m_threadPool, m_id + 2, m_cpuLoad, m_gpuLoad, m_delay, m_priority, false ) );
            printf( "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)\n", m_id + 2, m_cpuLoad,
                    m_gpuLoad, static_cast<int>( m_priority ) );
            m_threadPool->submitJob( job );
            m_threadPool->yield();
            RT_ASSERT( !s_completedYieldChildJob );

            // check that yielding for lower priority job has no effect
            s_completedYieldChildJob = false;
            job.reset( new YieldJob( m_threadPool, m_id + 3, m_cpuLoad, m_gpuLoad, m_delay, m_priority + 1, false ) );
            printf( "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)\n", m_id + 3, m_cpuLoad,
                    m_gpuLoad, static_cast<int>( m_priority + 1 ) );
            m_threadPool->submitJob( job );
            m_threadPool->yield();
            RT_ASSERT( !s_completedYieldChildJob );
        }

        printf( "    Finished yield %s job %u (CPU load %.1f, GPU load %.1f, priority %d)\n",
                m_parent ? "parent" : "child", m_id, m_cpuLoad, m_gpuLoad, static_cast<int>( m_priority ) );
    }

  private:
    ThreadPool*  m_threadPool;
    unsigned int m_id;
    float        m_cpuLoad;
    float        m_gpuLoad;
    float        m_delay;
    Priority     m_priority;
    bool         m_parent;
    static bool  s_completedYieldChildJob;
};

bool YieldJob::s_completedYieldChildJob = false;

/// A test for jobs that generate exceptions.
class ExceptionJob : public FragmentedJob
{
  public:
    ExceptionJob( ThreadPool* thread_pool, float cpuLoad, float gpuLoad, float delay )
        : FragmentedJob( s_count )
        , m_threadPool( thread_pool )
        , m_cpuLoad( cpuLoad )
        , m_gpuLoad( gpuLoad )
        , m_delay( delay )
    {
    }

    float    getCpuLoad() const { return m_cpuLoad; }
    float    getGpuLoad() const { return m_gpuLoad; }
    Priority getPriority() const { return 0; }

    void exelwteFragment( size_t index, size_t count ) noexcept
    {
        try
        {
            if( index == 0 )
                throw std::exception();  // base class
            else if( index == 1 )
                throw std::runtime_error( "42" );  // derived class
            else if( index == 2 )
                throw std::string( "42" );  // any class not derived from std::exception
            else if( index == 3 )
                throw 42;  // non-class
            else
                RT_ASSERT_FAIL();
        }
        catch( ... )
        {
            m_exception[index] = std::lwrrent_exception();
        }
    }

    int getResult( size_t index )
    {
        if( m_exception[index] )
            std::rethrow_exception( m_exception[index] );
        else
            return 42;
    }

    static const size_t s_count = 4;

  private:
    ThreadPool* m_threadPool;
    float       m_cpuLoad;
    float       m_gpuLoad;
    float       m_delay;

    std::exception_ptr m_exception[s_count];
};

/// Submits a high number of simple jobs with different loads and delays.
TEST( ThreadPool, ManyJobs )
{
    printf( "Testing many jobs ...\n" );

    ThreadPool thread_pool( 5.0, 5.0, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    for( size_t i = 0; i < 100; ++i )
    {
        // Note: actual loads might be higher due to clipping against the global minimum.
        float                     cpuLoad = ( ( i % 10 ) + 1 ) * 0.1f;
        float                     gpuLoad = ( 10 - ( i % 10 ) ) * 0.1f;
        float                     delay   = ( ( i % 10 ) + 1 ) * 0.01f;
        std::shared_ptr<Test_job> job( std::make_shared<Test_job>( &thread_pool, i, cpuLoad, gpuLoad, delay ) );
        printf( "Submitting test job %zu (CPU load %.1f, GPU load %.1f)\n", i, cpuLoad, gpuLoad );
        thread_pool.submitJob( job );
    }

    // Wait for low priority job before shutting down.
    float                        cpuLoad = 5.0f;
    float                        gpuLoad = 5.0f;
    float                        delay   = 0.01f;
    std::shared_ptr<PriorityJob> job( std::make_shared<PriorityJob>( &thread_pool, 100, cpuLoad, gpuLoad, delay, 127 ) );
    printf( "Submitting priority job 100 (CPU load %.1f, GPU load %.1f, priority 127)\n", cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job );

    printf( "\n" );
}

/// Submits a job that relwrsively submits child jobs.
TEST( ThreadPool, ChildJobs )
{
    printf( "Testing child jobs ...\n" );

    float        cpuLoad = 1.0f;
    float        gpuLoad = 1.0f;
    float        delay   = 0.01f;
    unsigned int levels  = 8;

    ThreadPool thread_pool( cpuLoad, gpuLoad, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    std::shared_ptr<Relwrsive_job> job0( std::make_shared<Relwrsive_job>( &thread_pool, 0, levels, cpuLoad, gpuLoad, delay ) );
    printf( "Submitting relwrsive job level %d (CPU load %.1f, GPU load %.1f), waiting for completion\n", 0, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job0 );

    cpuLoad = 0.1f;
    gpuLoad = 0.1f;
    std::shared_ptr<Tree_job> job1( std::make_shared<Tree_job>( &thread_pool, 0, levels, cpuLoad, gpuLoad, delay ) );
    printf( "Submitting tree job level %d (CPU load %.1f, GPU load %.1f), waiting for completion\n", 0, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job1 );

    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    printf( "\n" );
}

/// Submits a job that can be split into several fragments.
TEST( ThreadPool, FragmentedJobs )
{
    printf( "Testing fragmented jobs ...\n" );

    float        cpuLoad = 1.0f;
    float        gpuLoad = 1.0f;
    float        delay   = 0.1f;
    unsigned int count   = 16;

    ThreadPool thread_pool( 5.0, 5.0, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    std::shared_ptr<Job> job0( std::make_shared<FragmentedJobWithoutMixin>( &thread_pool, cpuLoad, gpuLoad, delay, count, 0 ) );
    printf(
        "Submitting fragmented job w/o mixin with %u fragments (CPU load %.1f, GPU load %.1f, no thread limit), "
        "waiting for completion\n",
        count, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job0 );

    std::shared_ptr<Job> job1( std::make_shared<FragmentedJobUsingMixin>( &thread_pool, cpuLoad, gpuLoad, delay, count, 0 ) );
    printf(
        "Submitting fragmented job using mixin with %u fragments (CPU load %.1f, GPU load %.1f, no thread limit), "
        "waiting for completion\n",
        count, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job1 );

    std::shared_ptr<Job> job2( std::make_shared<FragmentedJobWithoutMixin>( &thread_pool, cpuLoad, gpuLoad, delay, count, 1 ) );
    printf(
        "Submitting fragmented job w/o mixin with %u fragments (CPU load %.1f, GPU load %.1f, thread limit 1), "
        "waiting for completion\n",
        count, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job2 );

    std::shared_ptr<Job> job3( std::make_shared<FragmentedJobUsingMixin>( &thread_pool, cpuLoad, gpuLoad, delay, count, 1 ) );
    printf(
        "Submitting fragmented job using mixin with %u fragments (CPU load %.1f, GPU load %.1f, thread limit 1), "
        "waiting for completion\n",
        count, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job3 );

    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    printf( "\n" );
}

/// Submits a second job that never fits the limits. It will be exelwted eventually when the load
/// drops to 0 after the first job has been finished.
TEST( ThreadPool, ExpensiveJobs )
{
    printf( "Testing expensive jobs ...\n" );

    float cpuLoad = 1.0f;
    float gpuLoad = 1.0f;
    float delay   = 0.1f;

    ThreadPool thread_pool( cpuLoad, gpuLoad, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    unsigned int              id = 0;
    std::shared_ptr<Test_job> job0( std::make_shared<Test_job>( &thread_pool, id, cpuLoad, gpuLoad, delay ) );
    printf( "Submitting expensive job %u (CPU load %.1f, GPU load %.1f)\n", id, cpuLoad, gpuLoad );
    thread_pool.submitJob( job0 );

    // job1 will be exelwted even though it exceeds the limits but not before the current load is
    // 0.0, i.e., job0 is finished
    cpuLoad *= 10;
    gpuLoad *= 20;
    id = 1;
    std::shared_ptr<Test_job> job1( std::make_shared<Test_job>( &thread_pool, id, cpuLoad, gpuLoad, delay ) );
    printf( "Submitting expensive job %u (CPU load %.1f, GPU load %.1f)\n", id, cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job1 );

    printf( "\n" );
}

/// Submits jobs with various priorities. Checks that they are exelwted in reverse order.
TEST( ThreadPool, Priorities )
{
    printf( "Testing priorities ...\n" );

    float cpuLoad = 1.0f;
    float gpuLoad = 1.0f;
    float delay   = 0.1f;

    ThreadPool thread_pool( cpuLoad, gpuLoad, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    // Keep thread pool busy to be able to submit the priority jobs without exelwting them during
    // submission. Otherwise a low priority job might get exelwted out of order.
    std::shared_ptr<Block_job> job0( std::make_shared<Block_job>( &thread_pool, 0, cpuLoad, gpuLoad ) );
    printf( "Submitting block job %u (CPU load %.1f, GPU load %.1f)\n", 0, cpuLoad, gpuLoad );
    thread_pool.submitJob( job0 );

    PriorityJob::s_expectedPriority = 1;

    // Submit priority jobs with priorities from 10 (lowest) to 1 (highest), exelwted in reverse
    // order.
    for( size_t i = 1; i < 11; ++i )
    {
        Priority                     priority = static_cast<Priority>( 11 - i );
        std::shared_ptr<PriorityJob> job( std::make_shared<PriorityJob>( &thread_pool, i, cpuLoad, gpuLoad, delay, priority ) );
        printf( "Submitting priority job %zu (CPU load %.1f, GPU load %.1f, priority %d)\n", i, cpuLoad, gpuLoad,
                static_cast<int>( priority ) );
        thread_pool.submitJob( job );
    }

    job0->continue_job();

    // Wait for low priority job before shutting down.
    std::shared_ptr<PriorityJob> job1( std::make_shared<PriorityJob>( &thread_pool, 11, cpuLoad, gpuLoad, delay, 127 ) );
    printf( "Submitting priority job 11 (CPU load %.1f, GPU load %.1f, priority 127)\n", cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job1 );

    printf( "\n" );
}

/// Tests the yield() functionality using child jobs of higher/same/lower priority.
TEST( ThreadPool, Yield )
{
    printf( "Testing yield() ...\n" );

    float cpuLoad = 1.0f;
    float gpuLoad = 1.0f;
    float delay   = 0.1f;

    ThreadPool thread_pool( cpuLoad, gpuLoad, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    std::shared_ptr<YieldJob> job0( std::make_shared<YieldJob>( &thread_pool, 0, cpuLoad, gpuLoad, delay, 42, true ) );
    printf( "Submitting yield parent job %u (CPU load %.1f, GPU load %.1f, priority 42)\n", 0, cpuLoad, gpuLoad );
    thread_pool.submitJob( job0 );

    // Wait for low priority job before shutting down.
    std::shared_ptr<PriorityJob> job4( std::make_shared<PriorityJob>( &thread_pool, 3, cpuLoad, gpuLoad, delay, 127 ) );
    printf( "Submitting priority job 3 (CPU load %.1f, GPU load %.1f, priority 127)\n", cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job4 );

    printf( " \n" );
}

/// Submits a job that generates exceptions.
TEST( ThreadPool, Exceptions )
{
    printf( "Testing exceptions ...\n" );

    float cpuLoad = 1.0f;
    float gpuLoad = 1.0f;
    float delay   = 0.1f;

    ThreadPool thread_pool( 5.0, 5.0, 1 );
    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    std::shared_ptr<ExceptionJob> job( std::make_shared<ExceptionJob>( &thread_pool, cpuLoad, gpuLoad, delay ) );
    printf( "Submitting exception job (CPU load %.1f, GPU load %.1f), waiting for completion\n", cpuLoad, gpuLoad );
    thread_pool.submitJobAndWait( job );

    thread_pool.dumpThreadStateCounters();
    thread_pool.dumpLoad();

    // Check that getResult() always returns the exception, and not the actual result value.
    EXPECT_THROW( job->getResult( 0 ), std::exception );

    // derived class via base class catch
    EXPECT_THROW( job->getResult( 1 ), std::exception );

    try  // derived class via derived class catch
    {
        job->getResult( 1 );
        EXPECT_TRUE( false );
    }
    catch( const std::runtime_error& e )
    {
        EXPECT_STREQ( "42", e.what() );
    }

    try
    {
        job->getResult( 2 );
        EXPECT_TRUE( false );
    }
    catch( const std::string& e )
    {
        EXPECT_EQ( "42", e );
    }

    try
    {
        job->getResult( 3 );
        EXPECT_TRUE( false );
    }
    catch( const int& e )
    {
        EXPECT_EQ( 42, e );
    }

    printf( "\n" );
}
