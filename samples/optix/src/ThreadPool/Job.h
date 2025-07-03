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

/// \file
/// \brief Job and FragmentedJob

#pragma once

#include <prodlib/exceptions/Assert.h>

#include <atomic>

namespace optix {

typedef signed char Priority;

/// The abstract interface for jobs processed by the thread pool.
///
/// See #FragmentedJob for an adapter that simplifies writing jobs to be exelwted in parallel by
/// multiple threads.
class Job
{
  public:
    /// Returns the CPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float getCpuLoad() const = 0;

    /// Returns the GPU load that the job is assumed to create.
    ///
    /// The returned value must never ever change for a given instance of this interface.
    virtual float getGpuLoad() const = 0;

    /// Returns the priority of the job.
    ///
    /// The smaller the value the higher the priority of the job to be exelwted.
    ///
    /// \note Negative values are reserved for internal purposes of the thread pool.
    virtual Priority getPriority() const = 0;

    /// Notifies the job about an upcoming #execute() call from the same thread.
    ///
    /// The call to #preExelwte() is still done under the main lock of the thread pool. The state
    /// of the worker thread is still THREAD_IDLE. This method should not do any work except trivial
    /// and fast book-keeping.
    virtual void preExelwte() = 0;

    /// Exelwtes the job.
    virtual void execute() = 0;

    /// Indicates whether the remaining work of the job can be split into multiple fragments.
    ///
    /// \note This method must either always returns \c false, or the value must change from \c true
    ///       to \c false (and not back) exactly once during the lifetime of a given instance.
    ///
    /// \note If this method ever returns \c true, then the #execute() method must handle parallel
    ///       ilwocations from different threads, even if (at that time) the work cannot be split
    ///       anymore into multiple fragments or if there is no more work to do at all.
    ///
    /// \note Note that returning \c true here is only a hint, i.e., the #execute() method must not
    ///       rely on further multiple ilwocations. The #execute() method must not return until all
    ///       work is done (unless it is guaranteed that the remaining conlwrrently ongoing calls to
    ///       #execute() will do all the work).
    virtual bool isRemainingWorkSplittable() = 0;
};

/// Base class for fragmented jobs of the thread pool.
///
/// The adaptor implements #preExelwte(), #execute() and isRemainingWorkSplittable() from the
/// #Job interface in terms of the additional virtual method #exelwteFragment(). Derived classes
/// need to implement this new virtual method. In addition, they can override the methods
/// #getCpuLoad(), getGpuLoad(), and #getPriority() from the base class.
///
/// Using this base class is not mandatory for fragmented jobs, but strongly recommended.
class FragmentedJob : public Job
{
  public:
    /// Constructor.
    ///
    /// \param count          The number of fragments.
    FragmentedJob( size_t count )
        : m_count( count )
        , m_nextFragment( 0 )
        , m_outstandingFragments( count )
        , m_threads( 0 )
    {
        RT_ASSERT( m_count > 0 );
    }

    /// Fragmented jobs are non-copyable.
    FragmentedJob( const FragmentedJob& ) = delete;
    FragmentedJob& operator=( const FragmentedJob& ) = delete;

    /// Destructor.
    virtual ~FragmentedJob() {}

    /// Returns the fragment count.
    size_t getCount() const { return m_count; }

    /// Exelwtes one fragment of the fragmented job.
    ///
    /// Used by #execute().
    ///
    /// \param index         The index of the fragment.
    /// \param count         The total number of fragments.
    virtual void exelwteFragment( size_t index, size_t count ) noexcept = 0;

    /// Bounds the maximum number of worker threads for this job.
    ///
    /// Can be used to disable parallelization, e.g., for debugging.
    ///
    /// Unbounded in the default implementation. Can be overridden if desired.
    virtual size_t getThreadLimit() const { return 0; }

    /// Notification that all fragments have been exelwted.
    ///
    /// Empty in default implementation. Can be overridden if desired.
    virtual void jobFinished() {}

    //@{ Default implementation of some methods of optix::Job

    /// Returns a default CPU load of 1.0. Can be overridden if desired.
    float getCpuLoad() const override { return 1.0; }

    /// Returns a default GPU load of 0.0. Can be overridden if desired.
    float getGpuLoad() const override { return 0.0; }

    /// Returns a default priority of 0. Can be overridden if desired.
    Priority getPriority() const override { return 0; }

    /// Do not change/override this implementation unless you really know what you are doing.
    void preExelwte() final { ++m_threads; }

    /// Implemented in terms of #exelwteFragment().
    ///
    /// This method ilwokes #exelwteFragment() in a loop. Each loop iteration performs one call
    /// using a unique fragment index. The method is thread-safe, parallel calls will use distinct
    /// fragment indices.
    ///
    /// Do not change/override this implementation unless you really know what you are doing.
    void execute() final
    {
        unsigned int index                = m_nextFragment++;
        bool         exelwtedLastFragment = false;
        while( index < m_count )
        {
            exelwteFragment( index, m_count );
            exelwtedLastFragment = --m_outstandingFragments == 0;
            index                = m_nextFragment++;
        }
        if( exelwtedLastFragment )
            jobFinished();
    }

    /// Do not change/override this implementation unless you really know what you are doing.
    bool isRemainingWorkSplittable() final
    {
        size_t threadLimit = getThreadLimit();
        return ( m_nextFragment + 1 < m_count ) && ( threadLimit == 0 || m_threads < threadLimit );
    }

    //@}

  private:
    /// The number of fragments of this fragmented job.
    size_t m_count;
    /// The next fragment that will be exelwted.
    std::atomic<unsigned int> m_nextFragment;
    /// The number of fragments not yet completed.
    std::atomic<unsigned int> m_outstandingFragments;
    /// The number of threads in #execute().
    std::atomic<unsigned int> m_threads;
};

}  // namespace optix
