// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES


#pragma once

#include <Device/DeviceManager.h>
#include <corelib/misc/Concepts.h>

#include <deque>
#include <memory>


namespace optix {

class Context;
class Plan;

class PlanManager : private corelib::NonCopyable
{
  public:
    PlanManager( Context* context );
    ~PlanManager();

    // Plan caching.
    Plan* findValidPlan( unsigned int entry, int dim, const DeviceSet& devices, int numLaunchDevices ) const;
    Plan* findOrCachePlan( std::unique_ptr<Plan> plan );

    // Activate the specified plan's frame task. Deactivates the previous
    // active one first if necessary. Passing nullptr is allowed.
    void activate( Plan* plan );

    // Remove all plans associated with the given devices from the cache.
    void removePlansForDevices( const DeviceArray& removedDevices );

    void setMaxCachedPlanCount( const int value );
    int getMaxCachedPlanCount() { return m_maxCachedPlanCount; }
  private:
    typedef std::deque<std::unique_ptr<Plan>> PlanCacheType;

    void ensureDeactivated( Plan* plan );
    void deactivateLwrrent();
    void trimCache();

    Context*      m_context            = nullptr;
    Plan*         m_activePlan         = nullptr;
    int           m_maxCachedPlanCount = 1;
    PlanCacheType m_cachedPlans;
};
}
