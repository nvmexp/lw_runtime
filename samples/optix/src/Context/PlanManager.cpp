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

#include <Context/PlanManager.h>

#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/Plan.h>

#include <corelib/misc/String.h>
#include <prodlib/system/Knobs.h>

#include <algorithm>

using namespace optix;


namespace {
// clang-format off
  Knob<int> k_planCacheSize( RT_DSTRING( "context.planCacheSize"), 64, RT_DSTRING( "Maximum number of entries in the plan cache" ) );
// clang-format on
}

PlanManager::PlanManager( Context* context )
    : m_context( context )
    , m_maxCachedPlanCount( k_planCacheSize.get() )
{
}

PlanManager::~PlanManager()
{
    deactivateLwrrent();
}

Plan* PlanManager::findValidPlan( unsigned int entry, int dim, const DeviceSet& devices, int numLaunchDevices ) const
{
    // Look through N recent plans to find the first one that is still
    // valid.  Note that this code does not remove invalid plans for the
    // cache, since we might still generate another compatible plan.

    for( const auto& plan : m_cachedPlans )
    {
        if( plan->isValid() && plan->supportsLaunchConfiguration( entry, dim, devices, numLaunchDevices ) )
            return plan.get();
    }
    return nullptr;
}

Plan* PlanManager::findOrCachePlan( std::unique_ptr<Plan> plan )
{
    llog( 20 ) << "findOrCachePlan:   " << plan->summaryString() << '\n';

    // See if there is a compatible version in the cache
    for( auto iter = m_cachedPlans.begin(); iter != m_cachedPlans.end(); ++iter )
    {
        Plan* cachedPlan = iter->get();
        llog( 20 ) << "Looking at plan:   " << cachedPlan->summaryString() << '\n';

        if( cachedPlan->isCompatibleWith( plan.get() ) )
        {
            llog( 20 ) << "Revalidating plan: " << cachedPlan->summaryString() << '\n';

            // Update the plan so that it is no longer invalid
            cachedPlan->revalidate();

            // Move it to the front (LRU) if necessary
            if( iter != m_cachedPlans.begin() )
            {
                iter->release();  // briefly take ownership so we can erase iter
                m_cachedPlans.erase( iter );
                m_cachedPlans.push_front( std::unique_ptr<Plan>( cachedPlan ) );
            }

            return cachedPlan;
        }
    }

    // Trim cache
    trimCache();

    // Add the plan
    llog( 20 ) << "Saving plan:       " << plan->summaryString() << '\n';

    Plan* retVal = plan.get();
    m_cachedPlans.push_front( std::move( plan ) );
    return retVal;
}

void PlanManager::activate( Plan* plan )
{
    llog( 30 ) << "Active Plan: " << m_activePlan << " Activating: " << plan << '\n';
    if( plan != m_activePlan )
    {
        if( m_activePlan )
            m_activePlan->getTask()->deactivate();
        if( plan )
            plan->getTask()->activate();
        m_activePlan = plan;
    }
}

void PlanManager::removePlansForDevices( const DeviceArray& removedDevices )
{
    const DeviceSet removedSet( removedDevices );

    // Find all plans that are associated with the devices, make sure
    // none of them is active, and remove them from the cache.

    m_cachedPlans.erase( std::remove_if( m_cachedPlans.begin(), m_cachedPlans.end(),
                                         [this, removedSet]( const std::unique_ptr<Plan>& plan ) {
                                             if( !( plan->getDevices() & removedSet ).empty() )
                                             {
                                                 ensureDeactivated( plan.get() );
                                                 return true;
                                             }
                                             return false;
                                         } ),
                         m_cachedPlans.end() );
}

void PlanManager::ensureDeactivated( Plan* plan )
{
    if( plan != nullptr && plan == m_activePlan )
    {
        plan->getTask()->deactivate();
        m_activePlan = nullptr;
    }
}

void PlanManager::deactivateLwrrent()
{
    if( m_activePlan )
    {
        m_activePlan->getTask()->deactivate();
        m_activePlan = nullptr;
    }
}

void PlanManager::trimCache()
{
    // Trim cache
    while( (int)m_cachedPlans.size() >= m_maxCachedPlanCount )
    {
        ensureDeactivated( m_cachedPlans.back().get() );
        m_cachedPlans.pop_back();
    }
}

void PlanManager::setMaxCachedPlanCount( const int value )
{
    // Knob has precedence. If it is set, we only use that value.
    if( k_planCacheSize.isDefault() )
    {
        // We will not shrink below the knob (default) value.
        m_maxCachedPlanCount = std::max( k_planCacheSize.get(), value );
        // Trim cache
        trimCache();
    }
}