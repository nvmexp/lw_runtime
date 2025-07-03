// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "MotionRefitter.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/bvhtools/src/common/Utils.hpp> // refit kernel
#include <math.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void MotionRefitter::configure(const Config& cfg)
{
    m_cfg = cfg;
}

//------------------------------------------------------------------------

void MotionRefitter::execute(void)
{
    if (m_cfg.lwca)
    {
        m_cfg.lwca->beginTimer(getName());
        execDevice();
        m_cfg.lwca->endTimer();
    }
    else
    {
        execHost();
    }
}

//------------------------------------------------------------------------

void MotionRefitter::execDevice(void)
{
    m_cfg.ioBvh           .readWriteLWDA();
    m_cfg.ioAllNodes      .readWriteLWDA();
    m_cfg.inAabbs         .readHost();
    m_cfg.ioNodeParents   .readWriteLWDA();

    // Assumed to be a device pointer
    const char* aabbPtr = (const char*)m_cfg.inAabbs[0].aabbs;
    const int aabbStride = m_cfg.inAabbs[0].strideInBytes;
    ModelPointers model    (aabbPtr, m_cfg.numAabbs, aabbStride*m_cfg.motionSteps, MemorySpace_LWDA);

    BufferRef<BvhNode> nodes0 = m_cfg.ioAllNodes.getSubrange(0, m_cfg.maxNodes);

    BvhHeader* bvh = m_cfg.ioBvh.readWriteLWDA();

    // Refit at remaining time steps.  Note: would need to also refit at first step if we built at some other time.
    for (int i = 1; i < m_cfg.motionSteps; ++i)
    {
        // Copy topology to new time step
        memcpyDtoD( /*dest*/ m_cfg.ioAllNodes.getSubrange(m_cfg.maxNodes*i, m_cfg.maxNodes), /*src*/ nodes0);

        // Adjust offset for current motion step
        model.inputAABBPtr += aabbStride;

        // Refit
        refitBvh2( bvh, m_cfg.maxNodes, i, &model, m_cfg.ioNodeParents.readWriteLWDA() );  
    }

}

//------------------------------------------------------------------------

void MotionRefitter::execHost(void)
{
    RT_ASSERT_MSG(0, "No host path for motion refitting");
#if 0
    m_cfg.ioBvh           .readWriteHost();
    m_cfg.ioAllNodes      .readWriteHost();
    m_cfg.inAabbs         .readHost();
    // ... TODO ...
#endif

}

//------------------------------------------------------------------------
