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

#include "GatherPrimBits.hpp"
#include "GatherPrimBitsKernels.hpp"
#include <prodlib/bvhtools/src/common/SharedKernelFunctions.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void GatherPrimBits::configure(const Config& cfg)
{
  m_cfg = cfg;

  if( !m_cfg.useBufferOverlay )
  {
    if( m_cfg.primBitsFormat == PRIMBITS_DIRECT_64 )
    {
      BufferRef<uint64_t> outPrimBits = m_cfg.outPrimBitsRaw.reinterpret<uint64_t>();
      outPrimBits.setNumElems( m_cfg.numRemaps );
    }
    else if ( m_cfg.primBitsFormat == PRIMBITS_LEGACY_DIRECT_32 )
    {
      BufferRef<int> outPrimBits = m_cfg.outPrimBitsRaw.reinterpret<int>();
      outPrimBits.setNumElems( m_cfg.numRemaps );
    }
    else
    {
      RT_ASSERT_MSG(false, "GatherPrimBits only handles PRIMBITS_DIRECT_64 or PRIMBITS_LEGACY_DIRECT_32 formats");
    }
  }
}

//------------------------------------------------------------------------

void GatherPrimBits::execute(void)
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

void GatherPrimBits::execDevice(void)
{
    // Launch GatherPrimBits.
    {
        GatherPrimBitsExecParams p = {};
        p.flags                 = m_cfg.flags;
        p.numPrims              = m_cfg.numPrims;
        p.numRemaps             = m_cfg.numRemaps;
        p.primBitsFormat        = m_cfg.primBitsFormat;
        p.inRemap               = m_cfg.inRemap.readLWDA();
        p.inPrimBits            = m_cfg.inPrimBitsRaw.readLWDA() + m_cfg.inPrimBitsOffset;
        p.inPrimBitsStride      = m_cfg.inPrimBitsStride;
        p.outPrimBits           = m_cfg.outPrimBitsRaw.writeLWDA() + m_cfg.outPrimBitsOffset;
        p.outPrimBitsStride     = m_cfg.outPrimBitsStride;

        LAUNCH(*m_cfg.lwca, GatherPrimBitsExec, GATHER_PRIMBITS_EXEC_WARPS_PER_BLOCK, p.numRemaps, p);
    }
}

//------------------------------------------------------------------------

void GatherPrimBits::execHost(void)
{
    m_cfg.inRemap.readHost();
    unsigned char *outPrimBits = m_cfg.outPrimBitsRaw.writeHost() + m_cfg.outPrimBitsOffset;
    const unsigned char *inPrimBits = m_cfg.inPrimBitsRaw.readHost() + m_cfg.inPrimBitsOffset;

    if(m_cfg.primBitsFormat == PRIMBITS_DIRECT_64)
    {
        for( int i = 0; i < m_cfg.numRemaps; ++i )
        {
            int primIdx = m_cfg.inRemap[i];
          
            uint64_t *dest = make_ptr<uint64_t>(outPrimBits, size_t(i) * size_t(m_cfg.outPrimBitsStride));
            if( primIdx < 0 || m_cfg.numPrims <= primIdx )
                *dest = ~uint64_t(0);
            else if( inPrimBits )
               *dest = *make_ptr<const uint64_t>(inPrimBits, size_t(primIdx) * size_t(m_cfg.inPrimBitsStride)) | m_cfg.flags;
            else
               *dest = primIdx | m_cfg.flags;
        }
    }
    else // Must be PRIMBITS_LEGACY_DIRECT_32
    {
        uint32_t flags = (uint32_t)m_cfg.flags;
        for( int i = 0; i < m_cfg.numRemaps; ++i )
        {
            int primIdx = m_cfg.inRemap[i];
            uint32_t *dest = make_ptr<uint32_t>(outPrimBits, size_t(i) * size_t(m_cfg.outPrimBitsStride));
            if( primIdx < 0 || m_cfg.numPrims <= primIdx )
                *dest = ~0;
            else if( inPrimBits )
               *dest = *make_ptr<const uint32_t>(inPrimBits, size_t(primIdx) * size_t(m_cfg.inPrimBitsStride)) | m_cfg.flags;
            else
               *dest = primIdx | flags;
        }
    }
}

//------------------------------------------------------------------------
