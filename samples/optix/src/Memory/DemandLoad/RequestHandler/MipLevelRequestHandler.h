//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <Device/DeviceSet.h>
#include <Memory/DemandLoad/RequestHandler/RequestHandler.h>

#include <vector>

namespace optix {

class Buffer;
class DevicePaging;
class StagingPageAllocator;
class TextureSampler;

// MipLevelRequestHandler represents a request for one of the levels in the mip tail.
class MipLevelRequestHandler : public RequestHandler
{
  public:
    MipLevelRequestHandler( unsigned int          mipLevel,
                            unsigned int          pageId,
                            DeviceSet             devices,
                            const Buffer*         buffer,
                            unsigned int          startPage,
                            bool                  useSimpleAllocationStrategy );
    ~MipLevelRequestHandler() override = default;

    void synchronize( StagingPageAllocator* stagingPages, unsigned int allDeviceListIndex, DevicePaging& devicePaging, bool synchronous ) const override;
    void releaseMemoryBlock( StagingPageAllocator* stagingPages ) override;

    unsigned int getMipLevel() const { return m_mipLevel; }

  private:
    unsigned int          m_mipLevel;
    std::vector<uint8_t>  m_stagingMemory;
    bool                  m_useSimpleAllocationStrategy;

    void createMemoryBlock( StagingPageAllocator* stagingPages ) override;
};

}  // namespace optix
