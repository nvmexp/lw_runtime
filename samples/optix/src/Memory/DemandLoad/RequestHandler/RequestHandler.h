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
#include <Memory/DemandLoad/DevicePaging.h>
#include <Memory/DemandLoad/PinnedMemoryRingBuffer.h>

#include <o6/optix.h>  // for RTmemoryblock

#include <string>
#include <vector>

namespace optix {

class ApiCapture;
class DevicePaging;
class StagingPageAllocator;
class TextureSampler;

// RequestHandler is the base class for a page request from multiple devices.  The ilwokeCallback()
// method allocates host staging memory and ilwokes the client callback.  Derived classes implement
// handling of buffer page requests, texture tile requests and whole miplevels.
class RequestHandler
{
  public:
    virtual ~RequestHandler() = default;

    virtual void fillMemoryBlock( StagingPageAllocator* stagingPages );
    virtual bool processRequest( DeviceManager* dm, StagingPageAllocator* stagingPages, std::vector<DevicePaging>& deviceState, bool synchronous );
    virtual void captureTrace( ApiCapture& capture ) const;
    virtual void synchronize( StagingPageAllocator* stagingPages,
                              unsigned int          allDeviceListIndex,
                              DevicePaging&         devicePaging,
                              bool                  synchronous ) const = 0;
    virtual void releaseMemoryBlock( StagingPageAllocator* stagingPages );

    static void setStampMemoryBlocks( bool stampMemoryBlocks );
    static bool isSmallNonMipmappedTextureRequest( unsigned int pageId, unsigned int startPage, const TextureSampler* );
    static bool isMipTailRequest( unsigned int pageId, unsigned int startPage, const Buffer* buffer );

    bool                 isFilled() const { return m_isFilled; }
    const Buffer*        getBuffer() const { return m_buffer; }
    DeviceSet            getDeviceSet() const { return m_devices; }
    const RTmemoryblock& getMemoryBlock() const { return m_memoryBlock; }

    bool               caughtException() const { return m_caughtException; }
    const std::string& getExceptionMessage() const { return m_exceptionMessage; }

  protected:
    RequestHandler( unsigned int pageId, DeviceSet devices, const Buffer* buffer, unsigned int startPage );

    unsigned int getPageIndex() const { return m_pageId - m_startPage; }

    std::string getCallbackTraceString( bool wasFilled, const Buffer* buffer, const RTmemoryblock& memoryBlock ) const;
    std::string createExceptionMessage() const;

    void ilwokeCallback( RTmemoryblock& block );
    virtual void createMemoryBlock( StagingPageAllocator* stagingPages ) = 0;

    void stampMemoryBlock() { stampMemoryBlock( m_memoryBlock, m_allocation.id ); }
    void stampMemoryBlock( const RTmemoryblock& block, unsigned int id, unsigned int xOffset = 0, unsigned int yOffset = 0 );
    void fillMemoryBlockDebugPattern() { fillMemoryBlockDebugPattern( m_memoryBlock ); }
    void fillMemoryBlockDebugPattern( const RTmemoryblock& block );
    void checkMemoryBlockDebugPattern() { checkMemoryBlockDebugPattern( m_memoryBlock ); }
    void checkMemoryBlockDebugPattern( const RTmemoryblock& block );

    void copyDataToDevices( DeviceManager* dm, StagingPageAllocator* stagingPages, std::vector<DevicePaging>& devicePaging, bool synchronous );
    void makeLwrrent( DeviceManager* dm, unsigned int allDeviceListIndex );

    void logCallback( const char* classLabel ) const;
    void logCallback( const char* classLabel, const RTmemoryblock& memoryBlock ) const;

    unsigned int          m_pageId;
    DeviceSet             m_devices;
    const Buffer*         m_buffer;
    unsigned int          m_startPage;
    RTmemoryblock         m_memoryBlock{};
    StagingPageAllocation m_allocation{};
    bool                  m_isFilled        = false;
    bool                  m_caughtException = false;
    std::string           m_exceptionMessage;

    static bool s_stampMemoryBlocks;
};

std::string toString( const RTmemoryblock& memoryBlock );

}  // namespace optix
