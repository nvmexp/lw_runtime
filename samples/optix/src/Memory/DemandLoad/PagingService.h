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

#include <Memory/DemandLoad/PagingMode.h>

#include <corelib/misc/Concepts.h>

#include <vector_types.h>

#include <memory>
#include <string>

#include <stdlib.h>

namespace optix {

class Buffer;
class BufferDimensions;
class Context;
class Device;
class DeviceManager;
class DeviceSet;
class MAccess;
class MemoryManager;

class PagingService : public corelib::AbstractInterface
{
  public:
    static const unsigned int PAGE_SIZE_IN_BYTES;

    virtual void enable() = 0;

    virtual void tearDown() = 0;

    virtual void launchPrepare( const DeviceSet& devices ) = 0;
    virtual void launchComplete()                          = 0;

    virtual std::shared_ptr<size_t> reservePageTableEntries( size_t numPages ) = 0;
    virtual void releasePageTableEntries( size_t startPage, size_t numPages ) = 0;

    virtual void bufferMAccessDidChange( Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA ) = 0;

    virtual unsigned int getSoftwareMipTailFirstLevel( const BufferDimensions& dims ) const   = 0;
    virtual uint3 getPageDimensions( const BufferDimensions& dims ) const                     = 0;
    virtual size_t computeNumDemandBufferPages( const BufferDimensions& dims ) const          = 0;
    virtual size_t computeSoftwareNumDemandTexturePages( const BufferDimensions& dims ) const = 0;
    virtual size_t computeNumPages( const BufferDimensions& dims,
                                    unsigned int            tileWidth,
                                    unsigned int            tileHeight,
                                    unsigned int            mipTailFirstLevel ) const = 0;

    virtual unsigned int* getUsageBits( const Device* device ) const        = 0;
    virtual unsigned int* getResidenceBits( const Device* device ) const    = 0;
    virtual unsigned long long* getPageTable( const Device* device ) const  = 0;
    virtual unsigned long long* getTileArrays( const Device* device ) const = 0;

    virtual void preSetActiveDevices( const DeviceSet& removedDevices ) = 0;

    virtual bool getMultiThreadedCallbacksEnabled() const         = 0;
    virtual void setMultiThreadedCallbacksEnabled( bool enabled ) = 0;

    virtual void setLwdaSparseTexturesEnabled( bool enabled ) = 0;

    virtual PagingMode getLwrrentPagingMode() const = 0;

    virtual unsigned int getTileWidth() const  = 0;
    virtual unsigned int getTileHeight() const = 0;

    virtual void forceSynchronousRequestsChanged() = 0;
    virtual void useAsynchronousCopiesChanged()    = 0;
    virtual void stampMemoryBlocksChanged()        = 0;
};

std::unique_ptr<PagingService> createPagingService( Context* context, bool useWholeMipLevelCallbacks, unsigned int numVirtualPages );

}  // namespace optix
