//
//  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
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
#include <Memory/MBuffer.h>
#include <Memory/MemoryManager.h>

#include <algorithm>

namespace optix {

class MBufferDeviceVectorBase
{
  public:
    MBufferDeviceVectorBase() = default;
    MBufferDeviceVectorBase( MemoryManager* mm, const DeviceSet& allowedDevices, MBufferPolicy policy, const BufferDimensions& dims )
        : m_handle( mm->allocateMBuffer( dims, policy, allowedDevices ) )
    {
    }
    ~MBufferDeviceVectorBase() = default;

    void addListener( MBufferListener* listener ) { m_handle->addListener( listener ); }

    void manualSynchronize( MemoryManager* mm ) { mm->manualSynchronize( m_handle ); }

    void reset() { m_handle.reset(); }

    bool operator==( const MBuffer* buffer ) const { return buffer == m_handle.get(); }

  protected:
    void* getDevicePtr( unsigned int allDeviceListIndex ) const
    {
        const int device = static_cast<int>( allDeviceListIndex );
        return m_handle ? m_handle->getAccess( device ).getLinearPtr() : nullptr;
    }

    MBufferHandle m_handle;
};

inline bool operator==( const MBuffer* buffer, const MBufferDeviceVectorBase& deviceVec )
{
    return deviceVec == buffer;
}

// Mappings from type T to an appropriate RTformat are only provided for the types
// used by PagingManager.  If you need more buffer types, then add more specializations
// here.
template <typename T>
RTformat bufferFormat()
{
    return RT_FORMAT_USER;
}

template <>
inline RTformat bufferFormat<unsigned int>()
{
    return RT_FORMAT_UNSIGNED_INT;
}

template <>
inline RTformat bufferFormat<unsigned long long>()
{
    return RT_FORMAT_UNSIGNED_LONG_LONG;
}

template <typename T>
BufferDimensions vectorDims( unsigned int numElements )
{
    return BufferDimensions( bufferFormat<T>(), sizeof( T ), 1, numElements, 1, 1 );
}

template <typename T>
class MBufferDeviceVector : public MBufferDeviceVectorBase
{
  public:
    MBufferDeviceVector() = default;

    MBufferDeviceVector( MemoryManager* mm, const DeviceSet& allowedDevices, unsigned int numElements, MBufferPolicy policy = MBufferPolicy::internal_readwrite )
        : MBufferDeviceVectorBase( mm, allowedDevices, policy, vectorDims<T>( numElements ) )
    {
    }

    T* getDevicePtr( unsigned int allDeviceListIndex ) const
    {
        return static_cast<T*>( MBufferDeviceVectorBase::getDevicePtr( allDeviceListIndex ) );
    }

    void copyToHost( MemoryManager* mm, T* dest, unsigned int numElements )
    {
        const T* source = reinterpret_cast<const T*>( mm->mapToHost( m_handle, MAP_READ ) );
        std::copy( &source[0], &source[numElements], dest );
        mm->unmapFromHost( m_handle );
    }

    void copyToDevice( MemoryManager* mm, const T* source, unsigned int numElements )
    {
        T* dest = reinterpret_cast<T*>( mm->mapToHost( m_handle, MAP_WRITE_DISCARD ) );
        std::copy( &source[0], &source[numElements], dest );
        mm->unmapFromHost( m_handle );
    }
};

}  // namespace optix
