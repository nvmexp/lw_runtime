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

#include <Memory/MResources.h>

#include <Memory/MBuffer.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/misc/GLFunctions.h>

#include <limits>

using namespace optix;
using namespace prodlib;


MResources::MResources( MBuffer* buf )
    : m_buf( buf )
{
    algorithm::fill( m_resourceKind, None );
    algorithm::fill( m_demandTextureMinMipLevel, std::numeric_limits<unsigned int>::max() );
    algorithm::fill( m_demandTextureMaxMipLevel, 0 );
}

MResources::~MResources()
{
    // Delete cached interop FBO
    if( m_gfxInteropFBO )
        GL::DeleteFramebuffers( 1, &m_gfxInteropFBO );

    RT_ASSERT_NOTHROW( !m_gfxInteropLWDARegisteredResource.get(), "Abandoned LWCA graphics resource" );

    RT_ASSERT_NOTHROW( !m_texHeapAllocation, "Abandoned texheap allocation" );

    // GfxInterop resource has a destructor, but all other resources
    // should have been cleared by MemoryManager before this object goes
    // away.
    for( ResourceKind resourceKind : m_resourceKind )
    {
        static_cast<void>( resourceKind );  // prevents unused variable warning becoming an error
        RT_ASSERT_NOTHROW( resourceKind == None, "Abandoned resource" );
    }
}

void MResources::setResource( unsigned int allDeviceIndex, ResourceKind resourceKind, const MAccess& access )
{
    RT_ASSERT_MSG( resourceKind == None || m_resourceKind[allDeviceIndex] == None, "Double allocation of resource" );
    m_resourceKind[allDeviceIndex] = resourceKind;
    m_buf->setAccess( allDeviceIndex, access );
}

std::string MResources::toString( ResourceKind kind )
{
    switch( kind )
    {
        case HostMalloc:
            return "HostMalloc";
        case LwdaMalloc:
            return "LwdaMalloc";
        case LwdaArray:
            return "LwdaArray";
        case LwdaSparseArray:
            return "LwdaSparseArray";
        case TexHeap:
            return "TexHeap";
        case ZeroCopy:
            return "ZeroCopy";
        case LwdaMallocP2P:
            return "LwdaMallocP2P";
        case LwdaArrayP2P:
            return "LwdaArrayP2P";
        case LwdaMallocSingleCopy:
            return "LwdaMallocSingleCopy";
        case DemandLoad:
            return "DemandLoad";
        case DemandLoadArray:
            return "DemandLoadArray";
        case DemandLoadTileArray:
            return "DemandLoadTileArray";
        case LwdaSparseBacking:
            return "LwdaSparseBacking";
        case None:
            return "None";
            // default case intentionally omitted
    }
    return "Invalid";
}
