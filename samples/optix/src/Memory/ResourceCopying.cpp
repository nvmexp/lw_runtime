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

#include <Memory/ResourceManager.h>

#include <LWCA/Memory.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Memory/MResources.h>
#include <Memory/MemoryManager.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/misc/BufferFormats.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>

#include <cstring>

using namespace optix;
using namespace prodlib;


namespace {
Knob<int> k_mmll( RT_DSTRING( "mem.logLevel" ), 30, RT_DSTRING( "Log level used for MemoryManager logging" ) );
}

/****************************************************************
 *
 * Inter-device and inter-allocation copies. This is an N^2
 * operation. Boil it down to some common cases.  SGP: consider a
 * better naming scheme, perhaps HM_to_GI or similar.
 *
 ****************************************************************/

namespace {

enum CopyMethod
{
    cpMemcpy,
    cpDtoD,
    cpDtoD_th,
    cpArraytoArray,
    cpDtoArray,
    cpArraytoD,
    cpDtoH,
    cpArraytoH,
    cpDtoH_th,
    cpDtoH_zc,
    cpHtoD,
    cpHtoArray,
    cpHtoD_th,
    cpHtoD_zc,
    cpZeroCopy,
    cpTwoStage,
    cpHtoP2P,
    cpP2PtoH,
    cpP2PtoP2P,
    cpArrayP2PtoArrayP2P,
    cpHtoArrayP2P,
    cpSCtoH,
    cpHtoSC,
    cpSCtoSC,
    cpIlwalid,
};

}  // namespace

// clang-format off
static CopyMethod copyTable[14][14] = {
    //   TO                                                   FROM
    //                          HostMalloc,    LwdaArray,      LwdaSparseArray, LwdaMalloc, TexHeap,    ZeroCopy,   LwdaMallocP2P, LwdaArrayP2P,         LwdaSingleCopy, DemandLoad, DemandLoadArray, DemandLoadTileArray, LwdaSparseBacking, None
    /* HostMalloc          */ { cpMemcpy,      cpArraytoH,     cpIlwalid,       cpDtoH,     cpDtoH_th,  cpDtoH_zc,  cpP2PtoH,      cpIlwalid,            cpSCtoH,        cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaArray           */ { cpHtoArray,    cpArraytoArray, cpIlwalid,       cpDtoArray, cpTwoStage, cpTwoStage, cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaSparseArray     */ { cpHtoArray,    cpArraytoArray, cpIlwalid,       cpDtoArray, cpTwoStage, cpTwoStage, cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaMalloc          */ { cpHtoD,        cpArraytoD,     cpIlwalid,       cpDtoD,     cpTwoStage, cpTwoStage, cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* TexHeap             */ { cpHtoD_th,     cpTwoStage,     cpIlwalid,       cpTwoStage, cpDtoD_th,  cpTwoStage, cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* ZeroCopy            */ { cpHtoD_zc,     cpTwoStage,     cpIlwalid,       cpTwoStage, cpTwoStage, cpZeroCopy, cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaMallocP2P       */ { cpHtoP2P,      cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpP2PtoP2P,    cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaArrayP2P        */ { cpHtoArrayP2P, cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpArrayP2PtoArrayP2P, cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaSingleCopy      */ { cpHtoSC,       cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpSCtoSC,       cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* DemandLoad          */ { cpIlwalid,     cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* DemandLoadArray     */ { cpIlwalid,     cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* DemandLoadTileArray */ { cpIlwalid,     cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* LwdaSparseBacking   */ { cpIlwalid,     cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
    /* None                */ { cpIlwalid,     cpIlwalid,      cpIlwalid,       cpIlwalid,  cpIlwalid,  cpIlwalid,  cpIlwalid,     cpIlwalid,            cpIlwalid,      cpIlwalid,  cpIlwalid,       cpIlwalid,           cpIlwalid,         cpIlwalid, },
};
// clang-format on

static std::string toString( CopyMethod kind )
{
    switch( kind )
    {
        case cpMemcpy:
            return "cpMemcpy";
        case cpDtoD:
            return "cpDtoD";
        case cpDtoD_th:
            return "cpDtoD_th";
        case cpArraytoArray:
            return "cpArraytoArray";
        case cpDtoArray:
            return "cpDtoArray";
        case cpArraytoD:
            return "cpArraytoD";
        case cpDtoH:
            return "cpDtoH";
        case cpArraytoH:
            return "cpArraytoH";
        case cpDtoH_th:
            return "cpDtoH_th";
        case cpDtoH_zc:
            return "cpDtoH_zc";
        case cpHtoD:
            return "cpHtoD";
        case cpHtoArray:
            return "cpHtoArray";
        case cpHtoD_th:
            return "cpHtoD_th";
        case cpHtoD_zc:
            return "cpHtoD_zc";
        case cpZeroCopy:
            return "cpZeroCopy";
        case cpTwoStage:
            return "cpTwoStage";
        case cpHtoP2P:
            return "cpHtoP2P";
        case cpP2PtoH:
            return "cpP2PtoH";
        case cpP2PtoP2P:
            return "cpP2PtoP2P";
        case cpArrayP2PtoArrayP2P:
            return "cpArrayP2PtoArrayP2P";
        case cpHtoArrayP2P:
            return "cpHtoArrayP2P";
        case cpSCtoH:
            return "cpSCtoH";
        case cpHtoSC:
            return "cpHtoSC";
        case cpSCtoSC:
            return "cpSCtoSC";
        case cpIlwalid:
            return "cpIlwalid";
            // default case intentionally omitted
    }
    return "invalid";
}


void ResourceManager::copyResource( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_MSG( dst != src || dstDevice != srcDevice, "Invalid in-place copy" );
    TIMEVIZ_FUNC;

    // Select the type of copy based on the destination and source addresses.
    int                      dstIndex    = dstDevice->allDeviceListIndex();
    int                      srcIndex    = srcDevice->allDeviceListIndex();
    MResources::ResourceKind dstResource = dst->m_resourceKind[dstIndex];
    MResources::ResourceKind srcResource = src->m_resourceKind[srcIndex];

    static_assert( MResources::None == 13, "Insufficient entries in copyTable for MResource::ResourceKind enums" );

    RT_ASSERT( dstResource < 14 );
    RT_ASSERT( srcResource < 14 );
    CopyMethod kind = copyTable[dstResource][srcResource];

    llog( k_mmll.get() ) << " - copy resource from: " << srcDevice->allDeviceListIndex() << "/"
                         << MResources::toString( srcResource ) << " to: " << dstDevice->allDeviceListIndex() << "/"
                         << MResources::toString( dstResource ) << " using copy method: " << ::toString( kind ) << '\n';

    if( dims.zeroSized() )
        return;

    switch( kind )
    {
        case cpMemcpy:
            copyMemcpy( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoD:
            copyDtoD( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoD_th:
            copyDtoD_th( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpArraytoArray:
            copyArraytoArray( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoArray:
            copyDtoArray( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpArraytoD:
            copyArraytoD( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoH:
            copyDtoH( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpArraytoH:
            copyArraytoH( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoH_th:
            copyDtoH_th( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpDtoH_zc:
            copyDtoH_zc( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoD:
            copyHtoD( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoArray:
            copyHtoArray( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoD_th:
            copyHtoD_th( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoD_zc:
            copyHtoD_zc( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpTwoStage:
            copyTwoStage( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpZeroCopy:
            copyZeroCopy( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoP2P:
            copyHtoP2P( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpP2PtoH:
            copyP2PtoH( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpP2PtoP2P:
            copyP2PtoP2P( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpArrayP2PtoArrayP2P:
            copyArrayP2PtoArrayP2P( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoArrayP2P:
            copyHtoArrayP2P( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpSCtoH:
            copySCtoH( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpHtoSC:
            copyHtoSC( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpSCtoSC:
            copySCtoSC( dst, dstDevice, src, srcDevice, dims );
            break;
        case cpIlwalid:
            copyIlwalid( dst, dstDevice, src, srcDevice, dims );
            break;
            // Default case intentionally omitted
    }
}


void ResourceManager::copyMemcpy( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_FAIL_MSG( "copyMemcpy" );
}

void ResourceManager::copyDtoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );
    char*         srcPtr    = srcAccess.getLinearPtr();
    char*         dstPtr    = dstAccess.getLinearPtr();

    LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
    LWDADevice* dstLwda = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );

    const size_t sizeInBytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - copyDtoD bytes: " << sizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)srcPtr << " dst: " << dstDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)dstPtr << '\n';

    // Differentiate between copies on the same device and copies across devices
    if( srcDevice == dstDevice )
    {
        // This case is needed for policy changes of buffers with RT_BUFFER_DISCARD_HOST_MEMORY
        // when ResourceManager::acquireLwdaArray fails and falls back to LWDA_MALLOC
        lwca::memcpy( (LWdeviceptr)dstPtr, (LWdeviceptr)srcPtr, sizeInBytes );
    }
    else
    {
        // TODO: async?
        lwca::memcpyPeer( (LWdeviceptr)dstPtr, dstLwda->lwdaContext(), (LWdeviceptr)srcPtr, srcLwda->lwdaContext(), sizeInBytes );
    }
}

void ResourceManager::copyDtoD_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    // Compute the pointers as an offset from the base of the backing store
    const MAccess srcBackingAccess = m_texHeapBacking->getAccess( srcDevice );
    const MAccess dstBackingAccess = m_texHeapBacking->getAccess( dstDevice );
    size_t        srcOffset        = *src->m_texHeapAllocation * m_texHeapBacking->getDimensions().elementSize();
    size_t        dstOffset        = *dst->m_texHeapAllocation * m_texHeapBacking->getDimensions().elementSize();

    char* srcPtr = srcOffset + srcBackingAccess.getLinearPtr();
    char* dstPtr = dstOffset + dstBackingAccess.getLinearPtr();

    LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
    LWDADevice* dstLwda = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );

    const size_t sizeInBytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - copyDtoD_th bytes: " << sizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)srcPtr << " dst: " << dstDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)dstPtr << '\n';

    // Differentiate between copies on the same device and copies across devices
    if( srcDevice == dstDevice )
    {
        // This case is lwrrently not needed, so we leave it unimplemented because hitting it indicates a bug.
        RT_ASSERT_FAIL_MSG( "copyDtoD_th - intra GPU" );
    }
    else
    {
        // TODO: async?
        lwca::memcpyPeer( (LWdeviceptr)dstPtr, dstLwda->lwdaContext(), (LWdeviceptr)srcPtr, srcLwda->lwdaContext(), sizeInBytes );
    }
}

void ResourceManager::copyArraytoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    unsigned int dstIndex = dstDevice->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[dstIndex] == MResources::LwdaArray, "Illegal destination memory space" );
    unsigned int srcIndex = srcDevice->allDeviceListIndex();
    RT_ASSERT_MSG( src->m_resourceKind[srcIndex] == MResources::LwdaArray, "Illegal source memory space" );

    const lwca::MipmappedArray& dstArray = dst->m_lwdaArrays[dstIndex];
    const lwca::MipmappedArray& srcArray = src->m_lwdaArrays[srcIndex];
    LWDADevice*                 dstLwda  = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );
    LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );

    for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
    {
        const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

        llog( k_mmll.get() ) << " - copyArraytoArray bytes: " << levelSizeInBytes
                             << ", src: " << srcDevice->allDeviceListIndex()
                             << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

        const size_t elemSizeInBytes = dims.elementSize();
        const size_t width           = dims.levelWidth( level );
        const size_t height          = dims.levelHeight( level );
        const size_t depth           = dims.levelDepth( level );

        LWDA_MEMCPY3D_PEER copyParam;
        std::memset( &copyParam, 0, sizeof( copyParam ) );

        copyParam.srcMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.srcArray      = srcArray.getLevel( level ).get();
        copyParam.srcContext    = srcLwda->lwdaContext().get();

        copyParam.dstMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.dstArray      = dstArray.getLevel( level ).get();
        copyParam.dstContext    = dstLwda->lwdaContext().get();

        copyParam.WidthInBytes = width * elemSizeInBytes;
        copyParam.Height       = height;
        copyParam.Depth        = depth;

        // Issue the copy
        lwca::memcpy3DPeerAsync( &copyParam, dstLwda->primaryStream() );
    }
}

void ResourceManager::copyArraytoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    unsigned int srcIndex = srcDevice->allDeviceListIndex();
    RT_ASSERT_MSG( src->m_resourceKind[srcIndex] == MResources::LwdaArray || src->m_resourceKind[srcIndex] == MResources::LwdaArrayP2P,
                   "Illegal source memory space" );

    unsigned int dstIndex = dstDevice->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[dstIndex] == MResources::LwdaMalloc || dst->m_resourceKind[dstIndex] == MResources::LwdaMallocP2P,
                   "Illegal destination memory space" );

    const lwca::MipmappedArray& srcArray = src->m_lwdaArrays[srcIndex];
    LWDADevice*                 srcLwda  = deviceCast<LWDADevice>( srcDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );

    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    RT_ASSERT_MSG( dstAccess.getKind() == MAccess::LINEAR || dstAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR,
                   "Illegal destination memory space" );

    LWDADevice* dstLwda = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );

    for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
    {
        const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

        llog( k_mmll.get() ) << " - copyArraytoD bytes: " << levelSizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                             << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

        const size_t elemSizeInBytes = dims.elementSize();
        const size_t width           = dims.levelWidth( level );
        const size_t height          = dims.levelHeight( level );
        const size_t depth           = dims.levelDepth( level );

        LWDA_MEMCPY3D_PEER copyParam;
        std::memset( &copyParam, 0, sizeof( copyParam ) );

        copyParam.srcMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.srcArray      = srcArray.getLevel( level ).get();
        copyParam.srcContext    = srcLwda->lwdaContext().get();

        copyParam.dstMemoryType = LW_MEMORYTYPE_DEVICE;
        if( dstAccess.getKind() == MAccess::LINEAR )
        {
            copyParam.dstDevice = (LWdeviceptr)dstAccess.getLinearPtr();
            copyParam.dstPitch  = dims.getLevelNaturalPitchInBytes( level );
        }
        else
        {
            copyParam.dstDevice = (LWdeviceptr)dstAccess.getPitchedLinear( level ).ptr;
            copyParam.dstPitch  = dstAccess.getPitchedLinear( level ).pitch;
        }
        copyParam.dstContext = dstLwda->lwdaContext().get();

        copyParam.WidthInBytes = width * elemSizeInBytes;
        copyParam.Height       = height;
        copyParam.Depth        = depth;

        // Issue the copy
        lwca::memcpy3DPeerAsync( &copyParam, srcLwda->primaryStream() );
    }
}

void ResourceManager::copyDtoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    unsigned int dstIndex = dstDevice->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[dstIndex] == MResources::LwdaArray || dst->m_resourceKind[dstIndex] == MResources::LwdaArrayP2P,
                   "Illegal destination memory space" );

    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );
    RT_ASSERT_MSG( srcAccess.getKind() == MAccess::LINEAR || srcAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR,
                   "Illegal source memory space" );

    const lwca::MipmappedArray& dstArray = dst->m_lwdaArrays[dstIndex];
    LWDADevice*                 dstLwda  = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );
    dstLwda->makeLwrrent();

    LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal destination device" );

    for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
    {
        const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

        llog( k_mmll.get() ) << " - copyDtoArray bytes: " << levelSizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                             << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

        const size_t elemSizeInBytes = dims.elementSize();
        const size_t width           = dims.levelWidth( level );
        const size_t height          = dims.levelHeight( level );
        const size_t depth           = dims.levelDepth( level );

        LWDA_MEMCPY3D_PEER copyParam;
        std::memset( &copyParam, 0, sizeof( copyParam ) );

        copyParam.srcMemoryType = LW_MEMORYTYPE_DEVICE;
        if( srcAccess.getKind() == MAccess::LINEAR )
        {
            copyParam.srcDevice = ( LWdeviceptr )( srcAccess.getLinearPtr() + dims.getLevelOffsetInBytes( level ) );
            copyParam.srcPitch  = dims.getLevelNaturalPitchInBytes( level );
        }
        else
        {
            copyParam.srcDevice = (LWdeviceptr)srcAccess.getPitchedLinear( level ).ptr;
            copyParam.srcPitch  = srcAccess.getPitchedLinear( level ).pitch;
        }

        copyParam.srcHeight  = height;
        copyParam.srcContext = srcLwda->lwdaContext().get();

        copyParam.dstMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.dstArray      = dstArray.getLevel( level ).get();
        copyParam.dstContext    = dstLwda->lwdaContext().get();

        copyParam.WidthInBytes = width * elemSizeInBytes;
        copyParam.Height       = height;
        copyParam.Depth        = depth;

        // Issue the copy
        lwca::memcpy3DPeerAsync( &copyParam, dstLwda->primaryStream() );
    }
}

void ResourceManager::copyDtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );

    if( dstAccess.getKind() == MAccess::LINEAR )
    {
        copyDtoH_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), srcDevice, dims );
    }
    else
    {
        // This can happen if a layered buffer with RT_BUFFER_MAP_WRITE_DISCARD flag set
        // is mapped again without already having a TextureSample attached.
        // In that case it is copied back from global memory into non linear host memory.
        // This could be simplified by copying all data to level 0 of the host buffer using lwca::memcpyDtoH
        // since the levels are actually in one host allocation, but that may change.
        LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
        RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );
        srcLwda->makeLwrrent();
        RT_ASSERT( srcAccess.getKind() == MAccess::LINEAR );  // Would have called copyArrayToH otherwise
        for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
        {
            const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

            llog( k_mmll.get() ) << " - copyDtoArray bytes: " << levelSizeInBytes
                                 << ", src: " << srcDevice->allDeviceListIndex()
                                 << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

            const size_t  elemSizeInBytes = dims.elementSize();
            const size_t  width           = dims.levelWidth( level );
            const size_t  height          = dims.levelHeight( level );
            const size_t  depth           = dims.levelDepth( level );
            LWDA_MEMCPY3D copyParam;
            std::memset( &copyParam, 0, sizeof( copyParam ) );

            copyParam.srcMemoryType = LW_MEMORYTYPE_DEVICE;
            copyParam.srcDevice     = ( LWdeviceptr )( srcAccess.getLinearPtr() + dims.getLevelOffsetInBytes( level ) );
            copyParam.srcPitch      = dims.getLevelNaturalPitchInBytes( level );

            copyParam.srcHeight = height;

            copyParam.dstMemoryType = LW_MEMORYTYPE_HOST;
            copyParam.dstHost       = dstAccess.getPitchedLinear( level ).ptr;
            copyParam.dstPitch      = dstAccess.getPitchedLinear( level ).pitch;
            copyParam.WidthInBytes  = width * elemSizeInBytes;
            copyParam.Height        = height;
            copyParam.Depth         = depth;

            // Issue the copy
            lwca::memcpy3DAsync( &copyParam, srcLwda->primaryStream() );
        }
    }
}


void ResourceManager::copyArraytoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    unsigned int srcIndex = srcDevice->allDeviceListIndex();
    RT_ASSERT_MSG( src->m_resourceKind[srcIndex] == MResources::LwdaArray || src->m_resourceKind[srcIndex] == MResources::LwdaArrayP2P,
                   "Illegal source memory space" );
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    RT_ASSERT_MSG( dstAccess.getKind() == MAccess::LINEAR || dstAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR,
                   "Illegal destination memory space" );

    const lwca::MipmappedArray& srcArray = src->m_lwdaArrays[srcIndex];
    LWDADevice*                 srcLwda  = deviceCast<LWDADevice>( srcDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );
    srcLwda->makeLwrrent();

    for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
    {
        const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

        llog( k_mmll.get() ) << " - copyArraytoH bytes: " << levelSizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                             << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

        const size_t elemSizeInBytes = dims.elementSize();
        const size_t width           = dims.levelWidth( level );
        const size_t height          = dims.levelHeight( level );
        const size_t depth           = dims.levelDepth( level );

        LWDA_MEMCPY3D copyParam;
        std::memset( &copyParam, 0, sizeof( copyParam ) );

        copyParam.srcMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.srcArray      = srcArray.getLevel( level ).get();

        copyParam.dstMemoryType = LW_MEMORYTYPE_HOST;
        copyParam.dstHost =
            dstAccess.getKind() == MAccess::LINEAR ? dstAccess.getLinearPtr() : dstAccess.getPitchedLinear( level ).ptr;
        copyParam.dstPitch = dstAccess.getKind() == MAccess::LINEAR ? dims.getLevelNaturalPitchInBytes( level ) :
                                                                      dstAccess.getPitchedLinear( level ).pitch;
        copyParam.WidthInBytes = width * elemSizeInBytes;
        copyParam.Height       = height;
        copyParam.Depth        = depth;

        // Issue the copy
        lwca::memcpy3DAsync( &copyParam, srcLwda->primaryStream() );
    }
}


void ResourceManager::copyDtoH_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );

    // Compute the pointer as an offset from the base of the backing store
    const MAccess backingAccess = m_texHeapBacking->getAccess( srcDevice );
    size_t        offset        = *src->m_texHeapAllocation * m_texHeapBacking->getDimensions().elementSize();
    char*         srcPtr        = offset + backingAccess.getLinearPtr();

    copyDtoH_common( dstAccess.getLinearPtr(), dstDevice, srcPtr, srcDevice, dims );
}


void ResourceManager::copyDtoH_zc( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_FAIL_MSG( "copyDtoH_zc" );
}


void ResourceManager::copyHtoD_common( char* dstPtr, Device* dstDevice, char* srcPtr, Device* srcDevice, const BufferDimensions& dims )
{
    size_t sizeInBytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - copyHtoD bytes: " << sizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)srcPtr << " dst: " << dstDevice->allDeviceListIndex()
                         << "/ptr=" << (void*)dstPtr << '\n';

    LWDADevice* dstLwda = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );
    dstLwda->makeLwrrent();

    // NOTE: not using pinned, so not really async
    lwca::memcpyHtoDAsync( (LWdeviceptr)dstPtr, srcPtr, sizeInBytes, dstLwda->primaryStream() );
}


void ResourceManager::copyDtoH_common( char* dstPtr, Device* dstDevice, char* srcPtr, Device* srcDevice, const BufferDimensions& dims )
{
    size_t sizeInBytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - copyDtoH bytes: " << sizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                         << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    LWDADevice* srcLwda = deviceCast<LWDADevice>( srcDevice );
    RT_ASSERT_MSG( srcLwda != nullptr, "Illegal source device" );
    srcLwda->makeLwrrent();

    // SGP NOTE: Should be async copy - need to figure out how to implement it
    lwca::memcpyDtoH( dstPtr, (LWdeviceptr)srcPtr, sizeInBytes );
}


void ResourceManager::copyHtoD( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );

    if( srcAccess.getKind() == MAccess::LINEAR )
    {
        copyHtoD_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), srcDevice, dims );
    }
    else
    {
        // This may happen if a layered RT_BUFFER_DISCARD_HOST_MEMORY buffer is unmapped
        // without having a texture sample attached. In that case it will be copied from
        // non-linear host memory to global device memory.
        // This could be simplified by copying the complete size from level 0 using lwca::memcpyHtoDAsync
        // since the levels are actually linear inside a single host allocation
        // but that may change in the future.
        LWDADevice* dstLwda = deviceCast<LWDADevice>( dstDevice );
        RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );
        dstLwda->makeLwrrent();

        RT_ASSERT( dstAccess.getKind() == MAccess::LINEAR );  // Would have called copyHtoArray otherwise

        for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
        {
            const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

            llog( k_mmll.get() ) << " - copyHtoArray bytes: " << levelSizeInBytes
                                 << ", src: " << srcDevice->allDeviceListIndex()
                                 << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

            const size_t elemSizeInBytes = dims.elementSize();
            const size_t width           = dims.levelWidth( level );
            const size_t height          = dims.levelHeight( level );
            const size_t depth           = dims.levelDepth( level );

            LWDA_MEMCPY3D copyParam;
            std::memset( &copyParam, 0, sizeof( copyParam ) );

            copyParam.srcMemoryType = LW_MEMORYTYPE_HOST;
            copyParam.srcHost       = srcAccess.getPitchedLinear( level ).ptr;
            copyParam.srcPitch      = srcAccess.getPitchedLinear( level ).pitch;
            copyParam.srcHeight     = height;

            copyParam.dstMemoryType = LW_MEMORYTYPE_DEVICE;
            copyParam.dstDevice     = ( LWdeviceptr )( dstAccess.getLinearPtr() + dims.getLevelOffsetInBytes( level ) );
            copyParam.dstPitch      = dims.getLevelNaturalPitchInBytes( level );

            copyParam.WidthInBytes = width * elemSizeInBytes;
            copyParam.Height       = height;
            copyParam.Depth        = depth;

            // Issue the copy
            lwca::memcpy3DAsync( &copyParam, dstLwda->primaryStream() );
        }
    }
}


void ResourceManager::copyHtoD_th( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );

    // Compute the pointer as an offset from the base of the backing store
    const MAccess backingAccess = m_texHeapBacking->getAccess( dstDevice );
    size_t        offset        = *dst->m_texHeapAllocation * m_texHeapBacking->getDimensions().elementSize();
    char*         dstPtr        = offset + backingAccess.getLinearPtr();

    copyHtoD_common( dstPtr, dstDevice, srcAccess.getLinearPtr(), srcDevice, dims );
}


void ResourceManager::copyHtoArray( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    unsigned int dstIndex = dstDevice->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[dstIndex] == MResources::LwdaArray || dst->m_resourceKind[dstIndex] == MResources::LwdaArrayP2P,
                   "Illegal destination memory space" );
    const MAccess srcAccess = src->m_buf->getAccess( srcDevice );
    RT_ASSERT_MSG( srcAccess.getKind() == MAccess::LINEAR || srcAccess.getKind() == MAccess::MULTI_PITCHED_LINEAR,
                   "Illegal source memory space" );

    const lwca::MipmappedArray& dstArray = dst->m_lwdaArrays[dstIndex];
    LWDADevice*                 dstLwda  = deviceCast<LWDADevice>( dstDevice );
    RT_ASSERT_MSG( dstLwda != nullptr, "Illegal destination device" );
    dstLwda->makeLwrrent();

    for( unsigned int level = 0; level < dims.mipLevelCount(); ++level )
    {
        const size_t levelSizeInBytes = dims.getLevelSizeInBytes( level );

        llog( k_mmll.get() ) << " - copyHtoArray bytes: " << levelSizeInBytes << ", src: " << srcDevice->allDeviceListIndex()
                             << " dst: " << dstDevice->allDeviceListIndex() << " level: " << level << '\n';

        const size_t elemSizeInBytes = dims.elementSize();
        const size_t width           = dims.levelWidth( level );
        const size_t height          = dims.levelHeight( level );
        const size_t depth           = dims.levelDepth( level );

        LWDA_MEMCPY3D copyParam;
        std::memset( &copyParam, 0, sizeof( copyParam ) );

        copyParam.srcMemoryType = LW_MEMORYTYPE_HOST;
        copyParam.srcHost =
            srcAccess.getKind() == MAccess::LINEAR ? srcAccess.getLinearPtr() : srcAccess.getPitchedLinear( level ).ptr;
        copyParam.srcPitch = srcAccess.getKind() == MAccess::LINEAR ? dims.getLevelNaturalPitchInBytes( level ) :
                                                                      srcAccess.getPitchedLinear( level ).pitch;
        copyParam.srcHeight = height;

        copyParam.dstMemoryType = LW_MEMORYTYPE_ARRAY;
        copyParam.dstArray      = dstArray.getLevel( level ).get();
        copyParam.WidthInBytes  = width * elemSizeInBytes;
        copyParam.Height        = height;
        copyParam.Depth         = depth;

        // Issue the copy
        lwca::memcpy3DAsync( &copyParam, dstLwda->primaryStream() );
    }
}

void ResourceManager::copyHtoP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyHtoP2P bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    DeviceSet island     = m_deviceManager->getLwlinkIsland( dstDevice );
    DeviceSet allocOwner = dst->m_p2pAllocatedSet & island;
    RT_ASSERT( allocOwner.count() == 1 );

    // Perform the copy only if the destination device is the physical owner of
    // the allocation.  Otherwise, the memory lives on a different device, and
    // since we assume the caller always copies to the entire island (i.e.
    // calls this function for all devices), we skip the copy to avoid
    // duplicate work.
    if( allocOwner.isSet( dstDevice ) )
    {
        const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
        const MAccess srcAccess = src->m_buf->getAccess( srcDevice );
        copyHtoD_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), srcDevice, dims );
    }
}

void ResourceManager::copyP2PtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyP2PtoH bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    // Copy from the device that physically owns the allocation.
    DeviceSet island     = m_deviceManager->getLwlinkIsland( srcDevice );
    DeviceSet allocOwner = src->m_p2pAllocatedSet & island;
    RT_ASSERT( allocOwner.count() == 1 );
    const int srcDevIdx     = allocOwner[0];
    Device*   realSrcDevice = m_deviceManager->allDevices()[srcDevIdx];

    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    const MAccess srcAccess = src->m_buf->getAccess( realSrcDevice );
    copyDtoH_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), realSrcDevice, dims );
}

void ResourceManager::copyP2PtoP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyP2PtoP2P bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    DeviceSet srcIsland = m_deviceManager->getLwlinkIsland( srcDevice );
    DeviceSet dstIsland = m_deviceManager->getLwlinkIsland( dstDevice );

    // Nothing to do if we're in the same island
    if( srcIsland == dstIsland )
        return;

    // Find the devices on which the allocations actually live
    DeviceSet srcAllocOwner = src->m_p2pAllocatedSet & srcIsland;
    DeviceSet dstAllocOwner = dst->m_p2pAllocatedSet & dstIsland;
    RT_ASSERT( srcAllocOwner.count() == 1 );
    RT_ASSERT( dstAllocOwner.count() == 1 );
    const int srcDevIdx     = srcAllocOwner[0];
    const int dstDevIdx     = dstAllocOwner[0];
    Device*   realSrcDevice = m_deviceManager->allDevices()[srcDevIdx];
    Device*   realDstDevice = m_deviceManager->allDevices()[dstDevIdx];

    copyDtoD( dst, realDstDevice, src, realSrcDevice, dims );
}

void ResourceManager::copyArrayP2PtoArrayP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyArrayP2PtoArrayP2P bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    DeviceSet srcIsland = m_deviceManager->getLwlinkIsland( srcDevice );
    DeviceSet dstIsland = m_deviceManager->getLwlinkIsland( dstDevice );

    // Nothing to do if we're in the same island
    if( srcIsland == dstIsland )
        return;

    DeviceSet srcAllocOwner = src->m_p2pAllocatedSet & srcIsland;
    DeviceSet dstAllocOwner = dst->m_p2pAllocatedSet & dstIsland;
    RT_ASSERT( srcAllocOwner.count() == 1 );
    RT_ASSERT( dstAllocOwner.count() == 1 );
    const int srcDevIdx     = srcAllocOwner[0];
    const int dstDevIdx     = dstAllocOwner[0];
    Device*   realSrcDevice = m_deviceManager->allDevices()[srcDevIdx];
    Device*   realDstDevice = m_deviceManager->allDevices()[dstDevIdx];

    copyArraytoArray( dst, realDstDevice, src, realSrcDevice, dims );
}

void ResourceManager::copyHtoArrayP2P( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyHtoArrayP2P bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    DeviceSet island     = m_deviceManager->getLwlinkIsland( dstDevice );
    DeviceSet allocOwner = dst->m_p2pAllocatedSet & island;
    RT_ASSERT( allocOwner.count() == 1 );

    // Perform the copy only if the destination device is the physical owner of
    // the allocation.  Otherwise, the memory lives on a different device, and
    // since we assume the caller always copies to the entire island (i.e.
    // calls this function for all devices), we skip the copy to avoid
    // duplicate work.
    if( allocOwner.isSet( dstDevice ) )
    {
        copyHtoArray( dst, dstDevice, src, srcDevice, dims );
    }
}

void ResourceManager::copySCtoH( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copySCtoH bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    // Copy from the device that physically owns the allocation.
    DeviceSet allocOwner = src->m_singleCopyAllocatedSet;
    RT_ASSERT( allocOwner.count() == 1 );
    const int srcDevIdx     = allocOwner[0];
    Device*   realSrcDevice = m_deviceManager->allDevices()[srcDevIdx];

    llog( k_mmll.get() ) << " - performing copy from device: " << srcDevIdx;
    const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
    const MAccess srcAccess = src->m_buf->getAccess( realSrcDevice );
    copyDtoH_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), realSrcDevice, dims );
}

void ResourceManager::copyHtoSC( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    llog( k_mmll.get() ) << " - copyHtoSC bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';

    DeviceSet allocOwner = dst->m_singleCopyAllocatedSet;
    RT_ASSERT( allocOwner.count() == 1 );

    // Perform the copy only if the destination device is the physical owner of
    // the allocation.  Otherwise, the memory lives on a different device, and
    // since we assume the caller always copies to all devices, we skip the copy
    // to avoid duplicate work.
    if( allocOwner.isSet( dstDevice ) )
    {
        llog( k_mmll.get() ) << " - performing copy to device: " << allocOwner[0];
        const MAccess dstAccess = dst->m_buf->getAccess( dstDevice );
        const MAccess srcAccess = src->m_buf->getAccess( srcDevice );
        copyHtoD_common( dstAccess.getLinearPtr(), dstDevice, srcAccess.getLinearPtr(), srcDevice, dims );
    }
}

void ResourceManager::copySCtoSC( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    // Always a no-op, because we maintain a single copy across devices.
    llog( k_mmll.get() ) << " - copySCtoSC bytes: " << dims.getTotalSizeInBytes()
                         << ", src: " << srcDevice->allDeviceListIndex() << " dst: " << dstDevice->allDeviceListIndex() << '\n';
}

void ResourceManager::copyZeroCopy( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    if( dst == src )
        return;  // Trivial case - zerocopy is always synchronized within a device

    RT_ASSERT_FAIL_MSG( "forward zeroCopy to cpDtoH, cpHtoH, cpDtoD or cpHtoD" );
}


void ResourceManager::copyHtoD_zc( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_FAIL_MSG( "copyHtoD_zc" );
}


void ResourceManager::copyTwoStage( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_FAIL_MSG( "copyTwoStage" );
}


void ResourceManager::copyIlwalid( MResources* dst, Device* dstDevice, MResources* src, Device* srcDevice, const BufferDimensions& dims )
{
    RT_ASSERT_FAIL_MSG( "Illegal resource copy" );
}
