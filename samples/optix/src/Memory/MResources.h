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

#include <LWCA/Array.h>
#include <LWCA/GraphicsResource.h>
#include <Device/DeviceSet.h>
#include <Device/MaxDevices.h>
#include <Memory/GfxInteropResource.h>

#include <corelib/misc/Concepts.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace optix {

class LWDADevice;
class Device;
class MAccess;
class MBuffer;

/*
 * Container for low-level resources. There is always exactly one of
 * these per MBuffer, but it is separated into a different class to
 * allow separation of concerns and avoid include pollution.
 */
class MResources : private corelib::NonCopyable
{
  public:
    // This class is only used by ResourceManager and MBuffer but the
    // ctor/dtor need to be public for std::unique_ptr to work.
    MResources( MBuffer* buf );
    ~MResources();

  private:
    // Pointer back to the mbuffer (always valid)
    MBuffer* m_buf = nullptr;

    // Allocated resource kind. Pointers are typically stored in
    // MAccess for most resources.  Warning: if this is extended or
    // the order is changed, then the method table in
    // ResourceCopying.cpp also needs to be updated. In particular,
    // HostMalloc needs to be first.
    enum ResourceKind
    {
        HostMalloc,
        LwdaArray,
        LwdaSparseArray,
        LwdaMalloc,
        TexHeap,
        ZeroCopy,
        LwdaMallocP2P,
        LwdaArrayP2P,
        LwdaMallocSingleCopy,
        DemandLoad,           // demand load buffer
        DemandLoadArray,      // demand load texture mip tail
        DemandLoadTileArray,  // demand load texture tile array
        LwdaSparseBacking,
        None
    };
    ResourceKind m_resourceKind[OPTIX_MAX_DEVICES];

    // Info for LwdaMalloc kind
    DeviceSet m_lwdaMallocExternalSet;  // which of the pointers are externally owned

    // Info for LwdaArray, LwdaArrayP2P, DemandLoadArray and DemandLoadTile kind
    lwca::MipmappedArray m_lwdaArrays[OPTIX_MAX_DEVICES];
    
    // The number of mipmap levels in each LWCA array. Only used to determine
    // which memory bindings to free in sparse textures.
    int m_numMipmapLevels[OPTIX_MAX_DEVICES] = {0};

    // Info for DemandLoad and DemandLoadArray kind
    DeviceSet               m_demandLoadAllocatedSet;
    std::shared_ptr<size_t> m_demandLoadAllocation;

    // Info for DemandLoadArray kind.  The min miplevel is initially UINT_MAX, and the max miplevel is initially zero.
    unsigned int m_demandTextureMinMipLevel[OPTIX_MAX_DEVICES];
    unsigned int m_demandTextureMaxMipLevel[OPTIX_MAX_DEVICES];

    // Info for GfxInterop kind
    GfxInteropResource     m_gfxInteropResource;
    Device*                m_gfxInteropDevice = nullptr;
    unsigned int           m_gfxInteropFBO    = 0;
    lwca::GraphicsResource m_gfxInteropLWDARegisteredResource;
    bool                   m_gfxInteropRegistered = false;  // used only for error checking
    int                    m_gfxInteropMapped     = 0;

    // Info for TexHeap kind
    DeviceSet               m_texHeapAllocatedSet;
    std::shared_ptr<size_t> m_texHeapAllocation;

    // Info for ZeroCopy kind
    DeviceSet   m_zeroCopyAllocatedSet;
    char*       m_zeroCopyHostPtr         = nullptr;
    LWDADevice* m_zeroCopyRegistrarDevice = nullptr;

    // Info for SingleCopy kind.
    DeviceSet m_singleCopyAllocatedSet;
    char*     m_singleCopyPtr = nullptr;

    // Info for LwdaMallocP2P and LwdaArrayP2P kind
    DeviceSet m_p2pAllocatedSet;  // who owns the p2p allocation for each island

    // Helper functions
    void setResource( unsigned int allDeviceIndex, ResourceKind kind, const MAccess& access );
    static std::string toString( ResourceKind kind );

    friend class ResourceManager;
    friend class MBuffer;
};

struct GfxInteropResourceBatch
{
    GfxInteropResourceBatch()
        : resourceData( OPTIX_MAX_DEVICES )
    {
    }
    // Opaque LWCA Graphics interop resources batched per device.
    std::vector<std::vector<void*>> resourceData;
    // Graphics interop buffers.
    std::vector<MBuffer*> bufferData;
};
}
