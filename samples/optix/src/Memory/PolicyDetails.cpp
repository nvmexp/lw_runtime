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


#include <Memory/PolicyDetails.h>

#include <prodlib/exceptions/Assert.h>

using namespace optix;


//
// Policies are one of the central design concepts in the memory
// manager. As such, please keep this table tidy, so that it becomes a
// standard place to refer for understanding the semantics. Note: No
// code in memory manager should ever refer to a specific
// policy. Instead, add a new column to the table or adapt an existing
// one.
//

// clang-format off
const PolicyDetails PolicyDetails::policies[] = {
  //                                                                                                                                                                        clearOnAlloc                          discardHostMemoryOnUnmap
  //                                                             activeDeviceAccess         launchRequiresValidCopy                                                         |      allowsAttachedTextureSamplers  |
  //                                                             |   hostAccess             |      launchIlwalidatesOtherDevices                                            |      |      allowsRawAccess         |
  //                                                             |   |   cpuAllocation      |      |      accessHook   copyOnDirty                                lwdaTextureKind  |      |      isBackingStore   |
  //variant    policyKind                                        |   |   |                  |      |      |            |      lwdaAllocation          interopMode |         |      |      |      |      allowsP2P |
  { ALL,       MBufferPolicy::readonly,                          R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, false, false, true,     false },
  { ALL,       MBufferPolicy::readonly_discard_hostmem,          R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, false, false, true,     true  },
  { SINGLEGPU, MBufferPolicy::readwrite,                         RW, RW, CPU_MALLOC,        true,  true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, false, false, false,    false },
  { MULTIGPU,  MBufferPolicy::readwrite,                         RW, RW, CPU_PREFER_SINGLE, true,  true,  NOHOOK,      false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, false, false, false,    false },
  { SINGLEGPU, MBufferPolicy::writeonly,                         W,  RW, CPU_MALLOC,        false, true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, false, false, false,    false },
  { MULTIGPU,  MBufferPolicy::writeonly,                         W,  RW, CPU_PREFER_SINGLE, false, true,  NOHOOK,      false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, false, false, false,    false },

  { ALL,       MBufferPolicy::readonly_raw,                      R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { ALL,       MBufferPolicy::readonly_discard_hostmem_raw,      R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    true  },
  { SINGLEGPU, MBufferPolicy::readwrite_raw,                     RW, RW, CPU_MALLOC,        true,  true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::readwrite_raw,                     RW, RW, CPU_PREFER_SINGLE, true,  true,  NOHOOK,      false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::writeonly_raw,                     W,  RW, CPU_MALLOC,        false, true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::writeonly_raw,                     W,  RW, CPU_PREFER_SINGLE, false, true,  NOHOOK,      false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },

  { ALL,       MBufferPolicy::readonly_lwdaInterop,              R,  RW, CPU_MALLOC,        true,  false, LWDAINTEROP, false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::readwrite_lwdaInterop,             RW, RW, CPU_MALLOC,        true,  true,  LWDAINTEROP, false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::readwrite_lwdaInterop,             RW, RW, CPU_PREFER_SINGLE, true,  true,  LWDAINTEROP, false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::writeonly_lwdaInterop,             W,  RW, CPU_MALLOC,        false, true,  LWDAINTEROP, false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::writeonly_lwdaInterop,             W,  RW, CPU_PREFER_SINGLE, false, true,  LWDAINTEROP, false, LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },

  { ALL,       MBufferPolicy::readonly_lwdaInterop_copyOnDirty,  R,  RW, CPU_MALLOC,        true,  false, LWDAINTEROP, true,  LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::readwrite_lwdaInterop_copyOnDirty, RW, RW, CPU_MALLOC,        true,  true,  LWDAINTEROP, true,  LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::readwrite_lwdaInterop_copyOnDirty, RW, RW, CPU_PREFER_SINGLE, true,  true,  LWDAINTEROP, true,  LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::writeonly_lwdaInterop_copyOnDirty, W,  RW, CPU_MALLOC,        false, true,  LWDAINTEROP, true,  LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::writeonly_lwdaInterop_copyOnDirty, W,  RW, CPU_PREFER_SINGLE, false, true,  LWDAINTEROP, true,  LWDA_PREFER_SINGLE,     NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },

  { ALL,       MBufferPolicy::gpuLocal,                          RW, RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, true,  false, true,  false, false,    false },

  { ALL,       MBufferPolicy::readonly_gfxInterop,               R,  RW, CPU_MALLOC,        true,  false, GFXINTEROP,  false, LWDA_GLOBAL,            DIRECT,     TEX_NONE, false, false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::readwrite_gfxInterop,              RW, RW, CPU_MALLOC,        true,  true,  GFXINTEROP,  false, LWDA_GLOBAL,            DIRECT,     TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::readwrite_gfxInterop,              RW, RW, CPU_PREFER_SINGLE, true,  true,  GFXINTEROP,  false, LWDA_PREFER_SINGLE,     INDIRECT,   TEX_NONE, true,  false, true,  false, false,    false },
  { SINGLEGPU, MBufferPolicy::writeonly_gfxInterop,              W,  RW, CPU_MALLOC,        false, true,  GFXINTEROP,  false, LWDA_GLOBAL,            DIRECT,     TEX_NONE, true,  false, true,  false, false,    false },
  { MULTIGPU,  MBufferPolicy::writeonly_gfxInterop,              W,  RW, CPU_PREFER_SINGLE, false, true,  GFXINTEROP,  false, LWDA_PREFER_SINGLE,     INDIRECT,   TEX_NONE, true,  false, true,  false, false,    false },

  { ALL,       MBufferPolicy::texture_gfxInterop,                R,  RW, CPU_MALLOC,        true,  false, GFXINTEROP,  false, LWDA_PREFER_ARRAY,      DIRECT,     TEX_ANY,  false, true,  true,  false, false,    false },

  { ALL,       MBufferPolicy::texture_array,                     R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_PREFER_ARRAY,      NOINTEROP,  TEX_ANY,  false, true,  true,  false, true,     false },
  { ALL,       MBufferPolicy::texture_array_discard_hostmem,     R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_PREFER_ARRAY,      NOINTEROP,  TEX_ANY,  false, true,  true,  false, true,     true  },
  { ALL,       MBufferPolicy::texture_linear,                    R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_ANY,  false, true,  true,  false, false,    false },
  { ALL,       MBufferPolicy::texture_linear_discard_hostmem,    R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_ANY,  false, true,  true,  false, false,    true },

  { ALL,       MBufferPolicy::internal_readonly,                 R,  W,  CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { ALL,       MBufferPolicy::internal_readwrite,                RW, RW, CPU_MALLOC,        true,  true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },

  { ALL,       MBufferPolicy::internal_hostonly,                 N,  RW, CPU_MALLOC,        false, false, NOHOOK,      false, LWDA_NONE,              NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { ALL,       MBufferPolicy::internal_readonly_deviceonly,      R,  N,  CPU_NONE,          false, false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },

  { ALL,       MBufferPolicy::internal_readonly_manualSync,      R,  RW, CPU_MALLOC,        false, false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
  { ALL,       MBufferPolicy::internal_readwrite_manualSync,     RW, RW, CPU_MALLOC,        false, true,  NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },

  { ALL,       MBufferPolicy::internal_texheapBacking,           R,  N,  CPU_NONE,          false, false, NOHOOK,      false, LWDA_GLOBAL,            NOINTEROP,  TEX_REF,  false, true,  true,  true,  false,    false },
  { ALL,       MBufferPolicy::internal_preferTexheap,            R,  RW, CPU_MALLOC,        true,  false, NOHOOK,      false, LWDA_PREFER_TEX_HEAP,   NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },

  // TODO: Allow allowsP2P to be true?
  { ALL,       MBufferPolicy::readonly_demandload,               R,  N,  CPU_NONE,          true,  false, NOHOOK,      false, LWDA_DEMAND_LOAD,       NOINTEROP,  TEX_NONE, false, false, false, false, false,    false },
  { ALL,       MBufferPolicy::texture_readonly_demandload,       R,  N,  CPU_NONE,          true,  false, NOHOOK,      false, LWDA_PREFER_SPARSE,
                                                                                                                                                      NOINTEROP,  TEX_ANY,  false, true,  false, false, false,    false },
  { ALL,       MBufferPolicy::tileArray_readOnly_demandLoad,     R,  N,  CPU_NONE,          true,  false, NOHOOK,      false, LWDA_DEMAND_LOAD_TILE_ARRAY,
                                                                                                                                                      NOINTEROP,  TEX_ANY,  false, true,  false, false, false,    false },
  { ALL,       MBufferPolicy::readonly_sparse_backing,           R,  N,  CPU_NONE,          true,  false, NOHOOK,      false, LWDA_SPARSE_BACKING,
                                                                                                                                                      NOINTEROP,  TEX_ANY,  false, true,  false, false, false,    false },

  { ALL,       MBufferPolicy::unused,                            N,  RW, CPU_MALLOC,        false, false, NOHOOK,      false, LWDA_NONE,              NOINTEROP,  TEX_NONE, false, false, true,  false, false,    false },
    // clang-format on

    //
    // General notes:
    //    - "Read"/"write" in the policy name refer to access from the device point of view.
    //    - A "none" allocation policy means that acquire will not allocate on the respective
    //      device type, even if a device of that type is in the allowed set.
    //    - "Internal" means internal to Optix, not internal to the memory manager.
    //    - "Unused" means unused during kernel launch time. It should have no GPU allocations.
    //
    // Fields:
    //
    // activeDeviceAccess
    //    The access permissions that active devices of any kind have to the
    //    buffer during launch.
    //
    // hostAccess
    //    Permissions for mapping a buffer to host. This applies to external
    //    clients only, i.e. anything outside the memory manager. The memory
    //    manager itself can map any buffer as long as there is a valid CPU
    //    allocation policy.
    //
    // copyOnDirty
    //    Whether the buffer was created with RT_BUFFER_COPY_ON_DIRTY for
    //    LWCA interop.
    //
    // clearOnAlloc
    //    If set, will zero out memory after an allocation. If the memory manager
    //    isn't the one performing the allocation (e.g. native interop scenarios),
    //    then this has no effect.
    //    Note that the OptiX API never specifies that buffers are initialized
    //    anywhere. This flag mostly exists to make our lives easier for testing,
    //    where uninitialized output buffers can be quite a nuisance. Clearing isn't
    //    free if there are many buffers, so we do it selectively (in general on
    //    output only) instead of simply everywhere.
    //
    // allowsRawAccess
    //    If true, forces the memory manager to allocate the storage in a
    //    contiguous address range. Also makes sure the allocation is static
    //    for the duration of an OptiX launch (e.g. if an OptiX launch consists
    //    of multiple LWCA launches, the mem manager isn't allowed to move the
    //    allocation around between LWCA launches). This enables "raw" access
    //    to the storage, i.e. directly through a pointer or texture, without
    //    having to go through the runtime functions that may remap them (like
    //    rt_buffer_get) for each access. Frozen pointers must have raw access.
    //
    // isBackingStore
    //    Whether the buffer is to be used as a backing store for other buffers.
    //
    // allowsP2P
    //    Whether the memory manager is allowed to move (parts of) a buffer to
    //    a different device and access it through peer-to-peer, given capable
    //    hardware.
    //
    // accessHook
    //    Graphics interop can be indirect, foreign, immediate.
    //    Indirect interop uses intermediate zero copy memory in case of output buffer and MGPU
    //    Foreign interop uses intermediate host memory if there is no LWCA context on graphics devices
    //    Immediate means that there is no intermediate copy.
    //
};

static PolicyIndexMap g_policyIndexMap;

PolicyIndexMap::PolicyIndexMap()
{
    unsigned int numPolicies = PolicyDetails::getNumberOfPolicies();
    for( unsigned int i = 0; i < numPolicies; ++i )
    {
        const PolicyDetails* policy  = &PolicyDetails::policies[i];
        const unsigned int   variant = policy->variant;
        for( unsigned int v = 1; v <= variant; v <<= 1 )
        {
            if( ( variant & v ) != 0 )
            {
                const unsigned int pindex = getPolicyIndex( policy->policyKind, v );
                policyIndexMap[pindex]    = policy;
            }
        }
    }
}

inline unsigned int PolicyIndexMap::getPolicyIndex( MBufferPolicy policyKind, unsigned int variant )
{
    return variant + VARIANT_MULTIPLIER * static_cast<unsigned int>( policyKind );
}

inline const PolicyDetails* PolicyIndexMap::getPolicyDetailsPtr( MBufferPolicy policyKind, unsigned int variant )
{
    unsigned int pindex = getPolicyIndex( policyKind, variant );
    if( pindex < POLICY_INDEX_MAP_SIZE )
        return policyIndexMap[pindex];
    return nullptr;
}

size_t PolicyDetails::getNumberOfPolicies()
{
    return sizeof( PolicyDetails::policies ) / sizeof( PolicyDetails );
}

const PolicyDetails& PolicyDetails::getPolicyDetails( MBufferPolicy policyKind, unsigned int variant )
{
    const PolicyDetails* policy = g_policyIndexMap.getPolicyDetailsPtr( policyKind, variant );
    if( policy != nullptr )
        return *policy;

    RT_ASSERT_FAIL_MSG( "Cannot find policy for kind: " + toString( policyKind ) );
}

bool PolicyDetails::allowsHostReadAccess() const
{
    return hostAccess == R || hostAccess == RW;
}

bool PolicyDetails::allowsHostWriteAccess() const
{
    return hostAccess == W || hostAccess == RW;
}

bool PolicyDetails::allowsActiveDeviceReadAccess() const
{
    return activeDeviceAccess == R || activeDeviceAccess == RW;
}

bool PolicyDetails::allowsActiveDeviceWriteAccess() const
{
    return activeDeviceAccess == W || activeDeviceAccess == RW;
}
