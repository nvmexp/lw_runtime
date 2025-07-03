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

// NOTE: Implementation dependent header
//       MBufferPolicyDetails is an implementation detail of MemoryManager.
//       It exists as a separate header with the sole purpose of being used
//       by test_MemoryManager to check behavior against the defined policies.
//       DO NOT USE OUTSIDE MEMORYMANAGER OR TEST_MEMORYMANAGER.
//       SGP: Using it in test_memoryManager is also a bad idea
//       because it will tie the tests to a specific
//       implementation. It would be better to add specific query
//       functions for higher level concepts.

#include <Memory/MBufferPolicy.h>


namespace optix {

class PolicyIndexMap;

struct PolicyDetails
{
    static const unsigned int SINGLEGPU = 0x1;
    static const unsigned int MULTIGPU  = 0x2;
    static const unsigned int ALL       = 0x3;
    enum Access
    {
        R,
        W,
        RW,
        N
    };
    enum CpuAllocationKind
    {
        CPU_NONE,
        CPU_MALLOC,
        CPU_ZEROCOPY,
        CPU_PREFER_SINGLE,
    };
    enum LwdaAllocationKind
    {
        LWDA_NONE,
        LWDA_GLOBAL,
        LWDA_ZEROCOPY,
        LWDA_PREFER_ARRAY,
        LWDA_PREFER_TEX_HEAP,
        LWDA_PREFER_SINGLE,
        LWDA_DEMAND_LOAD,            // For demand load buffers.
        LWDA_DEMAND_LOAD_TILE_ARRAY, // For demand load texture tile arrays.
        LWDA_PREFER_SPARSE,          // Prefer hardware sparse textures, fall back to software and use mip tail 
        LWDA_SPARSE_BACKING,         // Sparse texture backing storage
    };
    enum InteropMode
    {
        NOINTEROP,
        DIRECT,
        INDIRECT
    };
    enum LwdaTextureKind
    {
        TEX_NONE,
        TEX_ANY,
        TEX_REF
    };
    enum Hook
    {
        NOHOOK,
        GFXINTEROP,
        LWDAINTEROP
    };

    unsigned int  variant;
    MBufferPolicy policyKind;

    // Device access policy
    Access activeDeviceAccess;
    Access hostAccess;

    // Host allocation control
    CpuAllocationKind cpuAllocation;

    // General launch mapping policies
    bool launchRequiresValidCopy;
    bool launchIlwalidatesOtherDevices;
    Hook accessHook;
    bool copyOnDirty;

    // LWCA allocation control
    LwdaAllocationKind lwdaAllocation;
    InteropMode        interopMode;
    LwdaTextureKind    lwdaTextureKind;

    bool clearOnAlloc;
    bool allowsAttachedTextureSamplers;
    bool allowsRawAccess;
    bool isBackingStore;
    bool allowsP2P;
    bool discardHostMemoryOnUnmap;

    // Helpers
    static size_t               getNumberOfPolicies();
    static const PolicyDetails& getPolicyDetails( MBufferPolicy policyKind, unsigned int variant );

    bool allowsHostReadAccess() const;
    bool allowsHostWriteAccess() const;
    bool allowsActiveDeviceReadAccess() const;
    bool allowsActiveDeviceWriteAccess() const;

  private:
    friend class PolicyIndexMap;
    static const PolicyDetails policies[];
};

class PolicyIndexMap
{
  public:
    PolicyIndexMap();
    unsigned int getPolicyIndex( MBufferPolicy policy, unsigned int variant );
    const PolicyDetails* getPolicyDetailsPtr( MBufferPolicy policyKind, unsigned int variant );

  private:
    static const unsigned int VARIANT_MULTIPLIER    = PolicyDetails::ALL + 1;
    static const unsigned int MAX_POLICY_KINDS      = static_cast<int>( MBufferPolicy::unused ) + 1;
    static const unsigned int POLICY_INDEX_MAP_SIZE = VARIANT_MULTIPLIER * MAX_POLICY_KINDS;

    const PolicyDetails* policyIndexMap[POLICY_INDEX_MAP_SIZE];
};
}
