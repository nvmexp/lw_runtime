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

#include <ExelwtionStrategy/Common/Specializations.h>
#include <ExelwtionStrategy/RTX/GenericCache.h>
#include <ExelwtionStrategy/RTX/RTXCompile.h>

#include <rtcore/interface/types.h>

#include <memory>
#include <mutex>

namespace optix {
class CanonicalProgram;
class LWDADevice;
class RTCore;


struct CompiledProgramCacheKey
{
    using AttributeDecoderList = std::vector<std::string>;
    using SpecializationsMap   = std::map<std::string, VariableSpecialization>;
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //

    const LWDADevice*         device = nullptr;
    RTXCompile::CompileParams compileParams;
    RtcCompileOptions         rtcoreCompileOptions;
    AttributeDecoderList      attributeDecoderUUNames;
    SpecializationsMap        specializations;
    std::set<std::string>     heavyweightCallSiteNames;
    SemanticType              stype                      = ST_ILWALID;
    SemanticType              callerStype                = ST_ILWALID;
    const CanonicalProgram*   cp                         = nullptr;
    int                       dimensionality             = 0;
    int                       numConlwrrentLaunchDevices = 0;
    PagingMode                pagingMode                 = PagingMode::UNKNOWN;

    CompiledProgramCacheKey();
    CompiledProgramCacheKey( const LWDADevice*                device,
                             const RTXCompile::CompileParams& compileParams,
                             const RtcCompileOptions&         rtcOptions,
                             const AttributeDecoderList&      attributeDecoderUUNames,
                             const SpecializationsMap&        specializations,
                             const std::set<std::string>&     heavyweightCallSites,
                             SemanticType                     stype,
                             SemanticType                     callerStype,
                             const CanonicalProgram*          cp,
                             int                              dimensionality,
                             int                              numConlwrrentLaunchDevices,
                             PagingMode                       pagingMode );

    bool operator<( const CompiledProgramCacheKey& other ) const;
};

// Persistence support
void readOrWrite( PersistentStream* stream, RtcCompileOptions* options, const char* label );
void readOrWrite( PersistentStream* stream, CompiledProgramCacheKey* key, const char* label );

typedef std::shared_ptr<RtcCompiledModule_t>    CompiledModuleHandle;
#if RTCORE_API_VERSION >= 25
typedef std::pair<CompiledModuleHandle, Rtlw32> ModuleEntryRefPair;
#else
typedef std::pair<CompiledModuleHandle, std::string> ModuleEntryRefPair;
#endif
class CompiledProgramCache
{
    // The hash returned from rtcore isn't unique to the device, so we need to encorporate
    // that into the cache key. Since we also need to remove stuff by device, it made
    // sence to just separate out the caches by device first. This made it easy to remove
    // by device as well as separate the m_cacheByRtcModuleHash hashes.
    struct Cache
    {
        std::map<CompiledProgramCacheKey, ModuleEntryRefPair> m_cacheByKey;
        std::map<Rtlw64, CompiledModuleHandle>                m_cacheByRtcModuleHash;
    };
    std::map<const LWDADevice*, Cache> m_cache;
    mutable std::mutex                 m_mutex;

  public:
    bool find( const LWDADevice* device, Rtlw64 hash, CompiledModuleHandle& handle ) const;
    bool find( CompiledProgramCacheKey key, ModuleEntryRefPair& handle ) const;
    void emplace( const LWDADevice* device, Rtlw64 hash, CompiledModuleHandle& handle );
    void emplace( const CompiledProgramCacheKey& key, ModuleEntryRefPair& module );
    void removeProgramsForDevice( const LWDADevice* device );
};
}
