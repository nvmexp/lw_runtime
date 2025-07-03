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

#include <ExelwtionStrategy/RTX/CompiledProgramCache.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>

#include <Context/RTCore.h>
#include <rtcore/interface/rtcore.h>

#include <Util/optixUuid.h>

using namespace optix;

static bool operator==( const RtcCompileOptions& lhs, const RtcCompileOptions& rhs )
{
    // clang-format off
  return
    lhs.abiVariant                   == rhs.abiVariant &&
    lhs.numPayloadRegisters          == rhs.numPayloadRegisters &&
    lhs.numAttributeRegisters        == rhs.numAttributeRegisters &&
    lhs.numCallableParamRegisters    == rhs.numCallableParamRegisters &&
    lhs.smVersion                    == rhs.smVersion &&
    lhs.maxRegisterCount             == rhs.maxRegisterCount &&
    lhs.optLevel                     == rhs.optLevel &&
    lhs.debugLevel                   == rhs.debugLevel &&
    lhs.exceptionFlags               == rhs.exceptionFlags &&
    lhs.smemSpillPolicy              == rhs.smemSpillPolicy &&
    lhs.targetSharedMemoryBytesPerSM == rhs.targetSharedMemoryBytesPerSM &&
    lhs.traversableGraphFlags        == rhs.traversableGraphFlags;
    // clang-format on
}

static bool operator!=( const RtcCompileOptions& lhs, const RtcCompileOptions& rhs )
{
    return !( lhs == rhs );
}

static bool operator<( const RtcCompileOptions& lhs, const RtcCompileOptions& rhs )
{
    if( lhs.abiVariant != rhs.abiVariant )
        return lhs.abiVariant < rhs.abiVariant;
    if( lhs.numPayloadRegisters != rhs.numPayloadRegisters )
        return lhs.numPayloadRegisters < rhs.numPayloadRegisters;
    if( lhs.numAttributeRegisters != rhs.numAttributeRegisters )
        return lhs.numAttributeRegisters < rhs.numAttributeRegisters;
    if( lhs.numCallableParamRegisters != rhs.numCallableParamRegisters )
        return lhs.numCallableParamRegisters < rhs.numCallableParamRegisters;
    if( lhs.smVersion != rhs.smVersion )
        return lhs.smVersion < rhs.smVersion;
    if( lhs.maxRegisterCount != rhs.maxRegisterCount )
        return lhs.maxRegisterCount < rhs.maxRegisterCount;
    if( lhs.optLevel != rhs.optLevel )
        return lhs.optLevel < rhs.optLevel;
    if( lhs.debugLevel != rhs.debugLevel )
        return lhs.debugLevel < rhs.debugLevel;
    if( lhs.exceptionFlags != rhs.exceptionFlags )
        return lhs.exceptionFlags < rhs.exceptionFlags;
    if( lhs.smemSpillPolicy != rhs.smemSpillPolicy )
        return lhs.smemSpillPolicy < rhs.smemSpillPolicy;
    if( lhs.targetSharedMemoryBytesPerSM != rhs.targetSharedMemoryBytesPerSM )
        return lhs.targetSharedMemoryBytesPerSM < rhs.targetSharedMemoryBytesPerSM;
    if( lhs.traversableGraphFlags != rhs.traversableGraphFlags )
        return lhs.traversableGraphFlags < rhs.traversableGraphFlags;

    return false;
}

CompiledProgramCacheKey::CompiledProgramCacheKey()
{
}

CompiledProgramCacheKey::CompiledProgramCacheKey( const LWDADevice*                device,
                                                  const RTXCompile::CompileParams& compileParams,
                                                  const RtcCompileOptions&         rtcoreCompileOptions,
                                                  const AttributeDecoderList&      attributeDecoderUUNames,
                                                  const SpecializationsMap&        specializations,
                                                  const std::set<std::string>&     heavyweightCallSites,
                                                  SemanticType                     stype,
                                                  SemanticType                     callerStype,
                                                  const CanonicalProgram*          cp,
                                                  int                              dimensionality,
                                                  int                              numConlwrrentLaunchDevices,
                                                  PagingMode                       pagingMode )
    : device( device )
    , compileParams( compileParams )
    , rtcoreCompileOptions( rtcoreCompileOptions )
    , attributeDecoderUUNames( attributeDecoderUUNames )
    , specializations( specializations )
    , heavyweightCallSiteNames( heavyweightCallSites )
    , stype( stype )
    , callerStype( callerStype )
    , cp( cp )
    , dimensionality( dimensionality )
    , numConlwrrentLaunchDevices( numConlwrrentLaunchDevices )
    , pagingMode( pagingMode )
{
}

bool CompiledProgramCacheKey::operator<( const CompiledProgramCacheKey& other ) const
{
    // Checking CanonicalProgramID first, because it is mostly likely to be varying and it's cheap to compute
    if( cp->getID() != other.cp->getID() )
        return cp->getID() < other.cp->getID();

    if( device != other.device )
        return device < other.device;

    if( compileParams != other.compileParams )
        return compileParams < other.compileParams;

    if( rtcoreCompileOptions != other.rtcoreCompileOptions )
        return rtcoreCompileOptions < other.rtcoreCompileOptions;

    if( heavyweightCallSiteNames != other.heavyweightCallSiteNames )
        return heavyweightCallSiteNames < other.heavyweightCallSiteNames;

    if( stype != other.stype )
        return stype < other.stype;

    if( callerStype != other.callerStype )
        return callerStype < other.callerStype;

    if( dimensionality != other.dimensionality )
        return dimensionality < other.dimensionality;

    if( numConlwrrentLaunchDevices != other.numConlwrrentLaunchDevices )
        return numConlwrrentLaunchDevices < other.numConlwrrentLaunchDevices;

    if( pagingMode != other.pagingMode )
        return pagingMode < other.pagingMode;

    if( attributeDecoderUUNames.size() != other.attributeDecoderUUNames.size() )
        return attributeDecoderUUNames.size() < other.attributeDecoderUUNames.size();

    for( size_t i = 0; i < attributeDecoderUUNames.size(); ++i )
        if( attributeDecoderUUNames[i] != other.attributeDecoderUUNames[i] )
            return attributeDecoderUUNames[i] < other.attributeDecoderUUNames[i];

    // Specializations goes last, because it is the most unwieldy.
    return specializations < other.specializations;
}

void optix::readOrWrite( PersistentStream* stream, RtcCompileOptions* options, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "RtcCompileOptions" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &options->abiVariant, "abiVariant" );
    readOrWrite( stream, &options->numPayloadRegisters, "numPayloadRegisters" );
    readOrWrite( stream, &options->numAttributeRegisters, "numAttributeRegisters" );
    readOrWrite( stream, &options->numCallableParamRegisters, "numCallableParamRegisters" );
    readOrWrite( stream, &options->numMemoryAttributeScalars, "numMemoryAttributeScalars" );
    readOrWrite( stream, &options->smVersion, "smVersion" );
    readOrWrite( stream, &options->maxRegisterCount, "maxRegisterCount" );
    readOrWrite( stream, &options->optLevel, "optLevel" );
    readOrWrite( stream, &options->debugLevel, "debugLevel" );
    readOrWrite( stream, &options->exceptionFlags, "exceptionFlags" );
    readOrWrite( stream, &options->smemSpillPolicy, "smemSpillPolicy" );
    readOrWrite( stream, &options->traversableGraphFlags, "traversableGraphFlags" );
    readOrWrite( stream, &options->useLWPTX, "useLWPTX" );
    readOrWrite( stream, &options->targetSharedMemoryBytesPerSM, "targetSharedMemoryBytesPerSM" );
    readOrWrite( stream, &options->usesTraversables, "usesTraversables" );
}

void optix::readOrWrite( PersistentStream* stream, CompiledProgramCacheKey* key, const char* label )
{
    RTCore                     rtcore;
    auto                       tmp     = stream->pushObject( label, "CompiledProgramCacheKey" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    // To be completely safe we need to ilwalidate on an rtcore change. So we just write the
    // rtcore UUID into the stream as well
    Rtlw32 rtcoreUUID[4];
    rtcore.rtcGetBuildUUID( rtcoreUUID );
    stream->readOrWriteObjectVersion( rtcoreUUID );

    // Device is not serialized
    readOrWrite( stream, &key->compileParams, "compileParams" );
    readOrWrite( stream, &key->rtcoreCompileOptions, "rtcoreCompileOptions" );  // See note above
    readOrWrite( stream, &key->attributeDecoderUUNames, "attributeDecoderUUNames" );
    readOrWrite( stream, &key->specializations, "specializations" );
    readOrWrite( stream, &key->heavyweightCallSiteNames, "heavyweightCallSites" );
    readOrWrite( stream, &key->stype, "stype" );
    readOrWrite( stream, &key->callerStype, "callerStype" );
    readOrWrite( stream, &key->dimensionality, "dimensionality" );
    readOrWrite( stream, &key->numConlwrrentLaunchDevices, "numConlwrrentLaunchDevices" );
    readOrWrite( stream, &key->pagingMode, "pagingMode" );
    // Canonical program is not serialized here - handle it in the caller
}

bool CompiledProgramCache::find( const LWDADevice* device, Rtlw64 hash, CompiledModuleHandle& handle ) const
{
    std::lock_guard<std::mutex> guard( m_mutex );
    const auto                  cacheIter = m_cache.find( device );
    if( cacheIter == m_cache.end() )
        return false;
    const auto iter = cacheIter->second.m_cacheByRtcModuleHash.find( hash );
    if( iter == cacheIter->second.m_cacheByRtcModuleHash.end() )
        return false;
    handle = iter->second;
    return true;
}

bool CompiledProgramCache::find( CompiledProgramCacheKey key, ModuleEntryRefPair& handle ) const
{
    std::lock_guard<std::mutex> guard( m_mutex );
    const auto                  cacheIter = m_cache.find( key.device );
    if( cacheIter == m_cache.end() )
        return false;
    const auto iter = cacheIter->second.m_cacheByKey.find( key );
    if( iter == cacheIter->second.m_cacheByKey.end() )
        return false;
    handle = iter->second;
    return true;
}

void CompiledProgramCache::emplace( const LWDADevice* device, Rtlw64 hash, CompiledModuleHandle& handle )
{
    std::lock_guard<std::mutex> guard( m_mutex );
    // insertion is desired if not found
    m_cache[device].m_cacheByRtcModuleHash.emplace( hash, handle );
}

void CompiledProgramCache::emplace( const CompiledProgramCacheKey& key, ModuleEntryRefPair& module )
{
    std::lock_guard<std::mutex> guard( m_mutex );
    // insertion is desired if not found
    m_cache[key.device].m_cacheByKey.emplace( key, module );
}

void CompiledProgramCache::removeProgramsForDevice( const LWDADevice* device )
{
    std::lock_guard<std::mutex> guard( m_mutex );
    m_cache.erase( device );
}


