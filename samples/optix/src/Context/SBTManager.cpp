#include <Context/SBTManager.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/RTCore.h>
#include <Context/TableManager.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/RTX/RTXFrameTask.h>
#include <ExelwtionStrategy/RTX/RTXPlan.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Memory/MBuffer.h>
#include <Memory/MemoryManager.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Util/ContainerAlgorithm.h>
#include <corelib/math/MathUtil.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/TimeViz.h>

#include <array>
#include <cstring>
#include <limits>
#include <utility>

using namespace optix;

namespace {
// clang-format off
  Knob<std::string> k_saveSBT( RT_DSTRING( "launch.saveSBT" ), "", RT_DSTRING( "Save Shader Binding Table in file at launch" ) );
  Knob<bool>        k_dumpUnusedSBTValues( RT_DSTRING( "launch.dumpUnusedSBTValues" ), false, RT_DSTRING( "When dumping the Shader Binding Table, also dump values that are not used." ) );
// clang-format on
}  // namespace

static std::string hexString( size_t u, unsigned int width = 8 )
{
    std::stringstream ss;
    ss << "0x" << std::setw( width ) << std::setfill( '0' ) << std::hex << u;
    return ss.str();
}
// -----------------------------------------------------------------------------
SBTManager::SBTManager( Context* context )
    : m_context( context )
    , m_allocator( std::numeric_limits<unsigned int>::max() )
{
    if( !m_context->useRtxDataModel() )
        return;

    // Query the Record Header Size.
    m_context->getRTCore()->getSbtRecordHeaderSize( (Rtlw64*)&m_sbtRecordHeaderSize );

    size_t interleavedSize = getSBTRecordSize();
    m_sbtRecordDataSize    = interleavedSize - m_sbtRecordHeaderSize;
    m_sbt.reset( new DeviceSpecificTableBase( context, m_sbtRecordDataSize, m_sbtRecordHeaderSize,
                                              m_sbtRecordHeaderSize, 0U, interleavedSize ) );


    entryPointCountDidChange();
    rayTypeCountDidChange();

    m_allocator.setFreeBlockCallback(
        [this]( size_t offset, size_t size ) { this->freeInstanceCallback( offset, size ); } );
}

// -----------------------------------------------------------------------------
SBTManager::~SBTManager()
{
    if( !m_context->useRtxDataModel() )
        return;

    unmapSBTFromHost();
    RT_ASSERT_NOTHROW( m_GGregistrations.empty() && m_GGupdateList.empty(),
                       "Found dangling GeometryGroup registrations" );
}

//------------------------------------------------------------------------------
void SBTManager::rebuildSBT()
{
    // This happens when the active devices change, reallocate SBT on all active devices and fill the data sections

    const size_t count = m_allocator.getUsedAddressRangeEnd();
    llog( 30 ) << "SBT has " << count << " records\n";

    DeviceManager* dm = m_context->getDeviceManager();

    m_sbt->resize( count );

    for( Device* device : dm->activeDevices() )
    {
        // now fill the data regions
        rebuildEntryPointData( device );

        rebuildMissProgramData( device, m_context->getRayTypeCount() );

        for( const GeometryGroupRegistration& gr : m_GGregistrations )
        {
            fillSBTMaterials( m_context->getRayTypeCount(), gr.gg, gr.gg->getSBTRecordIndex(), device );
        }

        rebuildCPData( device );
    }

    m_needsSync = true;
}
//------------------------------------------------------------------------------
void SBTManager::rebuildEntryPointData( Device* device )
{
    // rayGenAllocation size might be larger than the actual number of entry points
    // if we are in the process of increasing the number in GlobalScope::setEntryPointCount
    RT_ASSERT_MSG( m_rayGenAllocation.size >= m_context->getGlobalScope()->getAllEntryPointCount(),
                   "Did not allocate the correct number of raygen/exception records for the number of entry points" );
    for( unsigned int entryPoint = 0; entryPoint < m_context->getGlobalScope()->getAllEntryPointCount(); ++entryPoint )
    {
        // ray gen program
        Program*             program       = m_context->getGlobalScope()->getRayGenerationProgram( entryPoint );
        cort::SBTRecordData* sbtData       = getSBTDiDataHostPointer( *m_rayGenAllocation.index + entryPoint );
        size_t               programOffset = program->getRecordOffset();
        sbtData->ProgramData.programOffset = programOffset;
        updateProgramEntries( *m_rayGenAllocation.index + entryPoint, program, device, ST_RAYGEN );

        // exception program
        program                            = m_context->getGlobalScope()->getExceptionProgram( entryPoint );
        sbtData                            = getSBTDiDataHostPointer( *m_exceptionAllocation.index + entryPoint );
        programOffset                      = program->getRecordOffset();
        sbtData->ProgramData.programOffset = programOffset;
        updateProgramEntries( *m_exceptionAllocation.index + entryPoint, program, device, ST_EXCEPTION );
    }
}
//------------------------------------------------------------------------------
void SBTManager::rebuildMissProgramData( Device* device, unsigned int recordsToBuild )
{
    // we might rebuild fewer MS entries than we have allocated in case we are lwrrently growing the SBT
    RT_ASSERT_MSG( m_missAllocation.size >= recordsToBuild,
                   "Did not allocate the correct number of miss program records for the number of ray types" );
    for( unsigned int i = 0; i < recordsToBuild; ++i )
    {
        Program*             program       = m_context->getGlobalScope()->getMissProgram( i );
        cort::SBTRecordData* sbtData       = getSBTDiDataHostPointer( *m_missAllocation.index + i );
        size_t               programOffset = program->getRecordOffset();
        sbtData->ProgramData.programOffset = programOffset;
        updateProgramEntries( *m_missAllocation.index + i, program, device, ST_MISS );
    }
}
//------------------------------------------------------------------------------
void SBTManager::rebuildCPData( Device* device )
{
    // Update SBT records of callable programs.
    ObjectManager*          om       = m_context->getObjectManager();
    ReusableIDMap<Program*> programs = om->getPrograms();
    for( const Program* program : programs )
    {
        size_t sbtRecordIndex = program->getSBTRecordIndex();
        if( sbtRecordIndex != static_cast<size_t>( -1 ) )
        {
            // Callable programs keep track of their SBT index. Other programs do not,
            // so we know that this is a callable program
            if( program->isBindless() || program->isUsedAsBoundingBoxProgram() )
            {
                updateProgramEntries( sbtRecordIndex, program, device, ST_BINDLESS_CALLABLE_PROGRAM );
                // Heavyweight bindless callables have the same index, but a different key in the program entries
                // so we need to make sure that the program entries are correct.
                updateProgramEntries( sbtRecordIndex, program, device, ST_BINDLESS_CALLABLE_PROGRAM,
                                      ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE );
            }
            else
            {
                std::vector<SemanticType> inheritedTypes = program->getInheritedSemanticTypes();
                for( SemanticType t : inheritedTypes )
                {
                    updateProgramEntries( sbtRecordIndex + t, program, device, ST_BOUND_CALLABLE_PROGRAM, t );
                }
            }
        }
    }
}
// -----------------------------------------------------------------------------
void SBTManager::finalizeBeforeLaunch()
{
    if( !m_context->useRtxDataModel() )
        return;

    if( !m_needsSync )
    {
        return;
    }

    if( m_GGupdateList.size() )
    {
        for( Device* device : m_context->getDeviceManager()->activeDevices() )
        {
            const unsigned int numRayTypes = m_context->getRayTypeCount();

            // Update SBT (headers and data) for all delayed geometry groups
            for( const GeometryGroup* gg : m_GGupdateList )
            {
                int ggStartIndex = gg->getSBTRecordIndex();

                fillSBTMaterials( numRayTypes, gg, ggStartIndex, device );

                // add GG to the registered groups
                registerGeometryGroup( gg );
            }
        }

        // No more GGs with delayed update
        m_GGupdateList.clear();
    }

    if( !k_saveSBT.get().empty() )
    {
        const std::string& filename = k_saveSBT.get();
        std::ostringstream out;
        dumpSBT( out );
        std::string contents = out.str();
        if( contents != m_printedLayoutCache )
        {
            std::ofstream out_file( filename, m_context->getLaunchCount() <= 1 ? std::ios_base::out : std::ios_base::app );
            out_file << "=============================================================================================="
                        "======"
                        "======"
                        "=================\n";
            out_file << "Launch: " << m_context->getLaunchCount() << '\n';
            out_file << contents;
            m_printedLayoutCache.swap( contents );
        }
    }

    // Now that we are done filling in the headers, push it to the devices
    syncSBTForLaunch();

    m_needsSync = false;
}

// -----------------------------------------------------------------------------
void SBTManager::fillSBTMaterials( unsigned int numRayTypes, const GeometryGroup* gg, int ggStartIndex, const Device* device )
{
    // The default SBT entry for GI #giIndex is (ggStartIndex + giIndex*numRayTypes)
    // Material 0 for any GI must be at that position!
    // All sbt entries for materials with materialIndex>0 must be appended at the end (i.e., >= ggStartIndex+gg->getChildCount()*numRayTypes)
    // At runtime, mat0 will be queried and a 'skip' number (stored as sbtData->GeometryInstanceData.skip) must be used to select the actual sbt entry.
    // First pass will fill in materials 0 and the skip (for all GIs and ray types)
    // Second pass fills in the other materials (materialIndex >0) (for all GIs and ray types)

    // sbtIndexSkip defines the number of sbt entries that need to be skipped to access materials with materialIndex>0 for any given GI in the GG
    // note that this ignores numRayTypes (at runtime the stride will be adjusted by numRayTypes)!
    const unsigned int numGIs       = gg->getChildCount();
    int                sbtIndexSkip = numGIs;
    unsigned int       sbtIndex     = ggStartIndex;
    // If we are dealing with GeometryTriangles, the SBT entries of the materials of a GI must be conselwtive
    // The GeometryTriangles is virtually split up into multiple Geometries with a single material
    // Note that we can only do that, because we know how to split up the Geometry at bvh build time.
    bool conselwtiveLayout =
        numGIs > 0 && managedObjectCast<GeometryTriangles>( static_cast<GeometryInstance*>( gg->getChild( 0 ) )->getGeometry() );

    for( int pass = 0; pass < ( conselwtiveLayout ? 1 : 2 ); ++pass )
    {
        for( unsigned int giIndex = 0; giIndex < numGIs; ++giIndex )
        {
            LexicalScope*     child = gg->getChild( giIndex );
            GeometryInstance* gi    = managedObjectCast<GeometryInstance>( child );
            RT_ASSERT( gi );

            Geometry* geo       = gi->getGeometry();
            Program*  isProgram = geo ? geo->getIntersectionProgram() : nullptr;

            // if conselwtiveLayout -> [0;N), otherwise -> pass0: [0;1), pass1: [1;N)
            // (there is only one pass in the conselwtiveLayout case)
            const int firstMaterialIndex = conselwtiveLayout ? 0 : ( pass == 0 ? 0 : 1 );
            const int lastMaterialIndex = conselwtiveLayout ? gi->getMaterialCount() : ( pass == 0 ? 1 : gi->getMaterialCount() );
            for( int materialIndex = firstMaterialIndex; materialIndex < lastMaterialIndex; ++materialIndex )
            {
                Material* material = gi->getMaterial( materialIndex );
                for( unsigned int rayIndex = 0; rayIndex < numRayTypes; ++rayIndex )
                {
                    if( prodlib::log::active( 30 ) )
                    {
                        llog( 30 ) << "SBT_MANAGER: ScopeID: " << gg->getScopeID() << " ggStartIndex: " << ggStartIndex
                                   << " sbtIndex: " << sbtIndex << " giIndex: " << giIndex
                                   << " materialIndex: " << materialIndex << " rayIndex: " << rayIndex << "\n";
                    }
                    if( isProgram )
                    {
                        updateProgramEntries( sbtIndex, isProgram, device, ST_INTERSECTION );
                    }

                    if( material )
                    {
                        Program* ahProgram = material->getAnyHitProgram( rayIndex );
                        Program* chProgram = material->getClosestHitProgram( rayIndex );

                        updateProgramEntries( sbtIndex, ahProgram, device, ST_ANY_HIT );
                        updateProgramEntries( sbtIndex, chProgram, device, ST_CLOSEST_HIT );
                    }

                    cort::SBTRecordData* sbtData           = getSBTDiDataHostPointer( sbtIndex );
                    sbtData->GeometryInstanceData.giOffset = gi->getRecordOffset();
                    if( material )
                        sbtData->GeometryInstanceData.materialOffset = material->getRecordOffset();
                    sbtData->GeometryInstanceData.skip               = materialIndex == 0 ? sbtIndexSkip : 0;

                    ++sbtIndex;
                }
            }
            if( pass == 0 )
            {
                // increase by #materials-1 of GI_i
                // decrease by 1 as the next GI (GI_{i+1}) does not need to skip mat0 of GI_i
                sbtIndexSkip += gi->getMaterialCount() - 2;
            }
        }
    }
}
// -----------------------------------------------------------------------------
SBTManager::Handle SBTManager::allocateInstances( GeometryGroup* gg, unsigned int numRayTypes )
{
    m_needsSync = true;

    // For now, when we encounter a partially formed GeometryGroup, we will return an empty
    // allocation and unregister the group.
    unsigned int recordCount = 0;
    for( unsigned int giIndex = 0, numGIs = gg->getChildCount(); giIndex < numGIs; ++giIndex )
    {
        LexicalScope* child = gg->getChild( giIndex );
        if( child == nullptr )
        {
            unregisterGeometryGroup( gg );
            return Handle();
        }
        GeometryInstance* gi            = managedObjectCast<GeometryInstance>( child );
        int               materialCount = gi->getMaterialCount();
        if( materialCount == 0 )
        {
            unregisterGeometryGroup( gg );
            return Handle();
        }
        recordCount += materialCount * numRayTypes;
    }

    if( recordCount == 0 )
    {
        unregisterGeometryGroup( gg );
        return Handle();
    }

    Handle result;
    result = m_allocator.alloc( recordCount );
    growDeviceSpecificSBT();

    if( !m_GGupdateList.itemIsInList( gg ) )
    {
        registerGeometryGroupForUpdate( gg );
    }

    if( prodlib::log::active( 30 ) )
    {
        llog( 30 ) << "SBT_MANAGER: ScopeID: " << gg->getScopeID() << " - firstIndex: " << *result
                   << " - offset: " << hexString( *result * getSBTRecordSize() ) << "\n";
    }
    return result;
}

// -----------------------------------------------------------------------------
void SBTManager::freeInstanceCallback( size_t index, size_t size )
{
    if( !m_context->useRtxDataModel() )
        return;

    for( size_t i = index, E = index + size; i < E; ++i )
    {
        if( m_SBTEntries[i] )
        {
            m_SBTEntries[i]->AHorP.reset();
            m_SBTEntries[i]->CHorBCP.reset();
            m_SBTEntries[i]->IS.reset();
            m_SBTEntries[i].reset();
        }
    }
}

// -----------------------------------------------------------------------------
void SBTManager::registerGeometryGroup( const GeometryGroup* gg )
{
    m_GGregistrations.addItem( GeometryGroupRegistration( gg ) );
}
//------------------------------------------------------------------------------
void SBTManager::unregisterGeometryGroup( const GeometryGroup* gg )
{
    m_GGregistrations.removeItem( GeometryGroupRegistration( gg ) );
    m_GGupdateList.removeItem( gg );
}

// -----------------------------------------------------------------------------
void SBTManager::syncSBTForLaunch()
{
    if( !m_context->useRtxDataModel() )
        return;

    m_sbt->sync();
}

// -----------------------------------------------------------------------------
void SBTManager::launchCompleted()
{
}

// -----------------------------------------------------------------------------
size_t SBTManager::getSBTRecordSize()
{
    return corelib::roundUp( m_sbtRecordHeaderSize + sizeof( cort::SBTRecordData ), (size_t)16 );
}

// -----------------------------------------------------------------------------
void* SBTManager::getRaygenSBTRecordDevicePtr( const Device* device, unsigned int entryPoint )
{
    RT_ASSERT_MSG( entryPoint < m_rayGenAllocation.size,
                   "entryPoint is out of bounds to the number of allocated entrypoints" );
    return getSBTRecordDevicePtr( device, *m_rayGenAllocation.index + entryPoint );
}
// -----------------------------------------------------------------------------
void* SBTManager::getExceptionSBTRecordDevicePtr( const Device* device, unsigned int entryPoint )
{
    RT_ASSERT_MSG( entryPoint < m_rayGenAllocation.size,
                   "entryPoint is out of bounds to the number of allocated entrypoints" );
    return getSBTRecordDevicePtr( device, *m_exceptionAllocation.index + entryPoint );
}
// -----------------------------------------------------------------------------
void* SBTManager::getMissSBTRecordDevicePtr( const Device* device )
{
    if( m_missAllocation.size > 0 )
        return getSBTRecordDevicePtr( device, *m_missAllocation.index );
    else
        return nullptr;
}
// -----------------------------------------------------------------------------
void* SBTManager::getInstancesSBTRecordDevicePtr( const Device* device )
{
    // SBT indices for instances should be relative to the start of the SBT.
    return getSBTRecordDevicePtr( device, 0 );
}
// -----------------------------------------------------------------------------
void* SBTManager::getCallableProgramSBTRecordDevicePtr( const Device* device )
{
    // SBT indices for callable programs should be relative to the start of the SBT.
    return getSBTRecordDevicePtr( device, 0 );
}

// -----------------------------------------------------------------------------
void SBTManager::preSetActiveDevices( const DeviceArray& removedDevices )
{
    if( !m_context->useRtxDataModel() )
        return;

    unmapSBTFromHost();

    for( const Device* device : removedDevices )
    {
        m_sbt->activeDeviceRemoved( device->allDeviceListIndex() );
    }
}
// -----------------------------------------------------------------------------
void SBTManager::postSetActiveDevices()
{
    if( !m_context->useRtxDataModel() )
        return;
    m_sbt->setActiveDevices();
    rebuildSBT();
}


// -----------------------------------------------------------------------------
void SBTManager::rayTypeCountDidChange()
{
    if( !m_context->useRtxDataModel() )
        return;
    m_needsSync           = true;
    unsigned int newCount = m_context->getRayTypeCount();

    if( newCount )
    {
        unsigned int oldCount = m_missAllocation.size;
        if( growSBTRegion( m_missAllocation, newCount ) )
        {
            for( auto device : m_context->getDeviceManager()->activeDevices() )
            {
                rebuildMissProgramData( device, oldCount );
            }
        }
        m_missAllocation.size = newCount;
    }
    else
    {
        m_missAllocation.reset();
    }
}

//------------------------------------------------------------------------------
void SBTManager::entryPointCountDidChange()
{
    if( !m_context->useRtxDataModel() )
        return;
    m_needsSync = true;

    // Add 1 for the AABB iterator program
    size_t numEntryPoints = m_context->getEntryPointCount() + 1;

    if( numEntryPoints )
    {
        bool rgChanged = growSBTRegion( m_rayGenAllocation, numEntryPoints );
        bool exChanged = growSBTRegion( m_exceptionAllocation, numEntryPoints );
        if( rgChanged || exChanged )
        {
            for( auto device : m_context->getDeviceManager()->activeDevices() )
            {
                rebuildEntryPointData( device );
            }
        }
        m_rayGenAllocation.size    = numEntryPoints;
        m_exceptionAllocation.size = numEntryPoints;
    }
    else
    {
        m_rayGenAllocation.reset();
        m_exceptionAllocation.reset();
    }
}

//------------------------------------------------------------------------------
bool SBTManager::growSBTRegion( SBTRange& range, size_t count )
{
    if( range.capacity >= count )
    {
        range.size = count;
        return false;
    }

    bool   oldIndexValid = range.index.get();
    size_t oldIndex      = oldIndexValid ? *range.index : 0;
    size_t oldSize       = oldIndexValid ? range.size : 0;

    std::vector<std::unique_ptr<SBTEntry>> oldEntries( oldSize );
    for( size_t i = 0; i < oldSize; ++i )
    {
        // Keep old SBTEntries for this allocation. This is needed for in-place
        // re-allocations because the entries will be discarded on reset.
        // An alternative would be to rebuild the SBT region in all cases if oldIndexValid.
        oldEntries[i] = std::move( m_SBTEntries[oldIndex + i] );
    }

    // release old range so the old block is available for allocation
    range.index.reset();
    size_t   paddedSize = std::max( range.capacity * 2, count );
    SBTRange newRange   = {m_allocator.alloc( paddedSize ), paddedSize, count};
    range               = newRange;

    growDeviceSpecificSBT();

    if( oldIndexValid )
    {
        if( *newRange.index == oldIndex )
        {
            // Allocation was grown in place. Re-fill SBTEntries
            for( size_t i = 0; i < oldEntries.size(); ++i )
            {
                m_SBTEntries[oldIndex + i] = std::move( oldEntries[i] );
            }
            return false;
        }
        // Return that the allocation has been moved. If yes, the entries for the region need to be rebuilt.
        // Note that this is independent of whether there was a reallocation on the devices.
        // Even without that the allocation may be in a different place than before.
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------
bool SBTManager::growDeviceSpecificSBT()
{
    size_t count = m_allocator.getUsedAddressRangeEnd();

    if( m_SBTEntries.size() < count )
    {
        m_SBTEntries.resize( count );
    }

    return m_sbt->resize( count );
}
//------------------------------------------------------------------------------
void SBTManager::rayGenerationProgramDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( index < m_rayGenAllocation.size );
    m_needsSync = true;

    updateSBTProgramOffset( *m_rayGenAllocation.index + index, program, ST_RAYGEN );
}

//------------------------------------------------------------------------------
void SBTManager::exceptionProgramDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( index < m_exceptionAllocation.size );
    m_needsSync = true;

    updateSBTProgramOffset( *m_exceptionAllocation.index + index, program, ST_EXCEPTION );
}

//------------------------------------------------------------------------------
void SBTManager::missProgramDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( index < m_missAllocation.size );
    m_needsSync = true;

    updateSBTProgramOffset( *m_missAllocation.index + index, program, ST_MISS );
}

//------------------------------------------------------------------------------
void SBTManager::updateSBTProgramOffset( size_t recordIndex, const Program* program, SemanticType stype, SemanticType inheritedStype )
{
    if( inheritedStype == ST_ILWALID )
        inheritedStype = stype;

    cort::SBTRecordData* sbtData       = getSBTDiDataHostPointer( recordIndex );
    int                  programOffset = program ? program->getRecordOffset() : 0;
    sbtData->ProgramData.programOffset = programOffset;

    if( program )
    {
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            updateProgramEntries( recordIndex, program, device, stype, inheritedStype );
        }
    }
    else
    {
        if( m_SBTEntries[recordIndex] )
        {
            switch( stype )
            {
                case ST_CLOSEST_HIT:
                    m_SBTEntries[recordIndex]->CHorBCP.reset();
                    break;
                case ST_INTERSECTION:
                    m_SBTEntries[recordIndex]->IS.reset();
                    break;
                default:
                    m_SBTEntries[recordIndex]->AHorP.reset();
                    break;
            }
        }
    }
}

void SBTManager::removeProgramsForDevices( const DeviceArray& devices )
{
    for( auto it = m_lwrrentPrograms.begin(), end = m_lwrrentPrograms.end(); it != end; )
    {
        auto               lwrrProgramIter = it++;
        CompiledProgramKey lwrrKey         = std::get<0>( *lwrrProgramIter );

        for( const Device* device : devices )
        {
            if( lwrrKey.allDeviceListIndex == device->allDeviceListIndex() )
            {
                m_lwrrentPrograms.erase( lwrrProgramIter );
                break;
            }
        }
    }
}

//------------------------------------------------------------------------------
void SBTManager::updatePrograms( const CompiledProgramMap& lwrrentPlansCompiledPrograms )
{
    // Find out which programs are not yet know to the SBT.
    // Those that are already known will already have their SBT record headers up to date.
    std::vector<std::pair<CompiledProgramKey, ModuleEntryRefPair>> newPrograms;

    for( const auto& newProgram : lwrrentPlansCompiledPrograms )
    {
        auto compiledProgram = m_lwrrentPrograms.find( newProgram.first );
        if( compiledProgram == m_lwrrentPrograms.end() )
        {
            newPrograms.emplace_back( newProgram );
        }
        else
        {
            if( newProgram.second != compiledProgram->second )
            {
                newPrograms.emplace_back( newProgram );
            }
        }
    }
    if( newPrograms.empty() )
        return;

    // Update the lwrrentPrograms.  This is needed, so that when we encounter IS/CH/AH
    // program, we can get all the most current values all at once.
    for( const auto& newProgram : newPrograms )
        m_lwrrentPrograms[newProgram.first] = newProgram.second;

    // This is so we only do the IS/CH/AH combination once.  We may not need it if we don't
    // care and just process it (up to) three times.  We could also adjust the container
    // (unordered_set), since we don't need it ordered.  We don't need to track this for the
    // other program types, since we should really only process them once.
    std::set<std::pair<int, unsigned int>> processed;

    for( const auto& newProgram : newPrograms )
    {
        Device*     device_generic = m_context->getDeviceManager()->allDevices()[newProgram.first.allDeviceListIndex];
        LWDADevice* device         = deviceCast<LWDADevice>( device_generic );
        RT_ASSERT_MSG( device, "Device isn't a LWDADevice" );

        // we may have unused programs. Those will be in the compiled programs but not in m_programEntries
        auto iter = m_programEntries.find( {newProgram.first.cpID, newProgram.first.stype, newProgram.first.inheritedStype} );
        if( iter != m_programEntries.end() )
        {
            for( const SBTIndex* index : iter->second )
            {
                switch( newProgram.first.stype )
                {
                    case ST_BINDLESS_CALLABLE_PROGRAM:
                        if( !processed
                                 .insert( std::make_pair( index->recordIndex, newProgram.first.allDeviceListIndex ) )
                                 .second )
                            continue;
                    // intentional fall-through
                    case ST_RAYGEN:
                    case ST_EXCEPTION:
                    case ST_MISS:
                    case ST_BOUND_CALLABLE_PROGRAM:
                    {
                        // index->recordIndex already holds the absolute record index not the one relative to
                        // the allocation, so all of these cases can be handled identically.
                        char* hostPtr = getSBTDdDataHostPointer( index->recordIndex, newProgram.first.allDeviceListIndex );
                        fillProgramSBTRecordHeader( hostPtr, newProgram.first.cpID, newProgram.first.stype,
                                                    newProgram.first.inheritedStype, device );
                    }
                    break;
                    case ST_INTERSECTION:
                    case ST_CLOSEST_HIT:
                    case ST_ANY_HIT:
                    {
                        unsigned int deviceIndex = newProgram.first.allDeviceListIndex;
                        // Skip already processed entries
                        if( !processed.insert( std::make_pair( index->recordIndex, deviceIndex ) ).second )
                            continue;
                        char* hostPtr = getSBTDdDataHostPointer( index->recordIndex, deviceIndex );
                        const std::unique_ptr<SBTEntry>& entry = m_SBTEntries[index->recordIndex];
                        fillProgramSBTRecordHeader( hostPtr, entry->cpIds[deviceIndex].isCPID, entry->cpIds[deviceIndex].ahCPID,
                                                    entry->cpIds[deviceIndex].chCPID, device );
                    }
                    break;
                    case ST_ATTRIBUTE:
                    case ST_NODE_VISIT:
                    case ST_BOUNDING_BOX:
                    case ST_INTERNAL_AABB_ITERATOR:
                    case ST_INTERNAL_AABB_EXCEPTION:
                    case ST_ILWALID:
                    case ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE:
                        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Invalid Semantic Type for compiled program" );
                }
            }
        }
    }
}
//------------------------------------------------------------------------------
void SBTManager::fillProgramSBTRecordHeader( char*              hostPtr,
                                             CanonicalProgramID cpID,
                                             SemanticType       stype,
                                             SemanticType       inheritedStype,
                                             const LWDADevice*  device )
{
    CompiledProgramKey key  = {cpID, stype, inheritedStype, device->allDeviceListIndex()};
    auto               iter = m_lwrrentPrograms.find( key );
    RT_ASSERT_MSG( iter != m_lwrrentPrograms.end(),
                   "Didn't find key during looking for fillProgramSBTRecordHeader - cpID: " + std::to_string( cpID ) );
    ModuleEntryRefPair compilerOutput = iter->second;
    switch( stype )
    {
        case ST_ANY_HIT:
#if RTCORE_API_VERSION >= 25
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, ~0, compilerOutput.first.get(),
                                                         compilerOutput.second, nullptr, ~0, hostPtr );
#else
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, nullptr, compilerOutput.first.get(),
                                                         compilerOutput.second.c_str(), nullptr, nullptr, hostPtr );
#endif
            break;
        case ST_INTERSECTION:
#if RTCORE_API_VERSION >= 25
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, ~0, nullptr, ~0,
                                                         compilerOutput.first.get(), compilerOutput.second, hostPtr );
#else
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, nullptr, nullptr, nullptr,
                                                         compilerOutput.first.get(), compilerOutput.second.c_str(), hostPtr );
#endif
            break;
        case ST_BINDLESS_CALLABLE_PROGRAM:
        {
            // Check if there is a version of this bindless callable of the other inherited type
            // so we can fill both header slots in one go.
            SemanticType otherInheritedStype = inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM ?
                                                   ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE :
                                                   ST_BINDLESS_CALLABLE_PROGRAM;
            CompiledProgramKey otherKey  = {cpID, stype, otherInheritedStype, device->allDeviceListIndex()};
            auto               otherIter = m_lwrrentPrograms.find( otherKey );

            ModuleEntryRefPair lightCompilerOutput;
#if RTCORE_API_VERSION >= 25
            unsigned int       heavyIndex = compilerOutput.second;
            unsigned int       lightIndex = ~0;
#else
            const char*         heavyName = compilerOutput.second.c_str();
            const char*         lightName = nullptr;
#endif
            if( otherIter != m_lwrrentPrograms.end() )
            {
                lightCompilerOutput = otherIter->second;
#if RTCORE_API_VERSION >= 25
                lightIndex          = lightCompilerOutput.second;
#else
                lightName           = lightCompilerOutput.second.c_str();
#endif
            }
            if( inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM )
            {
                // We want the heavyweight (i.e. continuation callable) variant in the
                // first header slot and the lightweight (i.e. direct callable)
                // in the second one.
                std::swap( compilerOutput, lightCompilerOutput );
#if RTCORE_API_VERSION >= 25
                std::swap( heavyIndex, lightIndex );
#else
                std::swap( heavyName, lightName );
#endif
            }
#if RTCORE_API_VERSION >= 25
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), compilerOutput.first.get(), heavyIndex,
                                                         lightCompilerOutput.first.get(), lightIndex, nullptr, ~0, hostPtr );
#else
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), compilerOutput.first.get(), heavyName,
                                                         lightCompilerOutput.first.get(), lightName, nullptr, nullptr, hostPtr );
#endif
            break;
        }
        case ST_BOUND_CALLABLE_PROGRAM:
        {
            if( RTXPlan::isDirectCalledBoundCallable( ST_BOUND_CALLABLE_PROGRAM, inheritedStype ) )
            {
                // for bound callables that are called as direct callables we need to
                // fill the second header slot only.
#if RTCORE_API_VERSION >= 25
                m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, ~0,
                                                             compilerOutput.first.get(), compilerOutput.second,
                                                             nullptr, ~0, hostPtr );
#else
                m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), nullptr, nullptr,
                                                             compilerOutput.first.get(), compilerOutput.second.c_str(),
                                                             nullptr, nullptr, hostPtr );
#endif
                break;
            }
            // intentional fall-through
        }
        default:
#if RTCORE_API_VERSION >= 25
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), compilerOutput.first.get(),
                                                         compilerOutput.second, nullptr, ~0, nullptr,
                                                         ~0, hostPtr );
#else
            m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), compilerOutput.first.get(),
                                                         compilerOutput.second.c_str(), nullptr, nullptr, nullptr,
                                                         nullptr, hostPtr );
#endif
            break;
    }

    // headers were changed, schedule for re-syncing
    m_needsSync = true;
}

//------------------------------------------------------------------------------
void SBTManager::fillProgramSBTRecordHeader( char*              hostPtr,
                                             CanonicalProgramID isCPID,
                                             CanonicalProgramID ahCPID,
                                             CanonicalProgramID chCPID,
                                             const LWDADevice*  device )
{
    SBTManager::CompiledProgramKey chKey = {chCPID, ST_CLOSEST_HIT, ST_CLOSEST_HIT, device->allDeviceListIndex()};
    SBTManager::CompiledProgramKey ahKey = {ahCPID, ST_ANY_HIT, ST_ANY_HIT, device->allDeviceListIndex()};
    SBTManager::CompiledProgramKey isKey = {isCPID, ST_INTERSECTION, ST_INTERSECTION, device->allDeviceListIndex()};

    auto chIter = m_lwrrentPrograms.find( chKey );
    if( chIter == m_lwrrentPrograms.end() )
        return;
    RT_ASSERT_MSG( chIter != m_lwrrentPrograms.end(),
                   "Didn't find key during looking for fillProgramSBTRecordHeader - cpID: " + std::to_string( chCPID ) );

    auto ahIter = m_lwrrentPrograms.find( ahKey );
    if( ahIter == m_lwrrentPrograms.end() )
        return;
    RT_ASSERT_MSG( ahIter != m_lwrrentPrograms.end(),
                   "Didn't find key during looking for fillProgramSBTRecordHeader - cpID: " + std::to_string( ahCPID ) );

    auto isIter = m_lwrrentPrograms.find( isKey );
    if( isIter == m_lwrrentPrograms.end() )
        return;
    RT_ASSERT_MSG( isIter != m_lwrrentPrograms.end(),
                   "Didn't find key during looking for fillProgramSBTRecordHeader - cpID: " + std::to_string( isCPID ) );

    auto& ch = chIter->second;
    auto& ah = ahIter->second;
    auto& is = isIter->second;

#if RTCORE_API_VERSION >= 25
    m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), ch.first.get(), ch.second, ah.first.get(),
                                                 ah.second, is.first.get(), is.second, hostPtr );
#else
    m_context->getRTCore()->packSbtRecordHeader( device->rtcContext(), ch.first.get(), ch.second.c_str(), ah.first.get(),
                                                 ah.second.c_str(), is.first.get(), is.second.c_str(), hostPtr );
#endif
    m_needsSync = true;
}

//------------------------------------------------------------------------------
void SBTManager::updateProgramEntries( size_t recordIndex, const Program* program, const Device* device, SemanticType stype, SemanticType inheritedStype )
{
    if( inheritedStype == ST_ILWALID )
        inheritedStype = stype;

    auto cpID = program->getCanonicalProgram( device )->getID();

    ProgramEntriesKey key{cpID, stype, inheritedStype};
    SBTIndex*         indexToUpdate = nullptr;
    if( !m_SBTEntries[recordIndex] )
    {
        m_SBTEntries[recordIndex].reset( new SBTEntry() );
    }
    switch( stype )
    {
        case ST_ANY_HIT:
            m_SBTEntries[recordIndex]->cpIds.resize( std::max( m_SBTEntries[recordIndex]->cpIds.size(),
                                                               static_cast<size_t>( device->allDeviceListIndex() + 1 ) ) );
            m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].ahCPID = cpID;
        // intentional fall-through
        case ST_RAYGEN:
        case ST_BOUND_CALLABLE_PROGRAM:
        case ST_MISS:
        case ST_EXCEPTION:
            indexToUpdate = &m_SBTEntries[recordIndex]->AHorP;
            break;
        case ST_BINDLESS_CALLABLE_PROGRAM:
            if( inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM )
                indexToUpdate = &m_SBTEntries[recordIndex]->AHorP;
            else
                indexToUpdate = &m_SBTEntries[recordIndex]->CHorBCP;
            break;
        case ST_CLOSEST_HIT:
            m_SBTEntries[recordIndex]->cpIds.resize( std::max( m_SBTEntries[recordIndex]->cpIds.size(),
                                                               static_cast<size_t>( device->allDeviceListIndex() + 1 ) ) );
            m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].chCPID = cpID;
            indexToUpdate                                                         = &m_SBTEntries[recordIndex]->CHorBCP;
            break;
        case ST_INTERSECTION:
            m_SBTEntries[recordIndex]->cpIds.resize( std::max( m_SBTEntries[recordIndex]->cpIds.size(),
                                                               static_cast<size_t>( device->allDeviceListIndex() + 1 ) ) );
            m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].isCPID = cpID;
            indexToUpdate                                                         = &m_SBTEntries[recordIndex]->IS;
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type" );
            return;
    }
    indexToUpdate->reset();
    indexToUpdate->recordIndex = recordIndex;
    m_programEntries[key].addItem( indexToUpdate );
    indexToUpdate->m_lwrrentParent = &m_programEntries[key];

    if( device )
    {
        CompiledProgramKey key2 = {cpID, stype, inheritedStype, device->allDeviceListIndex()};
        auto               iter = m_lwrrentPrograms.find( key2 );
        if( iter != m_lwrrentPrograms.end() )
        {
            const LWDADevice* cDevice = deviceCast<const LWDADevice>( device );
            char*             hostPtr = getSBTDdDataHostPointer( recordIndex, device->allDeviceListIndex() );
            if( stype == ST_ANY_HIT || stype == ST_CLOSEST_HIT || stype == ST_INTERSECTION )
            {
                CanonicalProgramID isCPID = m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].isCPID;
                CanonicalProgramID ahCPID = m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].ahCPID;
                CanonicalProgramID chCPID = m_SBTEntries[recordIndex]->cpIds[device->allDeviceListIndex()].chCPID;
                fillProgramSBTRecordHeader( hostPtr, isCPID, ahCPID, chCPID, cDevice );
            }
            else
            {
                fillProgramSBTRecordHeader( hostPtr, cpID, stype, inheritedStype, cDevice );
            }
        }
    }
}

//------------------------------------------------------------------------------
bool SBTManager::geometryGroupHasDelayedUpdate( const GeometryGroup* gg )
{
    return m_GGupdateList.itemIsInList( gg );
}

//------------------------------------------------------------------------------
SBTManager::Handle SBTManager::callableProgramDidChange( const Program* program )
{
    Handle result;
    if( !m_context->useRtxDataModel() )
        return result;

    RT_ASSERT( program );

    m_needsSync = true;

    if( program->isBindless() || program->isUsedAsBoundingBoxProgram() )
    {
        // Bindless callables have two SBT entries, the first one is the regular one,
        // the second one is for heavyweight (i.e. continuation call) compiled.
        result = m_allocator.alloc( 1 );
        growDeviceSpecificSBT();
        if( prodlib::log::active( 30 ) )
        {
            llog( 30 ) << "SBT_MANAGER: bindless callable: " << program->getInputFunctionName() << " - firstIndex: " << *result
                       << " - offset: " << hexString( *result * getSBTRecordSize() ) << "\n";
        }

        updateSBTProgramOffset( *result, program, ST_BINDLESS_CALLABLE_PROGRAM );
        updateSBTProgramOffset( *result, program, ST_BINDLESS_CALLABLE_PROGRAM, ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE );
    }
    else
    {
        // Make enough allocation space for all possible semantic types
        // Reordering SemanticType enum could reduce number of SBT slots.
        // TODO: Analyze if reordering would break other code.
        size_t allocSize = ST_BINDLESS_CALLABLE_PROGRAM + 1;

        result = m_allocator.alloc( allocSize );
        growDeviceSpecificSBT();
        if( prodlib::log::active( 30 ) )
        {
            llog( 30 ) << "SBT_MANAGER: bound callable: " << program->getInputFunctionName() << " - firstIndex: " << *result
                       << " - offset: " << hexString( *result * getSBTRecordSize() ) << "\n";
        }

        std::vector<SemanticType> inheritedTypes = program->getInheritedSemanticTypes();
        for( SemanticType t : inheritedTypes )
        {
            updateSBTProgramOffset( *result + t, program, ST_BOUND_CALLABLE_PROGRAM, t );
        }
    }
    return result;
}
//------------------------------------------------------------------------------
void SBTManager::updateBoundCallableProgramEntry( const Program* program, SemanticType inheritedSemanticType )
{
    RT_ASSERT( program );
    size_t baseIndex = program->getSBTRecordIndex();
    updateSBTProgramOffset( baseIndex + inheritedSemanticType, program, ST_BOUND_CALLABLE_PROGRAM, inheritedSemanticType );
}
//------------------------------------------------------------------------------

void SBTManager::intersectionProgramDidChange( const GeometryGroup* gg, unsigned int instanceIndex )
{
    if( !m_context->useRtxDataModel() )
        return;
    LexicalScope*     child = gg->getChild( instanceIndex );
    GeometryInstance* gi    = managedObjectCast<GeometryInstance>( child );
    RT_ASSERT( gi );
    Geometry* geo = gi->getGeometry();
    if( !geo )
        return;
    Program*             isProgram   = geo->getIntersectionProgram();
    int                  numRayTypes = m_context->getRayTypeCount();
    size_t               recordIndex = gg->getSBTRecordIndex() + ( instanceIndex * numRayTypes );
    cort::SBTRecordData* sbtData     = getSBTDiDataHostPointer( recordIndex );
    int                  skip        = sbtData->GeometryInstanceData.skip;

    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        size_t ri = recordIndex;
        // first material
        for( unsigned int i = 0; i < m_context->getRayTypeCount(); ++i )
        {
            updateProgramEntries( ri + i, isProgram, device, ST_INTERSECTION );
        }

        if( gi->getMaterialCount() > 1 )
        {
            // all other materials
            ri += skip * m_context->getRayTypeCount();
            for( unsigned int i = 0; i < ( gi->getMaterialCount() - 1 ) * m_context->getRayTypeCount(); ++i )
            {
                updateProgramEntries( ri + i, isProgram, device, ST_INTERSECTION );
            }
        }
    }
    m_needsSync = true;
}

//------------------------------------------------------------------------------
void SBTManager::closestHitProgramDidChange( const GeometryGroup* gg, unsigned int instanceIndex, unsigned int materialIndex, unsigned int rayTypeIndex )
{
    if( !m_context->useRtxDataModel() )
        return;
    LexicalScope*     child = gg->getChild( instanceIndex );
    GeometryInstance* gi    = managedObjectCast<GeometryInstance>( child );
    RT_ASSERT( gi );

    int    numRayTypes = m_context->getRayTypeCount();
    size_t recordIndex = gg->getSBTRecordIndex() + ( instanceIndex * numRayTypes ) + rayTypeIndex;

    if( materialIndex != 0 )
    {
        cort::SBTRecordData* sbtData = getSBTDiDataHostPointer( recordIndex );
        recordIndex += sbtData->GeometryInstanceData.skip * m_context->getRayTypeCount();
    }

    Program* chProgram = gi->getMaterial( materialIndex )->getClosestHitProgram( rayTypeIndex );

    for( auto device : m_context->getDeviceManager()->activeDevices() )
        updateProgramEntries( recordIndex, chProgram, device, ST_CLOSEST_HIT );

    m_needsSync = true;
}
//------------------------------------------------------------------------------

void SBTManager::anyHitProgramDidChange( const GeometryGroup* gg, unsigned int instanceIndex, unsigned int materialIndex, unsigned int rayTypeIndex )
{
    if( !m_context->useRtxDataModel() )
        return;
    LexicalScope*     child = gg->getChild( instanceIndex );
    GeometryInstance* gi    = managedObjectCast<GeometryInstance>( child );
    RT_ASSERT( gi );

    int    numRayTypes = m_context->getRayTypeCount();
    size_t recordIndex = gg->getSBTRecordIndex() + ( instanceIndex * numRayTypes ) + rayTypeIndex;

    if( materialIndex != 0 )
    {
        cort::SBTRecordData* sbtData = getSBTDiDataHostPointer( recordIndex );
        recordIndex += sbtData->GeometryInstanceData.skip * m_context->getRayTypeCount();
    }

    Program* ahProgram = gi->getMaterial( materialIndex )->getAnyHitProgram( rayTypeIndex );

    for( auto device : m_context->getDeviceManager()->activeDevices() )
        updateProgramEntries( recordIndex, ahProgram, device, ST_ANY_HIT );

    m_needsSync = true;
}
//------------------------------------------------------------------------------

void SBTManager::geometryInstanceDidChange( const GeometryGroup* gg, unsigned int instanceIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( geometryGroupHasDelayedUpdate( managedObjectCast<const GeometryGroup>( gg ) ) )
    {
        return;
    }

    const GeometryInstance* gi = managedObjectCast<GeometryInstance>( gg->getChild( instanceIndex ) );
    if( !gi )
    {
        // removal of GeometryInstances triggers a reallocate for that group, so no need to handle it here
        return;
    }
    size_t ggStartIndex = gg->getSBTRecordIndex();
    int    numRayTypes  = m_context->getRayTypeCount();


    Geometry* geo       = gi->getGeometry();
    Program*  isProgram = geo ? geo->getIntersectionProgram() : nullptr;

    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        size_t recordIndex = ggStartIndex + instanceIndex * numRayTypes;
        int    skip        = 0;
        // handle first material and get skip value along the way
        for( int i = 0; i < numRayTypes; ++i )
        {
            cort::SBTRecordData* sbtData           = getSBTDiDataHostPointer( recordIndex + i );
            skip                                   = sbtData->GeometryInstanceData.skip;
            sbtData->GeometryInstanceData.giOffset = gi->getRecordOffset();
            if( isProgram )
            {
                updateProgramEntries( recordIndex + i, isProgram, device, ST_INTERSECTION );
            }
            const Material* material = gi->getMaterial( 0 );
            if( material )
            {
                sbtData->GeometryInstanceData.materialOffset = material->getRecordOffset();
                Program* chProgram                           = material->getClosestHitProgram( i );
                Program* ahProgram                           = material->getAnyHitProgram( i );
                updateProgramEntries( recordIndex + i, chProgram, device, ST_CLOSEST_HIT );
                updateProgramEntries( recordIndex + i, ahProgram, device, ST_ANY_HIT );
            }
        }

        // advance to next material for this GI
        recordIndex += skip * numRayTypes;
        // take care of the rest
        for( int matIndex = 1; matIndex < gi->getMaterialCount(); ++matIndex )
        {
            for( int rayType = 0; rayType < numRayTypes; ++rayType )
            {
                cort::SBTRecordData* sbtData           = getSBTDiDataHostPointer( recordIndex );
                sbtData->GeometryInstanceData.giOffset = gi->getRecordOffset();
                updateProgramEntries( recordIndex, isProgram, device, ST_INTERSECTION );
                const Material* material = gi->getMaterial( matIndex );
                if( material )
                {
                    sbtData->GeometryInstanceData.materialOffset = material->getRecordOffset();
                    Program* chProgram                           = material->getClosestHitProgram( rayType );
                    Program* ahProgram                           = material->getAnyHitProgram( rayType );
                    updateProgramEntries( recordIndex, chProgram, device, ST_CLOSEST_HIT );
                    updateProgramEntries( recordIndex, ahProgram, device, ST_ANY_HIT );
                }
                ++recordIndex;
            }
        }
    }
    m_needsSync = true;
}
//------------------------------------------------------------------------------

void SBTManager::geometryDidChange( const GeometryGroup* gg, unsigned int instanceIndex )
{
    if( !m_context->useRtxDataModel() )
        return;
    intersectionProgramDidChange( gg, instanceIndex );
    m_needsSync = true;
}
//------------------------------------------------------------------------------

void SBTManager::materialDidChange( const GeometryGroup* gg, unsigned int instanceIndex, unsigned int materialIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( geometryGroupHasDelayedUpdate( gg ) )
    {
        return;
    }
    const GeometryInstance* gi = managedObjectCast<GeometryInstance>( gg->getChild( instanceIndex ) );
    RT_ASSERT( gi );
    const Material* material = gi->getMaterial( materialIndex );
    if( !material )
    {
        // removal of Materials triggers a reallocate for that group, so no need to handle it here
        return;
    }

    const size_t ggStartIndex     = gg->getSBTRecordIndex();
    const int    numRayTypes      = m_context->getRayTypeCount();
    const size_t matOffset        = material->getRecordOffset();
    const size_t firstRecordIndex = ggStartIndex + instanceIndex * numRayTypes;

    // Update first material.
    if( materialIndex == 0 )
    {
        for( int i = 0; i < numRayTypes; ++i )
        {
            cort::SBTRecordData* sbtData                 = getSBTDiDataHostPointer( firstRecordIndex + i );
            sbtData->GeometryInstanceData.materialOffset = matOffset;

            const Program* chProgram = material->getClosestHitProgram( i );
            const Program* ahProgram = material->getAnyHitProgram( i );
            for( Device* device : m_context->getDeviceManager()->activeDevices() )
            {
                updateProgramEntries( firstRecordIndex + i, chProgram, device, ST_CLOSEST_HIT );
                updateProgramEntries( firstRecordIndex + i, ahProgram, device, ST_ANY_HIT );
            }
        }
    }
    else
    {
        const cort::SBTRecordData* sbtDataFirst = getSBTDiDataHostPointer( firstRecordIndex );
        const int                  skip         = sbtDataFirst->GeometryInstanceData.skip;
        // skip to the requested material record
        const size_t recordIndex = firstRecordIndex + ( skip + materialIndex - 1 ) * numRayTypes;

        for( int i = 0; i < numRayTypes; ++i )
        {
            cort::SBTRecordData* sbtData = getSBTDiDataHostPointer( recordIndex + i );

            sbtData->GeometryInstanceData.materialOffset = matOffset;

            const Program* chProgram = material->getClosestHitProgram( i );
            const Program* ahProgram = material->getAnyHitProgram( i );
            for( Device* device : m_context->getDeviceManager()->activeDevices() )
            {
                updateProgramEntries( recordIndex + i, chProgram, device, ST_CLOSEST_HIT );
                updateProgramEntries( recordIndex + i, ahProgram, device, ST_ANY_HIT );
            }
        }
    }
    m_needsSync = true;
}
//------------------------------------------------------------------------------
void SBTManager::rayGenerationProgramOffsetDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( program );
    updateSBTProgramOffset( *m_rayGenAllocation.index + index, program, ST_RAYGEN );
    m_needsSync = true;
}
//------------------------------------------------------------------------------
void SBTManager::exceptionProgramOffsetDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( program );
    updateSBTProgramOffset( *m_exceptionAllocation.index + index, program, ST_EXCEPTION );
    m_needsSync = true;
}
//------------------------------------------------------------------------------
void SBTManager::missProgramOffsetDidChange( const Program* program, unsigned int index )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( program );
    updateSBTProgramOffset( *m_missAllocation.index + index, program, ST_MISS );
    m_needsSync = true;
}

//------------------------------------------------------------------------------
void SBTManager::registerGeometryGroupForUpdate( const GeometryGroup* gg )
{
    if( !m_context->useRtxDataModel() )
        return;
    m_GGregistrations.removeItem( GeometryGroupRegistration( gg ) );
    m_GGupdateList.addItem( gg );
}

//------------------------------------------------------------------------------
void SBTManager::geometryInstanceOffsetDidChange( const AbstractGroup* gg, unsigned int instanceIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( geometryGroupHasDelayedUpdate( managedObjectCast<const GeometryGroup>( gg ) ) )
    {
        return;
    }

    const GeometryInstance* gi = managedObjectCast<GeometryInstance>( gg->getChild( instanceIndex ) );
    RT_ASSERT( gi );


    size_t ggStartIndex = gg->getSBTRecordIndex();
    int    numRayTypes  = m_context->getRayTypeCount();
    size_t recordIndex  = ggStartIndex + instanceIndex * numRayTypes;

    int skip = 0;
    // handle first material and get skip value along the way
    for( int i = 0; i < numRayTypes; ++i )
    {
        cort::SBTRecordData* sbtData           = getSBTDiDataHostPointer( recordIndex + i );
        skip                                   = sbtData->GeometryInstanceData.skip;
        sbtData->GeometryInstanceData.giOffset = gi->getRecordOffset();
    }
    recordIndex += skip * numRayTypes;
    // take care of the rest
    for( int matIndex = 1; matIndex < gi->getMaterialCount(); ++matIndex )
    {
        for( int rayType = 0; rayType < numRayTypes; ++rayType )
        {
            cort::SBTRecordData* sbtData           = getSBTDiDataHostPointer( recordIndex );
            sbtData->GeometryInstanceData.giOffset = gi->getRecordOffset();
            ++recordIndex;
        }
    }
    m_needsSync = true;
}

//------------------------------------------------------------------------------
void SBTManager::materialOffsetDidChange( const GeometryGroup* gg, unsigned int instanceIndex, unsigned int materialIndex )
{
    if( !m_context->useRtxDataModel() )
        return;
    if( geometryGroupHasDelayedUpdate( gg ) )
    {
        return;
    }

    const GeometryInstance* gi = managedObjectCast<GeometryInstance>( gg->getChild( instanceIndex ) );
    RT_ASSERT( gi );
    const Material* material = gi->getMaterial( materialIndex );
    RT_ASSERT( material );


    size_t ggStartIndex = gg->getSBTRecordIndex();
    int    numRayTypes  = m_context->getRayTypeCount();
    size_t recordIndex  = ggStartIndex + instanceIndex * numRayTypes;
    size_t matOffset    = material->getRecordOffset();
    int    skip         = 0;
    // update first material (or collect skip).
    for( int i = 0; i < numRayTypes; ++i )
    {
        cort::SBTRecordData* sbtData = getSBTDiDataHostPointer( recordIndex + i );
        if( materialIndex == 0 )
        {
            sbtData->GeometryInstanceData.materialOffset = matOffset;
        }
        skip = sbtData->GeometryInstanceData.skip;
    }
    if( materialIndex != 0 )
    {
        // skip to the requested material record
        recordIndex += ( skip + materialIndex - 1 ) * numRayTypes;
        for( int i = 0; i < numRayTypes; ++i )
        {
            cort::SBTRecordData* sbtData = getSBTDiDataHostPointer( recordIndex + i );

            sbtData->GeometryInstanceData.materialOffset = matOffset;
        }
    }
}

//------------------------------------------------------------------------------
void SBTManager::callableProgramOffsetDidChange( const Program* program )
{
    if( !m_context->useRtxDataModel() )
        return;
    RT_ASSERT( program );
    if( program->isBindless() || program->isUsedAsBoundingBoxProgram() )
    {
        updateSBTProgramOffset( program->getSBTRecordIndex(), program, ST_BINDLESS_CALLABLE_PROGRAM );
        updateSBTProgramOffset( program->getSBTRecordIndex(), program, ST_BINDLESS_CALLABLE_PROGRAM,
                                ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE );
    }
    else
    {
        std::vector<SemanticType> inheritedTypes = program->getInheritedSemanticTypes();
        for( SemanticType t : inheritedTypes )
        {
            updateSBTProgramOffset( program->getSBTRecordIndex() + t, program, ST_BOUND_CALLABLE_PROGRAM, t );
        }
    }

    m_needsSync = true;
}

// -----------------------------------------------------------------------------
void SBTManager::unmapSBTFromHost()
{
    // Unmap table buffers from the host.

    const DeviceArray& activeDevices = m_context->getDeviceManager()->activeDevices();
    for( Device* device : activeDevices )
    {
        m_sbt->unmapFromHost( device->allDeviceListIndex() );
    }
}

// -----------------------------------------------------------------------------
char* SBTManager::getSBTBaseDevicePtr( const Device* device )
{
    return m_sbt->getInterleavedDevicePtr( device->allDeviceListIndex() );
}

// -----------------------------------------------------------------------------
char* SBTManager::getSBTRecordDevicePtr( const Device* device, int recordIndex )
{
    return getSBTBaseDevicePtr( device ) + recordIndex * getSBTRecordSize();
}

//------------------------------------------------------------------------------
cort::SBTRecordData* SBTManager::getSBTDiDataHostPointer( int recordIndex )
{
    RT_ASSERT( static_cast<size_t>( recordIndex ) < m_sbt->size() );
    char* di = m_sbt->mapDeviceIndependentPtr( recordIndex );
    return reinterpret_cast<cort::SBTRecordData*>( di );
}

//------------------------------------------------------------------------------
char* SBTManager::getSBTDdDataHostPointer( int recordIndex, int deviceIndex )
{
    RT_ASSERT( static_cast<size_t>( recordIndex ) < m_sbt->size() );
    char* dd = m_sbt->mapDeviceDependentPtr( deviceIndex, recordIndex );
    return dd;
}

//------------------------------------------------------------------------------
const cort::SBTRecordData* SBTManager::getSBTDiDataHostPointerReadOnly( int recordIndex )
{
    RT_ASSERT( static_cast<size_t>( recordIndex ) < m_sbt->size() );
    const char* di = m_sbt->mapDeviceIndependentPtrReadOnly();
    di += recordIndex * m_sbtRecordDataSize;
    return reinterpret_cast<const cort::SBTRecordData*>( di );
}

//------------------------------------------------------------------------------
const char* SBTManager::getSBTDdDataHostPointerReadOnly( int recordIndex, int deviceIndex )
{
    RT_ASSERT( static_cast<size_t>( recordIndex ) < m_sbt->size() );
    const char* dd = m_sbt->mapDeviceDependentPtrReadOnly( deviceIndex );
    dd += recordIndex * m_sbtRecordHeaderSize;
    return dd;
}
// -----------------------------------------------------------------------------
bool SBTManager::CompiledProgramKey::operator<( const CompiledProgramKey& other ) const
{
    if( cpID != other.cpID )
        return cpID < other.cpID;
    if( stype != other.stype )
        return stype < other.stype;
    if( inheritedStype != other.inheritedStype )
        return inheritedStype < other.inheritedStype;
    return allDeviceListIndex < other.allDeviceListIndex;
}
// -----------------------------------------------------------------------------
bool SBTManager::ProgramEntriesKey::operator<( const ProgramEntriesKey& other ) const
{
    if( cpID != other.cpID )
        return cpID < other.cpID;
    if( stype != other.stype )
        return stype < other.stype;
    return inheritedStype < other.inheritedStype;
}


// SBT printing
//------------------------------------------------------------------------------
namespace {
constexpr unsigned COL_WIDTH       = 24U;
constexpr unsigned SHORT_COL_WIDTH = 10U;
}  // namespace
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
static void separator( std::ostream& out, char type )
{
    out << std::setfill( type ) << std::setw( COL_WIDTH * 4 + 20 + 1 ) << "" << std::setfill( ' ' ) << '\n';
}
//------------------------------------------------------------------------------
static void printCol( std::ostream& out, const std::string& c )
{
    out << "| " << std::left << std::setw( COL_WIDTH ) << c;
}
//------------------------------------------------------------------------------
static void printShortCol( std::ostream& out, const std::string& c )
{
    out << "| " << std::left << std::setw( SHORT_COL_WIDTH ) << c;
}
//------------------------------------------------------------------------------
static void printRow( std::ostream&      out,
                      const std::string& c1,
                      const std::string& c2,
                      const std::string& c3,
                      const std::string& c4,
                      const std::string& c5 = "" )
{
    printCol( out, c1 );
    printShortCol( out, c2 );
    printCol( out, c3 );
    printCol( out, c4 );
    printCol( out, c5 );
    out << '\n';
}
//------------------------------------------------------------------------------
static std::string addressString( size_t* catBase, size_t ptr )
{
    std::string str = "  " + hexString( ptr );
    if( catBase )
        str += "  " + hexString( ptr - *catBase, 2 );
    return str;
}

//------------------------------------------------------------------------------
void SBTManager::dumpSBT( std::ostream& out )
{
    std::vector<unsigned> deviceIndices;
    out << "Devices: ";
    const std::vector<Device*>& devices = m_context->getDeviceManager()->activeDevices();
    for( auto iter = devices.begin(); iter != devices.end(); ++iter )
    {
        deviceIndices.push_back( ( *iter )->allDeviceListIndex() );
        if( iter != devices.begin() )
            out << ", ";
        out << ( *iter )->allDeviceListIndex();
    }
    out << "\n\n";

    out << "Record size: " << getSBTRecordSize() << "\n"
        << "rayGenAlloc: { start: " << *m_rayGenAllocation.index << ", size: " << m_rayGenAllocation.size
        << ", capacity: " << m_rayGenAllocation.capacity << " }\n"
        << "exceptionAlloc: { start: " << *m_exceptionAllocation.index << ", size: " << m_exceptionAllocation.size
        << ", capacity: " << m_exceptionAllocation.capacity << " }\n"
        << "missAlloc: { start: " << *m_missAllocation.index << ", size: " << m_missAllocation.size
        << ", capacity: " << m_missAllocation.capacity << " }\n"
        << "sbtRecordStride: " << m_context->getRayTypeCount() << "\n\n";

    separator( out, '-' );
    printRow( out, "Offset", "Size", "Record Type", "Entry Type", "Value" );

    std::vector<SBTRecordRegion> records = getSortedRegions();

    std::array<std::string, 5> typeNames = {{"Ray Generation", "Exception", "Miss", "GI Instance", "Callable Program"}};

    size_t lastRecordEnd = 0;
    size_t lwrrentOffset = 0;
    for( const SBTRecordRegion& record : records )
    {
        lwrrentOffset = *record.range.index * getSBTRecordSize();
        // Check for a hole between the previous record and the current one
        if( lastRecordEnd && record.range.size )
        {
            size_t diff = lwrrentOffset - lastRecordEnd;
            if( diff )
            {
                printCol( out, addressString( nullptr, lastRecordEnd ) );
                printShortCol( out, std::to_string( diff ) );
                printCol( out, "------- Padding -------" );
                printCol( out, "" );
                out << '\n';
            }
        }

        separator( out, '=' );

        size_t rangeBase = lwrrentOffset;

        for( size_t recordOffset = 0; recordOffset < record.range.size; ++recordOffset )
        {
            size_t recordIndex = *record.range.index + recordOffset;

            size_t recordBase = recordIndex * getSBTRecordSize();

            printCol( out, hexString( recordBase ) + "  " + hexString( recordBase - rangeBase ) );
            printShortCol( out, std::to_string( getSBTRecordSize() ) );
            printCol( out, typeNames[record.type] );
            printCol( out, "" );
            printCol( out, "" );

            out << '\n';

            lwrrentOffset = printRecordHeader( out, recordBase, lwrrentOffset, recordIndex, record.type, deviceIndices );

            lwrrentOffset = printRecordData( out, recordBase, lwrrentOffset, recordIndex, record.type );

            separator( out, '=' );
        }
        lastRecordEnd = lwrrentOffset;
    }
}

//------------------------------------------------------------------------------
std::vector<SBTManager::SBTRecordRegion> SBTManager::getSortedRegions()
{
    // Sort SBT records based on their index in order to be able to print them in the order they are in memory.
    std::vector<SBTRecordRegion> records{{m_rayGenAllocation, Raygen}, {m_exceptionAllocation, Exception}, {m_missAllocation, Miss}};

    ObjectManager*          om       = m_context->getObjectManager();
    ReusableIDMap<Program*> programs = om->getPrograms();
    for( const Program* program : programs )
    {
        size_t recordIndex = program->getSBTRecordIndex();
        if( recordIndex != static_cast<size_t>( -1 ) )
        {
            size_t recordCount = 0;
            if( program->isBindless() || program->isUsedAsBoundingBoxProgram() )
            {
                recordCount = 1;
            }
            else
            {
                recordCount = ST_BINDLESS_CALLABLE_PROGRAM + 1;
            }
            records.push_back( {{std::make_shared<size_t>( recordIndex ), recordCount, recordCount}, Callable} );
        }
    }

    // Create a range for each GeometryGroup and add to the records to sort.
    for( const GeometryGroupRegistration& gr : m_GGregistrations )
    {
        size_t recordCount = 0;
        for( int giIndex = 0, numGIs = gr.gg->getChildCount(); giIndex < numGIs; ++giIndex )
        {
            LexicalScope*     child = gr.gg->getChild( giIndex );
            GeometryInstance* gi    = managedObjectCast<GeometryInstance>( child );
            RT_ASSERT( gi );
            recordCount += gi->getMaterialCount() * m_context->getRayTypeCount();
        }
        size_t recordIndex = (size_t)gr.gg->getSBTRecordIndex();
        records.push_back( {{std::make_shared<size_t>( recordIndex ), recordCount, recordCount}, Instance} );
    }

    // Sort by start index of the records' ranges
    algorithm::sort( records, []( const SBTRecordRegion& lhs, const SBTRecordRegion& rhs ) {
        return *lhs.range.index < *rhs.range.index;
    } );

    return records;
}

//------------------------------------------------------------------------------
void SBTManager::printDeviceValues( std::ostream& out, size_t recordIndex, size_t headerEntry, const std::vector<unsigned>& deviceIndices )
{
    std::ostringstream str;
    for( unsigned deviceIndex : deviceIndices )
    {

        const short* pHeader = reinterpret_cast<const short*>( getSBTDdDataHostPointerReadOnly( recordIndex, deviceIndex ) );

        str << std::to_string( pHeader[headerEntry] ) << " ";
    }
    printCol( out, str.str() );
}

//------------------------------------------------------------------------------
size_t SBTManager::printRecordHeader( std::ostream&                out,
                                      size_t                       recordBase,
                                      size_t                       lwrrentOffset,
                                      size_t                       recordIndex,
                                      SBTRecordType                type,
                                      const std::vector<unsigned>& deviceIndices )
{
    std::array<std::array<std::string, 3>, 5> typeStrings = {
        {{{"RG stateId [per device]", "unused", "unused"}},
         {{"EX stateId [per device]", "unused", "unused"}},
         {{"MS stateId [per device]", "unused", "unused"}},
         {{"CH stateId [per device]", "AH stateId [per device]", "IS stateId [per device]"}},
         {{"CC stateId [per device]", "DC stateId [per device]", "unused"}}}};
    const short* pHeader = reinterpret_cast<const short*>( getSBTDdDataHostPointerReadOnly( recordIndex, 0 ) );

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    size_t entrySize = (char*)&pHeader[1] - (char*)pHeader;
    lwrrentOffset += entrySize;
    printShortCol( out, std::to_string( entrySize ) );
    printCol( out, " " );
    printCol( out, typeStrings[type][0] );
    printDeviceValues( out, recordIndex, 0, deviceIndices );
    out << '\n';

    for( size_t headerEntry = 1; headerEntry < 3; ++headerEntry )
    {
        printCol( out, addressString( &recordBase, lwrrentOffset ) );
        entrySize = (char*)&pHeader[headerEntry + 1] - (char*)&pHeader[headerEntry];
        lwrrentOffset += entrySize;
        printShortCol( out, std::to_string( entrySize ) );
        printCol( out, " " );
        printCol( out, typeStrings[type][headerEntry] );

        if( type == Instance || ( type == Callable && headerEntry == 1 ) || k_dumpUnusedSBTValues.get() )
        {
            printDeviceValues( out, recordIndex, headerEntry, deviceIndices );
        }
        else
        {
            printCol( out, "" );
        }

        out << '\n';
    }

    separator( out, '-' );

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    entrySize = ( (char*)pHeader + m_sbtRecordHeaderSize ) - (char*)&pHeader[3];
    lwrrentOffset += entrySize;

    printShortCol( out, std::to_string( entrySize ) );
    printCol( out, "------- Padding -------" );
    printCol( out, " " );
    out << '\n';
    return lwrrentOffset;
}


//------------------------------------------------------------------------------
size_t SBTManager::printRecordData( std::ostream& out, size_t recordBase, size_t lwrrentOffset, size_t recordIndex, SBTRecordType type )
{
    std::array<std::array<std::string, 3>, 5> typeStrings = {{
        {{"ProgramHandle", "unused", "unused"}},
        {{"ProgramHandle", "unused", "unused"}},
        {{"ProgramHandle", "unused", "unused"}},
        {{"GIHandle", "MaterialHandle", "Skip"}},
        {{"ProgramHandle", "unused", "unused"}},
    }};

    const cort::SBTRecordData* sbtData = getSBTDiDataHostPointerReadOnly( recordIndex );

    separator( out, '-' );

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    size_t entrySize = (char*)&sbtData->GeometryInstanceData.materialOffset - (char*)&sbtData->ProgramData.programOffset;
    lwrrentOffset += entrySize;
    printShortCol( out, std::to_string( entrySize ) );
    printCol( out, " " );
    printCol( out, typeStrings[type][0] );
    printCol( out, hexString( sbtData->ProgramData.programOffset, 4 ) );
    out << '\n';

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    entrySize = (char*)&sbtData->GeometryInstanceData.materialOffset - (char*)&sbtData->ProgramData.programOffset;
    lwrrentOffset += entrySize;
    printShortCol( out, std::to_string( entrySize ) );
    printCol( out, " " );
    printCol( out, typeStrings[type][1] );
    if( type == Instance || k_dumpUnusedSBTValues.get() )
    {
        printCol( out, hexString( sbtData->GeometryInstanceData.materialOffset, 4 ) );
    }
    else
    {
        printCol( out, "" );
    }
    out << '\n';

    char* recordEnd = (char*)sbtData + sizeof( cort::SBTRecordData );

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    entrySize = recordEnd - (char*)&sbtData->GeometryInstanceData.skip;
    lwrrentOffset += entrySize;

    printShortCol( out, std::to_string( recordEnd - (char*)&sbtData->GeometryInstanceData.skip ) );
    printCol( out, " " );
    printCol( out, typeStrings[type][2] );

    entrySize = recordBase + getSBTRecordSize() - lwrrentOffset;

    if( type == Instance )
    {
        std::string jumpAddr;
        if( sbtData->GeometryInstanceData.skip != 0 )
        {
            size_t skip       = sbtData->GeometryInstanceData.skip * m_context->getRayTypeCount() * getSBTRecordSize();
            size_t nextOffset = recordBase + skip;
            jumpAddr          = " (Target: " + addressString( nullptr, nextOffset ) + ")";
        }
        printCol( out, std::to_string( sbtData->GeometryInstanceData.skip ) + jumpAddr );
    }
    else if( k_dumpUnusedSBTValues.get() )
    {
        printCol( out, std::to_string( sbtData->GeometryInstanceData.skip ) );
    }
    else
    {
        printCol( out, "" );
    }

    out << '\n';

    separator( out, '-' );

    printCol( out, addressString( &recordBase, lwrrentOffset ) );
    lwrrentOffset += entrySize;

    printShortCol( out, std::to_string( entrySize ) );
    printCol( out, "------- Padding -------" );
    printCol( out, " " );

    out << '\n';
    return lwrrentOffset;
}

//------------------------------------------------------------------------------
void SBTManager::SBTRange::reset()
{
    index.reset();
    size     = 0;
    capacity = 0;
}

//------------------------------------------------------------------------------
void SBTManager::SBTIndex::reset()
{
    if( m_lwrrentParent )
    {
        m_lwrrentParent->removeItem( this );
        m_lwrrentParent = nullptr;
    }
}
