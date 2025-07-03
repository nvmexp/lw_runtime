#include <ExelwtionStrategy/Common/ConstantMemoryPlan.h>

#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Context/TableManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/CommonRuntime.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/Plan.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/math/Bits.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/Constant.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

#include <sstream>

using namespace corelib;
using namespace prodlib;
using namespace llvm;
using namespace optix;

namespace {
// clang-format off
  Knob<size_t> k_constObjectRecordSizeIncrement( RT_DSTRING("constantMemory.constObjectRecordSizeIncrement"), 1024, RT_DSTRING( "Allocate object record in [N] byte chunks when putting into constant memory. 0 disabled using constant memory. Should be power of 2." ) );
  Knob<int> k_constBufferSizeGranularity(  RT_DSTRING("constantMemory.constBufferSizeGranularity"),  32, RT_DSTRING( "Allocate space in [N] buffer header size chunks when putting in constant memory. 0 disables using constant memory.") );
  Knob<int> k_constProgramSizeGranularity( RT_DSTRING("constantMemory.constProgramSizeGranularity"),  8, RT_DSTRING( "Allocate space in [N] program header size chunks when putting in constant memory. 0 disables using constant memory.") );
  Knob<int> k_constTextureSizeGranularity( RT_DSTRING( "constantMemory.constTextureSizeGranularity" ), 16, RT_DSTRING( "Allocate space in [N] texture header size chunks when putting in constant memory. 0 disables using constant memory." ) );
  Knob<int> k_constTraversableSizeGranularity( RT_DSTRING( "constantMemory.constTraversableSizeGranularity" ), 8, RT_DSTRING( "Allocate space in [N] texture header size chunks when putting in constant memory. 0 disables using constant memory." ) );
// clang-format on
}  // namespace

ConstantMemoryPlan::ConstantMemoryPlan( Plan*                       plan,
                                        Context*                    context,
                                        const PerDeviceProgramPlan& perDeviceProgramPlan,
                                        size_t                      constBytesToReserve,
                                        bool                        deduplicateConstants )
    : m_plan( plan )
    , m_context( context )
{
    m_context->getUpdateManager()->registerUpdateListener( this );
    createPlan( perDeviceProgramPlan, constBytesToReserve, deduplicateConstants );
}

ConstantMemoryPlan::~ConstantMemoryPlan()
{
    m_context->getUpdateManager()->unregisterUpdateListener( this );
}

const ConstantMemAllocations& ConstantMemoryPlan::getAllocationInfo()
{
    return m_constMemAllocs;
}

std::string ConstantMemoryPlan::summaryString() const
{
    std::ostringstream out;
    out << " {constMem:"
        << " Static:" << m_constMemAllocs.staticInitializers << " StrGlo:" << m_constMemAllocs.structGlobalSize
        << " TravTab:" << m_constMemAllocs.traversableTableSize << " ObjRec:" << m_constMemAllocs.objectRecordSize
        << " BufTab:" << m_constMemAllocs.bufferTableSize << " TexTab:" << m_constMemAllocs.textureTableSize
        << " PrgTab:" << m_constMemAllocs.programTableSize << "}";
    return out.str();
}

void ConstantMemoryPlan::createPlan( const PerDeviceProgramPlan& perDeviceProgramPlan, size_t constBytesToReserve, bool deduplicateConstants )
{
    ProgramManager* programManager = m_context->getProgramManager();

    // The amount of available constant memory is the minimum amount across all the unique devices.
    int constMemSize = std::numeric_limits<int>::max();
    for( auto& iter : perDeviceProgramPlan )
    {
        LWDADevice* device = iter.first;
        constMemSize       = std::min( constMemSize, device->lwdaDevice().TOTAL_CONSTANT_MEMORY() );
    }

    m_constMemAllocs.remainingSize = constMemSize;

    // Reserve bytes
    RT_ASSERT_MSG( m_constMemAllocs.remainingSize >= constBytesToReserve,
                   "Not enough constant memory to satisfy reservation" );
    m_constMemAllocs.remainingSize -= constBytesToReserve;

    // Compute the size necessary for any data already allocated in constant memory in the
    // incoming code.  Note that we aren't actually doing any of the work to uniquify them,
    // simply because during planning we don't make modifications to the canonical code.
    size_t staticInitializersSize = 0;

    for( auto& iter : perDeviceProgramPlan )
    {
        ProgramPlan* programPlan = iter.second;

        std::set<const Constant*> staticInitializers;
        size_t                    deviceStaticInitializerSize = 0;
        // Iterate over the canonical programs used by the current device.
        for( const CanonicalProgramID cpID : programPlan->getAllReachablePrograms() )
        {
            const CanonicalProgram* cp     = programManager->getCanonicalProgramById( cpID );
            const Function*         fn     = cp->llvmFunction();
            const Module*           module = fn->getParent();
            DataLayout              DL( module );
            for( Module::const_global_iterator G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
            {
                // Look for globals with initializers, declared in constant memory, and has at least one use
                if( includeForUniquifyingConstMemoryInitializers( &*G ) )
                {
                    auto inserted_pair = staticInitializers.insert( G->getInitializer() );

                    // Count the constant if we are not deduplicating the constants,
                    // or if this is the first time the constant was encountered
                    if( deduplicateConstants == false || inserted_pair.second == true )
                    {
                        // It was inserted, count it
                        unsigned int eltSize = DL.getTypeStoreSize( G->getInitializer()->getType() );
                        // Align each item to CONST_MEMORY_ALIGNMENT just to be safe, since we don't know how the driver will pack these.
                        deviceStaticInitializerSize += align( eltSize, CONST_MEMORY_ALIGNMENT );
                    }
                }
            }
        }

        staticInitializersSize = std::max( staticInitializersSize, deviceStaticInitializerSize );
    }

    if( m_constMemAllocs.remainingSize >= staticInitializersSize )
    {
        m_constMemAllocs.staticInitializers = staticInitializersSize;
        m_constMemAllocs.remainingSize -= staticInitializersSize;
    }
    else
    {
        std::ostringstream o;
        o << "Constant memory size required by input PTX for static initialization (" << staticInitializersSize
          << ") exceeds amount available on device (" << m_constMemAllocs.remainingSize << ")";
        // Gold star for looping over all the memory again and reporting all the memory
        // variables' sizes, names and locations.
        throw CompileError( RT_EXCEPTION_INFO, o.str() );
    }

    const size_t global_struct_size = align( sizeof( cort::Global ), CONST_MEMORY_ALIGNMENT );
    if( m_constMemAllocs.remainingSize >= global_struct_size )
    {
        m_constMemAllocs.structGlobalSize = global_struct_size;
        m_constMemAllocs.remainingSize -= global_struct_size;
    }
    else
    {
        std::ostringstream o;
        o << "Insufficient remaining constant memory for global struct.  Try removing statically initialized constant "
             "memory from input PTX (static memory requirements: "
          << staticInitializersSize << ", global struct size: " << global_struct_size
          << ", total limit: " << constMemSize << ")";
        throw CompileError( RT_EXCEPTION_INFO, o.str() );
    }

    // Now for the optional parts (i.e. space-available).

    TableManager* tm = m_context->getTableManager();

    if( k_constTraversableSizeGranularity.get() > 0 && m_context->useRtxDataModel() && !m_context->getPreferFastRecompiles() )
    {
        // Use roundUp when the multiple may not be a power of two.
        const size_t table_size = align( roundUp( tm->getTraversableTableSizeInBytes(),
                                                  k_constTraversableSizeGranularity.get() * sizeof( RtcTraversableHandle ) ),
                                         CONST_MEMORY_ALIGNMENT );
        if( m_constMemAllocs.remainingSize >= table_size )
        {
            m_constMemAllocs.traversableTableSize = table_size;
            m_constMemAllocs.remainingSize -= table_size;
        }
    }

    if( k_constObjectRecordSizeIncrement.get() > 0 && !m_context->getPreferFastRecompiles() )
    {
        const size_t object_record_size = align( tm->getObjectRecordSize(), k_constObjectRecordSizeIncrement.get() );
        if( m_constMemAllocs.remainingSize >= object_record_size )
        {
            m_constMemAllocs.objectRecordSize = object_record_size;
            m_constMemAllocs.remainingSize -= object_record_size;
        }
    }

    // For the following buffers their sizes can depend on the device.
    // Iterate over all unique devices and consider the maximum size across all the devices.

    if( k_constBufferSizeGranularity.get() > 0 && !m_context->getPreferFastRecompiles()  )
    {
        // Use roundUp when the multiple may not be a power of two.
        const size_t bufferTableSize =
            align( roundUp( tm->getBufferHeaderTableSizeInBytes(), k_constBufferSizeGranularity.get() * sizeof( cort::Buffer ) ),
                   CONST_MEMORY_ALIGNMENT );
        if( m_constMemAllocs.remainingSize >= bufferTableSize )
        {
            m_constMemAllocs.bufferTableSize = bufferTableSize;
            m_constMemAllocs.remainingSize -= bufferTableSize;
        }
    }

    if( k_constTextureSizeGranularity.get() > 0 && !m_context->getPreferFastRecompiles() )
    {
        // Use roundUp when the multiple may not be a power of two.
        const size_t textureTableSize = align( roundUp( tm->getTextureHeaderTableSizeInBytes(),
                                                        k_constTextureSizeGranularity.get() * sizeof( cort::TextureSampler ) ),
                                               CONST_MEMORY_ALIGNMENT );
        if( m_constMemAllocs.remainingSize >= textureTableSize )
        {
            m_constMemAllocs.textureTableSize = textureTableSize;
            m_constMemAllocs.remainingSize -= textureTableSize;
        }
    }

    if( k_constProgramSizeGranularity.get() > 0 && !m_context->getPreferFastRecompiles() )
    {
        const size_t programTableSize = align( roundUp( tm->getProgramHeaderTableSizeInBytes(),
                                                        k_constProgramSizeGranularity.get() * sizeof( cort::ProgramHeader ) ),
                                               CONST_MEMORY_ALIGNMENT );
        if( m_constMemAllocs.remainingSize >= programTableSize )
        {
            m_constMemAllocs.programTableSize = programTableSize;
            m_constMemAllocs.remainingSize -= programTableSize;
        }
    }

    llog( 20 ) << "Planning to use " << ( constMemSize - m_constMemAllocs.remainingSize ) << " bytes of "
               << constMemSize << " constant memory.\n";
}

// -----------------------------------------------------------------------------
bool ConstantMemoryPlan::isCompatibleWith( const ConstantMemoryPlan& otherPlan ) const
{
    return m_constMemAllocs == otherPlan.m_constMemAllocs;
}

// -----------------------------------------------------------------------------
bool ConstantMemoryPlan::tableSizeChanged( size_t tableSize, size_t newTableSize )
{
    // 0 indicates that the object records are stored in constant memory.
    // Always consider the table size to have changed in this case.
    return tableSize != 0 && tableSize != newTableSize;
}

// -----------------------------------------------------------------------------
void ConstantMemoryPlan::eventTableManagerObjectRecordResized( size_t oldSize, size_t newSize )
{
    if( !m_plan->isValid() )
        return;

    // Don't bother doing this if we aren't actually trying to do this
    if( k_constObjectRecordSizeIncrement.get() == 0 || m_context->getPreferFastRecompiles() )
        return;

    // Compare newSize to see if it will fit in oldSize, ilwalidate if so
    const size_t objectRecordSize = align( newSize, k_constObjectRecordSizeIncrement.get() );

    if( tableSizeChanged( m_constMemAllocs.objectRecordSize, objectRecordSize ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}

// -----------------------------------------------------------------------------
void ConstantMemoryPlan::eventTableManagerBufferHeaderTableResized( size_t oldSize, size_t newSize )
{
    if( !m_plan->isValid() )
        return;

    // Don't bother doing this if we aren't actually trying to do this
    if( k_constBufferSizeGranularity.get() == 0 || m_context->getPreferFastRecompiles() )
        return;

    // Compare newSize to see if it will fit in oldSize, ilwalidate if so

    const size_t bufferTableSize =
        align( roundUp( newSize, k_constBufferSizeGranularity.get() * sizeof( cort::Buffer ) ), CONST_MEMORY_ALIGNMENT );

    if( tableSizeChanged( m_constMemAllocs.bufferTableSize, bufferTableSize ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}

// -----------------------------------------------------------------------------
void ConstantMemoryPlan::eventTableManagerProgramHeaderTableResized( size_t oldSize, size_t newSize )
{
    if( !m_plan->isValid() )
        return;

    // Don't bother doing this if we aren't actually trying to do this
    if( k_constProgramSizeGranularity.get() == 0 || m_context->getPreferFastRecompiles() )
        return;

    // Compare newSize to see if it will fit in oldSize, ilwalidate if so
    const size_t programTableSize =
        align( roundUp( newSize, k_constProgramSizeGranularity.get() * sizeof( cort::ProgramHeader ) ), CONST_MEMORY_ALIGNMENT );

    if( tableSizeChanged( m_constMemAllocs.programTableSize, programTableSize ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}

// -----------------------------------------------------------------------------
void ConstantMemoryPlan::eventTableManagerTextureHeaderTableResized( size_t oldSize, size_t newSize )
{
    if( !m_plan->isValid() )
        return;

    // Don't bother doing this if we aren't actually trying to do this
    if( k_constTextureSizeGranularity.get() == 0 || m_context->getPreferFastRecompiles() )
        return;

    // Compare newSize to see if it will fit in oldSize, ilwalidate if so
    const size_t textureTableSize =
        align( roundUp( newSize, k_constTextureSizeGranularity.get() * sizeof( cort::TextureSampler ) ), CONST_MEMORY_ALIGNMENT );

    if( tableSizeChanged( m_constMemAllocs.textureTableSize, textureTableSize ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}

// -----------------------------------------------------------------------------
void optix::ConstantMemoryPlan::eventTableManagerTraversableHeaderTableResized( size_t oldSize, size_t newSize )
{
    if( !m_plan->isValid() )
        return;

    // We only care about this for the Rtx data model
    if( !m_context->useRtxDataModel() )
        return;

    // Don't bother doing this if we aren't actually trying to do this
    if( k_constTraversableSizeGranularity.get() == 0 || m_context->getPreferFastRecompiles() )
        return;

    // Compare newSize to see if it will fit in oldSize, ilwalidate if so
    const size_t traversableTableSize =
        align( roundUp( newSize, k_constTraversableSizeGranularity.get() * sizeof( RtcTraversableHandle ) ), CONST_MEMORY_ALIGNMENT );

    if( tableSizeChanged( m_constMemAllocs.traversableTableSize, traversableTableSize ) )
    {
        m_plan->ilwalidatePlan();
        return;
    }
}

// -----------------------------------------------------------------------------
void optix::ConstantMemoryPlan::eventContextSetPreferFastRecompiles( bool /*oldValue*/, bool /*newValue*/ )
{
    // This is expected to happen rarely, so don't bother with a detailed comparison.
    m_plan->ilwalidatePlan();
}
