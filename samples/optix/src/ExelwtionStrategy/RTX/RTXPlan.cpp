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

#include <ExelwtionStrategy/RTX/RTXPlan.h>

#include <g_lwconfig.h>

#include <Context/Context.h>
#include <Context/LLVMManager.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/RTCore.h>
#include <Context/SBTManager.h>
#include <Context/TableManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/Common/ConstantMemAllocations.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/RTX/RTXCompile.h>
#include <ExelwtionStrategy/RTX/RTXDemandBufferSpecializer.h>
#include <ExelwtionStrategy/RTX/RTXExceptionInstrumenter.h>
#include <ExelwtionStrategy/RTX/RTXFrameTask.h>
#include <ExelwtionStrategy/RTX/RTXSpecializer.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/LineInfo.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MemoryManager.h>
#include <ThreadPool/LockGuard.h>
#include <ThreadPool/ThreadPool.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/LWML.h>
#include <Util/PersistentStream.h>
#include <Util/PrintStream.h>
#include <Util/RecordCompile.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/ForceDeprecatedCompiler.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/math/MathUtil.h>
#include <corelib/system/System.h>
#include <corelib/system/Timer.h>
#include <prodlib/compiler/ModuleCache.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/math/Bits.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <exception>
#include <sstream>
#include <string>

namespace {
// clang-format off
Knob<bool> k_skipSpecialization( RT_DSTRING( "rtx.skipSpecialization" ), false, RT_DSTRING( "Skip specializations when compiling to RTX" ) );
Knob<int>  k_sassOptimization( RT_DSTRING("lwca.sassOptimization"), -1, RT_DSTRING( "Sass optimization flag value. -1 = don't specify optimization explicitly" ) );
Knob<std::string>  k_pipelineType( RT_DSTRING( "rtx.pipelineType" ), "megakernel_simple", RT_DSTRING( "The RTcore pipeline type to use. See rtcore/interface/types.h for a list of options." ) );
HiddenPublicKnob<std::string> k_traversalOverride( RT_PUBLIC_DSTRING( "rtx.traversalOverride" ), "", RT_PUBLIC_DSTRING( "Override traversal to the given traversal" ) );
// 128 is lwrrently the best known value. we'll later have some SM-specific default like for MK
Knob<int>  k_maxRegCount( RT_DSTRING("rtcore.maxRegCount"), 128, RT_DSTRING( "Set max reg count for rtcore. 0 = let OptiX figure out a good value based on GPU architecture. -1 = don't limit registers." ) );
Knob<int>  k_maxAttributeRegCount( RT_DSTRING( "rtx.maxAttributeRegCount" ), 2, RT_DSTRING( "Maximum registers for attribute passing. Attributes beyond the limit are passed on the stack. -1 = use rtcore limit." ) );
Knob<int>  k_maxParamRegCount( RT_DSTRING( "rtx.maxParamRegCount" ), 2, RT_DSTRING( "Maximum registers for callable program arguments. Arguments beyond the limit are passed on the stack. -1 = use rtcore limit." ) );
Knob<int>  k_maxPayloadRegCount( RT_DSTRING( "rtx.maxPayloadRegCount" ), 7, RT_DSTRING( "Maximum registers for payload passing. Payloads beyond the limit are passed on the stack. -1 = use rtcore limit." ) );
PublicKnob<int>  k_continuationStackSize( RT_PUBLIC_DSTRING( "rtx.continuationStackSize" ), -1, RT_PUBLIC_DSTRING( "RTX continuation stack size. -1 = compute continuation stack size." ) );
PublicKnob<int>  k_maxBindlessCallableDepth( RT_PUBLIC_DSTRING( "rtx.maxBindlessCallableDepth" ), -1, RT_PUBLIC_DSTRING( "Maximum depth of bindless callable programs. -1 = use default." ) );
PublicKnob<bool> k_parallelCompilation(RT_PUBLIC_DSTRING( "rtx.parallelCompilation" ), true, RT_PUBLIC_DSTRING( "Enable parallel compilation for RTXPlan::compileProgramToRTX()" ) );
Knob<std::string> k_saveLLVM( RT_DSTRING("rtx.saveLLVM"), "", RT_DSTRING( "Save LLVM stages during compilation" ) );
Knob<bool>        k_cacheEnabled( RT_DSTRING( "diskcache.rtx.enabled" ), true, RT_DSTRING( "Enable or disable the disk cache for RTX programs." ) );
Knob<bool>        k_enableCoroutines( RT_DSTRING( "compile.enableCoroutines" ), false, RT_DSTRING( "Enable coroutines." ) );

// Knob precedence (in decreasing priority, default value counts as unset):
// - enable individuals
// - enable all
// - disable all
PublicKnob<bool> k_enableAllExceptions( RT_PUBLIC_DSTRING("compile.enableAllExceptions"), false, RT_PUBLIC_DSTRING("Force creation of exception handling code for all exceptions that we support." ) );
PublicKnob<bool> k_disableAllExceptions( RT_PUBLIC_DSTRING("compile.disableAllExceptions"), false, RT_PUBLIC_DSTRING("Do not create code to handle any exceptions." ) );
Knob<bool> k_enableStackOverflowHandling( RT_DSTRING("compile.enableStackOverflowHandling"), false, RT_DSTRING("Create code to handle stack overflow exceptions." ) );
Knob<bool> k_enableTraceDepthHandling( RT_DSTRING("compile.enableTraceDepthHandling"), false, RT_DSTRING("Create code to handle trace depth exceptions." ) );
Knob<bool> k_enableIlwalidBufferIdHandling( RT_DSTRING("compile.enableIlwalidBufferIdHandling"), false, RT_DSTRING("Create code to handle invalid buffer id exceptions." ) );
Knob<bool> k_enableIlwalidTextureIdHandling( RT_DSTRING("compile.enableIlwalidTextureIdHandling"), false, RT_DSTRING("Create code to handle invalid texture id exceptions." ) );
Knob<bool> k_enableIlwalidProgramIdHandling( RT_DSTRING("compile.enableIlwalidProgramIdHandling"), false, RT_DSTRING("Create code to handle invalid program id exceptions." ) );
Knob<bool> k_enableBufferIndexOutOfBoundsHandling( RT_DSTRING("compile.enableBufferIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle buffer index out of bounds exceptions." ) );
Knob<bool> k_enableIndexOutOfBoundsHandling( RT_DSTRING("compile.enableIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle index out of bounds exceptions." ) );
Knob<bool> k_enableIlwalidRayHandling( RT_DSTRING("compile.enableIlwalidRayHandling"), false, RT_DSTRING("Create code to handle invalid ray exceptions." ) );
Knob<bool> k_enablePayloadAccessOutOfBoundsHandling( RT_DSTRING("compile.enablePayloadAccessOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle payload offset out of bounds exceptions." ) );

PublicKnob<bool> k_enableRTcoreProfiling( RT_PUBLIC_DSTRING("rtx.enableRTcoreProfiling"), false, RT_PUBLIC_DSTRING( "Enables or disables RTcore profiling. If enabled the profiling data of each device will be written to disk.") );
HiddenPublicKnob<int> k_maxTraceRelwrsionDepth( RT_PUBLIC_DSTRING("rtx.maxTraceRelwrsionDepth"), -1, RT_PUBLIC_DSTRING( "Maximum relwrsion depth of trace calls. -1 = use default." ) );

Knob<bool> k_recordCompileCalls( RT_DSTRING("compile.recordCalls"), false, RT_DSTRING("Save calls to rtcCompileModule." ) );

Knob<std::string> k_limitActiveLaunchIndices( RT_DSTRING( "launch.limitActiveIndices" ), "", RT_DSTRING( "When specified limit which launch indices are active. Syntax: [minX, maxX], [minY, maxY]" ) );

PublicKnob<bool> k_enableD2IR( RT_PUBLIC_DSTRING( "compile.enableFeature.newBackend" ), true, RT_PUBLIC_DSTRING( "Enable new compiler backend." ) );
// clang-format on
}  // namespace

using namespace corelib;
using namespace prodlib;
using namespace llvm;
using namespace optix;

//
// RTXPlan implementation
//

RTXPlan::RTXPlan( Context* context, CompiledProgramCache* compiledProgramCache, const DeviceSet& devices, int numLaunchDevices )
    : Plan( context, devices )
    , m_moduleCache( std::unique_ptr<prodlib::ModuleCache>() )
    , m_numLaunchDevices( numLaunchDevices )
    , m_needsUniversalTraversal( m_context->RtxUniversalTraversalEnabled() )
    , m_hasMotionBlur( m_context->RtxMotionBlurEnabled() )
    , m_compiledProgramCache( compiledProgramCache )
    , m_pagingMode( m_context->getPagingManager()->getLwrrentPagingMode() )
{
}

std::string RTXPlan::summaryString() const
{
    std::ostringstream out;
    out << "RTXPlan: entry(" << m_entry << "), ";

    for( int i = 0; i < (int)m_perUniqueDevice.size(); ++i )
    {
        const PerUniqueDevice& pud = m_perUniqueDevice[i];
        out << " devices[";
        bool first = true;
        for( int allDeviceListIndex : m_devices )
        {
            Device* device = m_context->getDeviceManager()->allDevices()[allDeviceListIndex];
            if( device->uniqueDeviceListIndex() == (unsigned int)i )
            {
                if( !first )
                    out << ", ";
                first = false;
                out << device->activeDeviceListIndex() << ":" << device->deviceName();
            }
        }
        out << "]";
        out << " numLaunchDevices: " << m_numLaunchDevices;

        out << " " << pud.m_programPlan->summaryString();

        out << " {\n";
        out << " boundTex:" << pud.m_numBoundTextures;
        out << " canPromotePayload: " << pud.m_canPromotePayload;
        out << " maxPayloadRegisterCount: " << pud.m_maxPayloadRegisterCount;
        out << " maxAttributeRegisterCount: " << pud.m_maxAttributeRegisterCount;
        out << " maxCallableProgramParamRegisterCount: " << pud.m_maxCallableProgramParamRegisterCount;
        out << " pagingMode: " << m_pagingMode;

        // TODO(bigler): when we add per CanonicalProgram attribute decoders, we need to print out that data here
        //               For now we can assume that all relavant information is in the set of IS/AP programs.
        out << "}";
    }

    out << m_constantMemoryPlan->summaryString() << "\n";

    out << m_specializationPlan->summaryString();

    out << status() << '\n';
    return out.str();
}

bool RTXPlan::supportsLaunchConfiguration( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const
{
    // Make sure the plan includes the entry point
    if( entry != m_entry )
        return false;

    // optixi_getLaunchIndex() is specialized depending on dimensionality == 3.
    if( ( m_dimensionality == 3 ) != ( dimensionality == 3 ) )
        return false;

    if( m_devices != devices )
        return false;

    if( m_numLaunchDevices != numLaunchDevices )
        return false;

    return true;
}

bool RTXPlan::isCompatibleWith( const Plan* otherPlan ) const
{
    // Must be the same type of plan
    const RTXPlan* other = dynamic_cast<const RTXPlan*>( otherPlan );
    if( !other )
        return false;

    if( m_entry != other->m_entry )
        return false;

    // optixi_getLaunchIndex() is specialized depending on dimensionality == 3.
    if( ( m_dimensionality == 3 ) != ( other->m_dimensionality == 3 ) )
        return false;

    if( m_numLaunchDevices != other->m_numLaunchDevices )
        return false;

    // Must support the same devices
    if( m_devices != other->m_devices )
        return false;

    if( m_maxTraceDepth != other->m_maxTraceDepth )
        return false;

    if( m_useContextAttributesForStackSize != other->m_useContextAttributesForStackSize )
        return false;

    if( m_needsUniversalTraversal != other->m_needsUniversalTraversal )
        return false;

    if( m_hasMotionBlur != other->m_hasMotionBlur )
        return false;

    if( m_useContextAttributesForStackSize )
    {
        // The stack size attribute is taken into account.
        if( m_continuationStackSize != other->m_continuationStackSize || m_directCallableStackSizeFromTraversal != other->m_directCallableStackSizeFromTraversal
            || m_directCallableStackSizeFromState != other->m_directCallableStackSizeFromState )
        {
            return false;
        }
    }
    else
    {
        if( m_maxCallableProgramDepth != other->m_maxCallableProgramDepth )
            return false;
    }

    if( !m_specializationPlan->isCompatibleWith( *other->m_specializationPlan ) )
        return false;

    if( !m_constantMemoryPlan->isCompatibleWith( *other->m_constantMemoryPlan.get() ) )
        return false;

    if( m_perUniqueDevice.size() != other->m_perUniqueDevice.size() )
        return false;

    for( size_t i = 0; i < m_perUniqueDevice.size(); ++i )
        if( !m_perUniqueDevice[i].isCompatibleWith( other->m_perUniqueDevice[i] ) )
            return false;

    return true;
}

RTXPlan::GetOrCompileProgramJob::GetOrCompileProgramJob( bool                             parallelize,
                                                         const std::vector<InputData>&    input,
                                                         const RTXPlan&                   rtxPlan,
                                                         const Context*                   context,
                                                         LWDADevice*                      lwdaDevice,
                                                         const RTXCompile::CompileParams& rtxParams,
                                                         const RtcCompileOptions&         rtcCompileOptions,
                                                         const AttributeDecoderList&      attributeDecoders,
                                                         int                              maxAttributeRegisterCount,
                                                         const std::set<const CallSiteIdentifier*>& heavyweightCallSites )
    : FragmentedJob( input.size() )
    , m_input( input )
    , m_output( input.size() )
    , m_exception( input.size() )
    // Limit number of threads to 6 since there is some serious lock contention in malloc()/free()
    // via LLVM data structures. See also Jira OP 2286.
    , m_threadLimit( parallelize ? 6 : 1 )
    , m_rtxPlan( rtxPlan )
    , m_context( context )
    , m_lwdaDevice( lwdaDevice )
    , m_rtxParams( rtxParams )
    , m_rtcCompileOptions( rtcCompileOptions )
    , m_attributeDecoders( attributeDecoders )
    , m_maxAttributeRegisterCount( maxAttributeRegisterCount )
    , m_heavyweightCallSites( heavyweightCallSites )
{
}

void RTXPlan::GetOrCompileProgramJob::exelwteFragment( size_t index, size_t count ) noexcept
{
    try
    {
        std::string s = "Fragment " + to_string( index ) + " / " + to_string( count ) + ": "
                        + semanticTypeToString( m_input[index].stype );
        TIMEVIZ_SCOPE( s.c_str() );

        LockGuard<std::mutex>   guard( m_mutex );
        const CanonicalProgram* cp = m_context->getProgramManager()->getCanonicalProgramById( m_input[index].cpId );
        // Thread-safe sections of getOrCompileProgram() unlock and re-lock the mutex.
        m_output[index] = m_rtxPlan.getOrCompileProgram( m_lwdaDevice, m_rtxParams, m_rtcCompileOptions, m_attributeDecoders,
                                                         m_heavyweightCallSites, m_input[index].numConlwrrentLaunchDevices,
                                                         m_input[index].pagingMode, m_input[index].stype,
                                                         m_input[index].inheritedStype, cp, m_mutex );
    }
    catch( ... )
    {
        m_exception[index] = std::lwrrent_exception();
    }
}

const ModuleEntryRefPair& RTXPlan::GetOrCompileProgramJob::getOutput( size_t index )
{
    if( m_exception[index] )
        std::rethrow_exception( m_exception[index] );
    else
        return m_output[index];
}

// Print to both private log and public usage report
namespace {
void logModuleStackSizes( optix::UsageReport& ur, const std::string& moduleName, Rtlw32 directStackFrameSize, Rtlw32 continuationStackFrameSize )
{
    llog( 30 ) << "RTXPLAN: module " << moduleName.c_str() << ": stack size direct=" << directStackFrameSize
               << ",  continuation=" << continuationStackFrameSize << ".\n";

    ureport2( ur, "INFO" ) << "Module: " << moduleName.c_str() << ": stack size (bytes): direct=" << directStackFrameSize
                           << ", continuation=" << continuationStackFrameSize << std::endl;
}
}

void RTXPlan::compile() const
{
    TIMEVIZ_FUNC;

    // Should never call compile after Plan has generated a Task
    RT_ASSERT( !hasBeenCompiled() );

    // Create the frametask for all active devices
    std::unique_ptr<RTXFrameTask> task(
        new RTXFrameTask( m_context, m_devices, {m_entry}, m_maxTransformHeight + m_maxAccelerationHeight ) );

    for( int allDeviceListIndex : m_devices )
    {
        Device*                device     = m_context->getDeviceManager()->allDevices()[allDeviceListIndex];
        const PerUniqueDevice& pud        = m_perUniqueDevice[device->uniqueDeviceListIndex()];
        LWDADevice*            lwdaDevice = deviceCast<LWDADevice>( device );

        // Specify compile options used for all shaders.
        RtcCompileOptions         rtcCompileOptions{};
        RTXCompile::CompileParams rtxParams{};
        determineRtxCompileOptions( lwdaDevice, pud, &rtcCompileOptions, &rtxParams );

        // Compile to sass (or get from the cache) each canonical program in the plan.
        //
        // Keep the list sorted by decreasing average compilation time such that the parallelization
        // below attempts to schedule programs with longest compilation time first.
        // clang-format off
        SemanticType stypes[] = {
            ST_CLOSEST_HIT,
            ST_RAYGEN,
            ST_BINDLESS_CALLABLE_PROGRAM,
            ST_BOUND_CALLABLE_PROGRAM,
            ST_INTERSECTION,
            ST_EXCEPTION,
            ST_INTERNAL_AABB_EXCEPTION,
            ST_MISS,
            ST_ANY_HIT,
            ST_INTERNAL_AABB_ITERATOR,
            ST_BOUNDING_BOX,
            // TODO ST_NODE_VISIT (selector)
        };
        // clang-format on

        // Make sure no program that has one of the lwrrently unsupported semantic types is reachable.
        // TODO: We lwrrently allow NODE_VISIT here and just ignore it. There is no
        //       decision yet how to handle topologies that are not supported in RTCore.
        //const std::array<CPIDSet, NUM_SEMANTIC_TYPES>& reachableProgs = pud.m_programPlan->getReachablePrograms();
        //RT_ASSERT_MSG( reachableProgs[ST_NODE_VISIT].empty(), "RTX does not support node visit programs" );

        ProgramManager* pm = m_context->getProgramManager();

        // Prepare parallel compilation
        std::vector<GetOrCompileProgramJob::InputData> input;
        for( SemanticType stype : stypes )
        {
            const CPIDSet& cpids = pud.m_programPlan->getReachablePrograms( stype );
            for( const CanonicalProgramID cpid : cpids )
            {
                const CanonicalProgram* cp = pm->getCanonicalProgramById( cpid );

                // Supply a device count for specialization if this program accesses the launch index and fast
                // recompiles is disabled.
                const bool specializeNumDevices = !m_context->getPreferFastRecompiles() && cp->hasLaunchIndexAccesses();
                const int cpNumDevices = specializeNumDevices ? m_numLaunchDevices : RTXCompile::DONT_SPECIALIZE_NUM_DEVICES;

                const PagingMode pagingMode     = m_context->getPagingManager()->getLwrrentPagingMode();

                if( stype == ST_BOUND_CALLABLE_PROGRAM )
                {
                    // Bound callable programs are compiled once per inherited semantic
                    // type.

                    std::vector<SemanticType> callerSTs = cp->getInheritedSemanticTypes();
                    for( SemanticType callerStype : callerSTs )
                    {
                        input.push_back( {cpid, stype, callerStype, cpNumDevices, pagingMode} );
                    }
                }
                else if( stype == ST_BINDLESS_CALLABLE_PROGRAM )
                {
                    if( pud.m_programPlan->needsLightweightCompilation( cp ) )
                        input.push_back( {cpid, stype, stype, cpNumDevices, pagingMode} );
                    if( pud.m_programPlan->needsHeavyweightCompilation( cp ) )
                        input.push_back( {cpid, stype, ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE, cpNumDevices, pagingMode} );
                }
                else
                {
                    input.push_back( {cpid, stype, stype, cpNumDevices, pagingMode} );
                }
            }
        }

        // Execute parallel compilation
        std::shared_ptr<GetOrCompileProgramJob> job( std::make_shared<GetOrCompileProgramJob>(
            k_parallelCompilation.get(), input, *this, m_context, lwdaDevice, rtxParams, rtcCompileOptions,
            pud.m_attributeDecoders, pud.m_maxAttributeRegisterCount, pud.m_programPlan->getHeavyweightCallsites() ) );
        m_context->getThreadPool()->submitJobAndWait( job );

        // Post-process parallel compilation results
        std::vector<CompiledModuleHandle> compiledModules;
#if RTCORE_API_VERSION >= 25
        std::vector<unsigned int>         compiledModulesEntryIndices;
#endif
        std::vector<std::string>          compiledModulesNames;
        for( size_t i = 0; i < input.size(); ++i )
        {
            const CanonicalProgram* cp             = pm->getCanonicalProgramById( input[i].cpId );
            ModuleEntryRefPair      compilerOutput = job->getOutput( i );

            optix::SemanticType stypeForAnnotation = input[i].stype;
            optix::SemanticType inheritedStype     = input[i].inheritedStype;
            if( stypeForAnnotation == ST_BOUNDING_BOX )
                stypeForAnnotation = inheritedStype = ST_BINDLESS_CALLABLE_PROGRAM;
            else if( stypeForAnnotation == ST_INTERNAL_AABB_ITERATOR )
                stypeForAnnotation = inheritedStype = ST_RAYGEN;
            else if( stypeForAnnotation == ST_INTERNAL_AABB_EXCEPTION )
                stypeForAnnotation = inheritedStype = ST_EXCEPTION;
            task->addCompiledModule( cp, stypeForAnnotation, inheritedStype, device, compilerOutput );
            compiledModules.push_back( compilerOutput.first );
#if RTCORE_API_VERSION >= 25
            compiledModulesEntryIndices.push_back( compilerOutput.second );

            Rtlw32 entryFunctionNameSize;
            m_context->getRTCore()->compiledModuleGetEntryFunctionName( compilerOutput.first.get(), compilerOutput.second, 0, nullptr, &entryFunctionNameSize  );
            std::vector<char> entryNameBuf(entryFunctionNameSize);
            m_context->getRTCore()->compiledModuleGetEntryFunctionName( compilerOutput.first.get(), compilerOutput.second, entryFunctionNameSize, entryNameBuf.data(), nullptr );
            compiledModulesNames.push_back( entryNameBuf.data() );
#else
            compiledModulesNames.push_back( compilerOutput.second );
#endif
        }

        {
            RtcCompiledModule             compiledModule = nullptr;
            const ConstantMemAllocations& constMemAllocs = m_constantMemoryPlan->getAllocationInfo();

            m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_Global",
                                                          sizeof( cort::Global ), &compiledModule );
            RTCore* rtcore               = m_context->getRTCore();
            auto    deleteCompiledModule = [rtcore]( RtcCompiledModule cm ) { rtcore->compiledModuleDestroy( cm ); };
            compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );

            // Creating constants of size 0 triggers an error when trying to link them in.
            // For this reason avoid to generate constants if the size of a buffer is empty 0.

            if( constMemAllocs.objectRecordSize > 0 )
            {
                m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_ObjectRecord",
                                                              constMemAllocs.objectRecordSize, &compiledModule );
                compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );
            }

            if( constMemAllocs.bufferTableSize > 0 )
            {
                m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_BufferHeaderTable",
                                                              constMemAllocs.bufferTableSize, &compiledModule );
                compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );
            }

            if( constMemAllocs.textureTableSize > 0 )
            {
                m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_TextureHeaderTable",
                                                              constMemAllocs.textureTableSize, &compiledModule );
                compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );
            }

            if( constMemAllocs.programTableSize > 0 )
            {
                m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_ProgramHeaderTable",
                                                              constMemAllocs.programTableSize, &compiledModule );
                compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );
            }

            if( !k_limitActiveLaunchIndices.isDefault() && m_entry != m_context->getAabbEntry() )
            {
                m_context->getRTCore()->compileNamedConstant( lwdaDevice->rtcContext(), "const_MinMaxLaunchIndex",
                                                              sizeof( unsigned int ) * 6, &compiledModule );
                compiledModules.push_back( CompiledModuleHandle( compiledModule, deleteCompiledModule ) );
            }
        }

        const Rtlw32 maxTraceDepth =
            k_maxTraceRelwrsionDepth.isDefault() ? m_context->getMaxTraceDepth() : k_maxTraceRelwrsionDepth.get();

        // Choose pipeline parameters
        RtcPipelineOptions pipelineOptions            = {};
        pipelineOptions.maxTraceRelwrsionDepth        = maxTraceDepth;
        pipelineOptions.defaultCallableRelwrsionDepth = 2;

        pipelineOptions.type = RTC_PIPELINE_TYPE_ILWALID;
        if( k_pipelineType.get() == "megakernel_simple" )
            pipelineOptions.type = RTC_PIPELINE_TYPE_MEGAKERNEL_SIMPLE;
        else
            RT_ASSERT_FAIL_MSG( "Unknown pipeline type: " + k_pipelineType.get() );

        if( k_traversalOverride.isDefault() )
        {
            if( m_context->RtxUniversalTraversalEnabled() )
            {
#if LWCFG( GLOBAL_ARCH_AMPERE )
                if( m_context->getDeviceManager()->activeDevicesSupportMotionTTU() )
                {
                    pipelineOptions.abiVariant    = RTC_ABI_VARIANT_MTTU;
                    pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_MTTU;
                }
                else
#endif
                {
                    pipelineOptions.abiVariant    = RTC_ABI_VARIANT_UTRAV;
                    pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_UNIVERSAL;
                }
            }
            else
            {
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
                if( m_context->getDeviceManager()->activeDevicesSupportTTU() )
                {
                    pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_TTU_2LH;
                    pipelineOptions.abiVariant    = RTC_ABI_VARIANT_TTU_A;
                }
                else
#endif
                {
                    pipelineOptions.abiVariant    = RTC_ABI_VARIANT_DEFAULT;
                    pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_BVH2_2LH;
                }
            }
        }
        else if( k_traversalOverride.get() == "Utrav" )
        {
            pipelineOptions.abiVariant    = RTC_ABI_VARIANT_UTRAV;
            pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_UNIVERSAL;
        }
        else if( k_traversalOverride.get() == "Bvh2" )
        {
            pipelineOptions.abiVariant    = RTC_ABI_VARIANT_DEFAULT;
            pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_BVH2_2LH;
        }
        else if( k_traversalOverride.get() == "Bvh8" )
        {
            pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_BVH8_2LH;
            pipelineOptions.abiVariant    = RTC_ABI_VARIANT_BVH8;
        }
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
        // Enable the TTU when explicitly asked to or if not overriden and the DeviceManager indicates we can
        else if( k_traversalOverride.get() == "TTU" )
        {
            pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_TTU_2LH;
            pipelineOptions.abiVariant    = RTC_ABI_VARIANT_TTU_A;
        }
#endif
#if LWCFG( GLOBAL_ARCH_AMPERE )
        else if( k_traversalOverride.get() == "MTTU" )
        {
            pipelineOptions.abiVariant    = RTC_ABI_VARIANT_MTTU;
            pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_MTTU;
        }
#endif
        else
        {
            throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported traversal override : " + k_traversalOverride.get() );
        }

        // Deduplicate modules passed to RTCore pipeliine creation
        const int                   numInputModules = (int)compiledModules.size();
        std::set<RtcCompiledModule> moduleSet;
        for( int i = 0; i < numInputModules; ++i )
            moduleSet.insert( compiledModules[i].get() );

        std::vector<RtcCompiledModule> deduplicatedModules( moduleSet.begin(), moduleSet.end() );

        lwdaDevice->makeLwrrent();  // TODO(jbigler) why do we need this??

        // Create the pipeline
        RtcPipeline pipeline = nullptr;
        {
            TIMEVIZ_SCOPE( "rtcPipelineCreate" );
            m_context->getRTCore()->pipelineCreate( lwdaDevice->rtcContext(), &pipelineOptions, &rtcCompileOptions,
                                                    deduplicatedModules.data(), (int)deduplicatedModules.size(), &pipeline );
        }

        Rtlw32 directCallableStackSizeFromTraversal = 0;
        Rtlw32 directCallableStackSizeFromState     = 0;
        Rtlw32 continuationStackSize                = k_continuationStackSize.get();

        // Get Attribute stack size values,
        // continuation stack size, direct callable stack size from traversal, direct callable stack size from state.
        // If all three values are 0 the attribute is not taken into account and a stack size estimate is computed, see below.

        if( !m_useContextAttributesForStackSize )
        {
            // Compute an estimate of the maximum required stack size.

            // Compute callable stack sizes
            // Collect callable stack sizes for callables that do not
            // call callables themselves separately because they are independent
            // of the callable depth.
            Rtlw32 callableDirectStackMax                 = 0;  // bindless callables
            Rtlw32 callableLeafDirectStackMax             = 0;  // bindless callables w/o callable calls
            Rtlw32 callableContinuationStackMax           = 0;  // bound & heavy bindless
            Rtlw32 callableLeafContinuationStackMax       = 0;  // bound & heavy bindless w/o callable calls
            Rtlw32 boundCallableDirectStackFromTrvMax     = 0;  // bound from trv
            Rtlw32 boundCallableLeafDirectStackFromTrvMax = 0;  // bound from trv w/o callable calls
            for( size_t i = 0; i < input.size(); ++i )
            {
                SemanticType stype = input[i].stype;
                if( stype != ST_BOUND_CALLABLE_PROGRAM && stype != ST_BINDLESS_CALLABLE_PROGRAM )
                    continue;
                SemanticType inheritedStype = input[i].inheritedStype;
                Rtlw32       directStackFrameSize;
                Rtlw32       continuationStackFrameSize;

#if RTCORE_API_VERSION >= 25
                m_context->getRTCore()->compiledModuleGetStackSize( compiledModules[i].get(), compiledModulesEntryIndices[i],
                                                                    &directStackFrameSize, &continuationStackFrameSize );
#else
                m_context->getRTCore()->compiledModuleGetStackSize( compiledModules[i].get(), compiledModulesNames[i].c_str(),
                                                                    &directStackFrameSize, &continuationStackFrameSize );
#endif
                logModuleStackSizes( m_context->getUsageReport(), compiledModulesNames[i], directStackFrameSize,
                                     continuationStackFrameSize );

                const CanonicalProgram* cp = pm->getCanonicalProgramById( input[i].cpId );
                if( inheritedStype == ST_ANY_HIT || inheritedStype == ST_INTERSECTION )
                {
                    // Collect direct stack size for bound callables from traversal separately
                    // because they are called differently when called from state.
                    RT_ASSERT( stype == ST_BOUND_CALLABLE_PROGRAM && continuationStackFrameSize == 0 );
                    if( cp->callsBindlessCallableProgram() || cp->callsBoundCallableProgram() )
                        boundCallableDirectStackFromTrvMax = std::max( boundCallableDirectStackFromTrvMax, directStackFrameSize );
                    else
                        boundCallableLeafDirectStackFromTrvMax =
                            std::max( boundCallableDirectStackFromTrvMax, directStackFrameSize );
                }
                else
                {
                    if( cp->callsBindlessCallableProgram() || cp->callsBoundCallableProgram() )
                    {
                        callableDirectStackMax       = std::max( callableDirectStackMax, directStackFrameSize );
                        callableContinuationStackMax = std::max( callableContinuationStackMax, continuationStackFrameSize );
                    }
                    else
                    {
                        callableLeafDirectStackMax = std::max( callableLeafDirectStackMax, directStackFrameSize );
                        callableLeafContinuationStackMax = std::max( callableLeafContinuationStackMax, continuationStackFrameSize );
                    }
                }
            }

            const Rtlw32 maxCallableDepth = k_maxBindlessCallableDepth.isDefault() ? m_context->getMaxCallableProgramDepth() :
                                                                                     k_maxBindlessCallableDepth.get();

            // Print a message if maxTraceDepth is used for computing the continuation stack size and wasn't set by the API or knob.
            if( !m_context->isSetByAPIMaxTraceDepth() && k_maxTraceRelwrsionDepth.isDefault() )
            {
                ureport2( m_context->getUsageReport(), "INFO" )
                    << "Used default value " << m_context->getMaxTraceDepthDefault()
                    << " as maximum trace depth for computing stack size. "
                    << "rtContextSetMaxTraceDepth can be used for setting a suitable value." << std::endl;
            }
            // Print a message if maxCallableProgramDepth is used for computing direct stack size and wasn't set by the API or knob.
            if( !m_context->isSetByAPIMaxCallableProgramDepth() && k_maxBindlessCallableDepth.isDefault() )
            {
                ureport2( m_context->getUsageReport(), "INFO" )
                    << "Used default value " << m_context->getMaxCallableProgramDepthDefault()
                    << " as maximum callable program depth for computing stack size. "
                    << "rtContextSetMaxCallableProgramDepth can be used for setting a suitable value." << std::endl;
            }

            Rtlw32 directCallableStackSize       = 0;
            Rtlw32 continuationCallableStackSize = 0;
            Rtlw32 boundCallableFromTrvStackSize = 0;
            if( maxCallableDepth > 0 )
            {
                // callwlate the last level of the call graph
                directCallableStackSize       = std::max( callableDirectStackMax, callableLeafDirectStackMax );
                continuationCallableStackSize = std::max( callableContinuationStackMax, callableLeafContinuationStackMax );
                boundCallableFromTrvStackSize =
                    std::max( boundCallableDirectStackFromTrvMax, boundCallableLeafDirectStackFromTrvMax );

                // account for the previous levels
                directCallableStackSize += callableDirectStackMax * ( maxCallableDepth - 1 );
                continuationCallableStackSize += callableContinuationStackMax * ( maxCallableDepth - 1 );
                boundCallableFromTrvStackSize += boundCallableDirectStackFromTrvMax * ( maxCallableDepth - 1 );
            }

            Rtlw32 rgExContinuationStackMax = 0;
            Rtlw32 isAhContinuationStackMax = 0;
            Rtlw32 chMsContinuationStackMax = 0;

            for( size_t i = 0; i < input.size(); ++i )
            {
                optix::SemanticType stype = input[i].stype;

                if( stype == ST_BOUND_CALLABLE_PROGRAM || stype == ST_BINDLESS_CALLABLE_PROGRAM )
                    continue;

                Rtlw32 directStackFrameSize;
                Rtlw32 continuationStackFrameSize;

#if RTCORE_API_VERSION >= 25
                m_context->getRTCore()->compiledModuleGetStackSize( compiledModules[i].get(), compiledModulesEntryIndices[i],
                                                                    &directStackFrameSize, &continuationStackFrameSize );
#else
                m_context->getRTCore()->compiledModuleGetStackSize( compiledModules[i].get(), compiledModulesNames[i].c_str(),
                                                                    &directStackFrameSize, &continuationStackFrameSize );
#endif
                logModuleStackSizes( m_context->getUsageReport(), compiledModulesNames[i], directStackFrameSize,
                                     continuationStackFrameSize );

                const CanonicalProgram* cp = pm->getCanonicalProgramById( input[i].cpId );

                Rtlw32 callableDirectStackSize = 0;
                if( cp->callsBindlessCallableProgram() )  // bound callable calls are handled below depending on stype
                    callableDirectStackSize += directCallableStackSize;


                switch( stype )
                {
                    case ST_CLOSEST_HIT:
                    case ST_MISS:
                    {
                        directCallableStackSizeFromState = std::max( directCallableStackSizeFromState, callableDirectStackSize );

                        if( cp->callsBoundCallableProgram() || pud.m_programPlan->getHeavyBindlessCallers().count( cp->getID() ) )
                            continuationStackFrameSize += continuationCallableStackSize;

                        chMsContinuationStackMax = std::max( chMsContinuationStackMax, continuationStackFrameSize );
                        break;
                    }
                    case ST_RAYGEN:
                        directCallableStackSizeFromState = std::max( directCallableStackSizeFromState, callableDirectStackSize );

                        if( cp->callsBoundCallableProgram() || pud.m_programPlan->getHeavyBindlessCallers().count( cp->getID() ) )
                            continuationStackFrameSize += continuationCallableStackSize;

                        rgExContinuationStackMax = std::max( rgExContinuationStackMax, continuationStackFrameSize );
                        break;
                    case ST_INTERSECTION:
                    case ST_ANY_HIT:
                        if( cp->callsBoundCallableProgram() )
                            callableDirectStackSize += boundCallableFromTrvStackSize;
                        directCallableStackSizeFromTraversal =
                            std::max( directCallableStackSizeFromTraversal, callableDirectStackSize );

                        //RT_ASSERT( continuationStackFrameSize == 0 );
                        isAhContinuationStackMax = std::max( isAhContinuationStackMax, continuationStackFrameSize );
                        break;
                    case ST_EXCEPTION:
                        //RT_ASSERT( continuationStackFrameSize == 0 );
                        rgExContinuationStackMax = std::max( rgExContinuationStackMax, continuationStackFrameSize );
                        break;
                    default:
                        break;
                }
            }

            // If not specified using knob, compute the continuation stack size.
            if( k_continuationStackSize.isDefault() )
            {
                continuationStackSize = rgExContinuationStackMax;
                if( maxTraceDepth > 0 )
                {
                    continuationStackSize += ( maxTraceDepth - 1 ) * chMsContinuationStackMax
                                             + std::max( chMsContinuationStackMax, ( isAhContinuationStackMax * 2 ) );
                }

                // Print a warning if the computed continuation stack size is >= 8K.
                if( continuationStackSize >= 8192 )
                {
                    ureport2( m_context->getUsageReport(), "INFO" )
                        << "WARNING: Large stack size " << continuationStackSize
                        << " computed from maximum trace depth and maximum callable program depth" << std::endl;
                }
            }
        }
        else
        {
            // Stack size was set by context attribute.
            if( k_continuationStackSize.isDefault() )
                continuationStackSize            = m_continuationStackSize;
            directCallableStackSizeFromTraversal = m_directCallableStackSizeFromTraversal;
            directCallableStackSizeFromState     = m_directCallableStackSizeFromState;
        }

        llog( 30 ) << "RTXPLAN: rtcore pipeline stack travCallableDirect=" << directCallableStackSizeFromTraversal
                   << ",  stateCallableDirect=" << directCallableStackSizeFromState
                   << ",  continuation=" << continuationStackSize << ".\n";

        ureport2( m_context->getUsageReport(), "INFO" )
            << "Traversal pipeline stack (bytes): traversalCallableDirect=" << directCallableStackSizeFromTraversal
            << ", stateCallableDirect=" << directCallableStackSizeFromState
            << ", continuation=" << continuationStackSize << std::endl;

        task->setDeviceInfo( lwdaDevice, pipeline, m_constantMemoryPlan->getAllocationInfo(),
                             m_entry == m_context->getAabbEntry(), directCallableStackSizeFromTraversal,
                             directCallableStackSizeFromState, continuationStackSize );
    }

    std::unique_ptr<FrameTask> frametask( task.release() );  // Implicit cast to FrameTask
    setTask( std::move( frametask ) );
}

// -----------------------------------------------------------------------------
// TODO:
// This function has similar purpose to computeAttributeOffsets in Compile.cpp
// We might think of unifying the two.
void RTXPlan::computeAttributeDataSizes( const std::set<CanonicalProgramID>& isectPrograms,
                                         int                                 maxAttributeRegisterCount,
                                         int&                                registerCount,
                                         int&                                memoryCount ) const
{
    int total_regcount = 0;
    for( const CanonicalProgramID& cpid : isectPrograms )
    {
        int cpAttributeCount = m_context->getProgramManager()->getCanonicalProgramById( cpid )->getMaxAttributeData32bitValues();
        total_regcount = std::max( total_regcount, cpAttributeCount );
    }
    registerCount = std::min( maxAttributeRegisterCount, total_regcount );
    memoryCount   = total_regcount - registerCount;
}

unsigned int RTXPlan::getRtxExceptionFlags( Context* context )
{
    uint64_t rtxExceptionFlags = context->getExceptionFlags();

    if( k_enableAllExceptions.get() )
        Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_ALL, true );
    else if( k_disableAllExceptions.get() )
        Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_ALL, false );
    else
    {
        if( k_enableStackOverflowHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_STACK_OVERFLOW, true );
        if( k_enableTraceDepthHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_TRACE_DEPTH_EXCEEDED, true );
        if( k_enableIlwalidBufferIdHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_BUFFER_ID_ILWALID, true );
        if( k_enableIlwalidTextureIdHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_TEXTURE_ID_ILWALID, true );
        if( k_enableIlwalidProgramIdHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_PROGRAM_ID_ILWALID, true );
        if( k_enableBufferIndexOutOfBoundsHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, true );
        if( k_enableIndexOutOfBoundsHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_INDEX_OUT_OF_BOUNDS, true );
        if( k_enableIlwalidRayHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_ILWALID_RAY, true );
        if( k_enablePayloadAccessOutOfBoundsHandling.get() )
            Context::setExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS, true );
    }

    return rtxExceptionFlags;
}

unsigned int RTXPlan::getRtcoreExceptionFlags( unsigned int rtxExceptionFlags )
{
    unsigned int rtcoreExceptionFlags = RTC_EXCEPTION_FLAG_ILWOKE_EX;
    if( Context::getExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_STACK_OVERFLOW ) )
        rtcoreExceptionFlags |= RTC_EXCEPTION_FLAG_STACK_OVERFLOW;
    if( Context::getExceptionEnabled( rtxExceptionFlags, RT_EXCEPTION_TRACE_DEPTH_EXCEEDED ) )
        rtcoreExceptionFlags |= RTC_EXCEPTION_FLAG_TRACE_DEPTH;
    if( Context::hasProductSpecificExceptionsEnabled( rtxExceptionFlags ) )
        rtcoreExceptionFlags |= RTC_EXCEPTION_FLAG_PRODUCT_SPECIFIC;

    return rtcoreExceptionFlags;
}

bool RTXPlan::isDirectCalledBoundCallable( SemanticType stype, SemanticType inheritedStype )
{
    if( stype == ST_BOUND_CALLABLE_PROGRAM )
        return inheritedStype == ST_INTERSECTION || inheritedStype == ST_ANY_HIT || inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM;
    return false;
}

void RTXPlan::determineRtxCompileOptions( const LWDADevice*          device,
                                          const PerUniqueDevice&     pud,
                                          RtcCompileOptions*         rtcoreOptions,
                                          RTXCompile::CompileParams* rtxParams ) const
{
    // Payload and callable program parameters are automatically moved
    // to local memory if exceeding the max amount supported by
    // rtcore. Attributes have a hard limit (for now). For all of these
    // counts, only use as many registers as necessary if the respective
    // threshold is not reached.

    Rtlw64 maxAttributeRegisters = 0;
    m_context->getRTCore()->deviceContextGetLimit( device->rtcContext(), RTC_LIMIT_MAX_ATTRIBUTE_REGISTERS, &maxAttributeRegisters );

    Rtlw64 maxPayloadRegisters = 0;
    m_context->getRTCore()->deviceContextGetLimit( device->rtcContext(), RTC_LIMIT_MAX_PAYLOAD_REGISTERS, &maxPayloadRegisters );

    Rtlw64 maxParamRegisters = 0;
    m_context->getRTCore()->deviceContextGetLimit( device->rtcContext(), RTC_LIMIT_MAX_CALLABLE_PARAM_REGISTERS, &maxParamRegisters );

    // Limit the number of registers for now. For example, some applications want to use a huge
    // amount of attributes and the rtcore limits are really too large.

    if( maxPayloadRegisters > 16 )
        maxPayloadRegisters = 16;

    if( maxAttributeRegisters > 4 )
        maxAttributeRegisters = 4;

    if( maxParamRegisters > 4 )
        maxParamRegisters = 4;

    RT_ASSERT( pud.m_maxAttributeRegisterCount <= (int)maxAttributeRegisters );

    if( k_maxPayloadRegCount.get() >= 0 && maxPayloadRegisters > (Rtlw64)k_maxPayloadRegCount.get() )
        maxPayloadRegisters = k_maxPayloadRegCount.get();
    if( k_maxParamRegCount.get() >= 0 && maxParamRegisters > (Rtlw64)k_maxParamRegCount.get() )
        maxParamRegisters = k_maxParamRegCount.get();

    // Determine compilation options
    const int exceptionFlags = getRtxExceptionFlags( m_context );

    int traversableGraphFlags = 0;  // default: strict two-level hierarchy without motion
    if( m_context->RtxUniversalTraversalEnabled() )
    {
        traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_BOTTOM_LEVEL_INSTANCE;
        traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_BLAS;

        // utrav supports motion blur AS
        if( m_hasMotionBlur )
            traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_MOTION_ACCEL;

        // utrav supports all possible transform traversables
        traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_STATIC_TRANSFORM_TRAVERSABLE;

        if( m_hasMotionBlur )
            traversableGraphFlags |= ( RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_MATRIX_MOTION_TRANSFORM_TRAVERSABLE
                                       | RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_SRT_MOTION_TRANSFORM_TRAVERSABLE );
    }

    const bool propagatePayloadSize = Context::getExceptionEnabled( exceptionFlags, RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS );
    const int  numPayloadSizeRegisters = propagatePayloadSize ? 1 : 0;
    const bool payloadInRegisters =
        pud.m_canPromotePayload && ( pud.m_maxPayloadRegisterCount + numPayloadSizeRegisters ) <= (int)maxPayloadRegisters;
    const int numPayloadRegisters   = payloadInRegisters ? pud.m_maxPayloadRegisterCount : 2;
    const int numAttributeRegisters = pud.m_maxAttributeRegisterCount;
    const int numParameterRegisters = std::min( (int)maxParamRegisters, pud.m_maxCallableProgramParamRegisterCount );  // ABI TODO: query for this?
    const int numMemoryAttributes = pud.m_maxAttributeMemoryCount;

    // RTX params are used in the optix-> lwvm-rt compilation process
    rtxParams->payloadInRegisters = payloadInRegisters;
    rtxParams->constMemAllocFlags = RTXCompile::ConstantMemAllocationFlags( m_constantMemoryPlan->getAllocationInfo() );
    rtxParams->numCallableParamRegisters = numParameterRegisters;
    rtxParams->forceInlineUserFunctions  = m_context->getForceInlineUserFunctions();
    rtxParams->addLimitIndicesCheck      = !k_limitActiveLaunchIndices.isDefault();
    rtxParams->exceptionFlags            = exceptionFlags;
    rtxParams->maxPayloadSize            = 4 * pud.m_maxPayloadRegisterCount;
    rtxParams->propagatePayloadSize      = propagatePayloadSize;
    rtxParams->maxAttributeRegisterCount = pud.m_maxAttributeRegisterCount;

    // rtcore options are used in the lwvm-rt -> sass compilation
    // process. WARNING: if ANY parameters are removed or added from
    // this list, you must:
    // 1. Update the operator< in CompiledProgramCache.cpp
    // 2. Update the operator== in CompiledProgramCache.cpp
    // 3. Update the readOrWrite function in CompiledProgramCache.cpp
    // 4. Bump the version number in the readOrWrite function in
    //    CompiledProgramCache.cpp

    *rtcoreOptions                           = {};
    rtcoreOptions->numPayloadRegisters       = numPayloadRegisters + numPayloadSizeRegisters;
    rtcoreOptions->numAttributeRegisters     = numAttributeRegisters;
    rtcoreOptions->numCallableParamRegisters = numParameterRegisters;
    rtcoreOptions->numMemoryAttributeScalars = numMemoryAttributes;
    rtcoreOptions->traversableGraphFlags     = traversableGraphFlags;
    rtcoreOptions->smVersion                 = device->computeCapability().version();
    rtcoreOptions->maxRegisterCount          = k_maxRegCount.get();
    rtcoreOptions->optLevel                  = k_sassOptimization.get();
    rtcoreOptions->debugLevel                = 0;
    rtcoreOptions->exceptionFlags            = getRtcoreExceptionFlags( rtxParams->exceptionFlags );
    // Spills to shared memory are lwrrently only exposed through D2IR.
    // As soon as this feature is available through ptxJit we will enable it for Optix as well.
    rtcoreOptions->smemSpillPolicy              = RTC_SMEM_SPILLING_DISABLED;
    rtcoreOptions->targetSharedMemoryBytesPerSM = -1;

    // Priority
    // 1. knob if set
    // 2. elw is set
    // 3. knob value
    rtcoreOptions->useLWPTX = !k_enableD2IR.get();
    if( !k_enableD2IR.isSet() )
    {
        std::string elwVarValue;
        if( corelib::getelw( OPTIX_FORCE_DEPRECATED_COMPILER_STR, elwVarValue ) )
        {
            unsigned int value = -1;
            sscanf( elwVarValue.c_str(), "%u", &value );

            switch( value )
            {
                case optix_exp::OptixForceDeprecatedCompilerValues::LWVM7_D2IR:
                    rtcoreOptions->useLWPTX = false;
                    break;
                case optix_exp::OptixForceDeprecatedCompilerValues::LWVM7_LWPTX:
                case optix_exp::OptixForceDeprecatedCompilerValues::LWVM34_LWPTX:
                    rtcoreOptions->useLWPTX = true;
                    break;
                default:
                    throw IlwalidValue( RT_EXCEPTION_INFO, std::string( "Unknown value for " )
                                                               + OPTIX_FORCE_DEPRECATED_COMPILER_STR + ": " + elwVarValue );
            }
        }
    }

    rtcoreOptions->enableCoroutines = k_enableCoroutines.get();

    rtcoreOptions->usesTraversables = true;

    bool enableProfiling = k_enableRTcoreProfiling.get();

    if( enableProfiling )
    {
        rtcoreOptions->enabledTools = RTC_TOOLS_FLAG_PROFILING | RTC_TOOLS_FLAG_DETAILED_SHADER_INFO;
    }
    else
    {
        rtcoreOptions->enabledTools = RTC_TOOLS_FLAG_NONE;
    }

    if( k_traversalOverride.isDefault() )
    {
        if( m_context->RtxUniversalTraversalEnabled() )
        {
#if LWCFG( GLOBAL_ARCH_AMPERE )
            if( m_context->getDeviceManager()->activeDevicesSupportMotionTTU() )
            {
                rtcoreOptions->abiVariant = RTC_ABI_VARIANT_MTTU;
            }
            else
#endif
            {
                rtcoreOptions->abiVariant = RTC_ABI_VARIANT_UTRAV;
            }
        }
        else
        {
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
            if( m_context->getDeviceManager()->activeDevicesSupportTTU() )
            {
                rtcoreOptions->abiVariant = RTC_ABI_VARIANT_TTU_A;
            }
            else
#endif
            {
                rtcoreOptions->abiVariant = RTC_ABI_VARIANT_DEFAULT;
            }
        }
    }
    else if( k_traversalOverride.get() == "Utrav" )
    {
        rtcoreOptions->abiVariant = RTC_ABI_VARIANT_UTRAV;
    }
    else if( k_traversalOverride.get() == "Bvh2" )
    {
        rtcoreOptions->abiVariant = RTC_ABI_VARIANT_DEFAULT;
    }
    else if( k_traversalOverride.get() == "Bvh8" )
    {
        rtcoreOptions->abiVariant = RTC_ABI_VARIANT_BVH8;
    }
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
    else if( k_traversalOverride.get() == "TTU" )
    {
        rtcoreOptions->abiVariant = RTC_ABI_VARIANT_TTU_A;
    }
#endif
#if LWCFG( GLOBAL_ARCH_AMPERE )
    else if( k_traversalOverride.get() == "MTTU" )
    {
        rtcoreOptions->abiVariant = RTC_ABI_VARIANT_MTTU;
    }
#endif
    else
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported traversal override : " + k_traversalOverride.get() );
    }
    rtcoreOptions->compileForLwda = true;
}

static CompiledProgramCacheKey::SpecializationsMap colwertToCacheKeyFormat( const Specializations& specializations,
                                                                            const ProgramManager*  pm )
{
    // Extract cache key data from specializations and attributeAssignments
    // removing compile order dependent data.
    CompiledProgramCacheKey::SpecializationsMap keySpecializations;
    for( const auto& elt : specializations.m_varspec )
    {
        auto varref = pm->getVariableReferenceById( elt.first );
        keySpecializations.emplace( varref->getUniversallyUniqueName(), elt.second );
    }
    return keySpecializations;
}

static CompiledProgramCacheKey::AttributeDecoderList colwertToCacheKeyFormat( const RTXPlan::AttributeDecoderList& attributeDecoder )
{
    CompiledProgramCacheKey::AttributeDecoderList decoders;
    for( const auto& decoder : attributeDecoder )
        decoders.push_back( decoder->getUniversallyUniqueName() );
    return decoders;
}

ModuleEntryRefPair optix::RTXPlan::getOrCompileProgram( const LWDADevice*                          device,
                                                        const RTXCompile::CompileParams&           rtxParams,
                                                        const RtcCompileOptions&                   options,
                                                        const AttributeDecoderList&                allDecoders,
                                                        const std::set<const CallSiteIdentifier*>& heavyweightCallSites,
                                                        int                     numConlwrrentLaunchDevices,
                                                        PagingMode              pagingMode,
                                                        SemanticType            stype,
                                                        SemanticType            inheritedStype,
                                                        const CanonicalProgram* cp,
                                                        std::mutex&             mutex ) const
{
    TIMEVIZ_FUNC;

    // Determine which (if any) attribute decoders we need
    bool                        needDecoders = stype == ST_ANY_HIT || stype == ST_CLOSEST_HIT;
    AttributeDecoderList        emptyList;
    const AttributeDecoderList& attributeDecoders = needDecoders ? allDecoders : emptyList;

    // Determine which specializations apply to this canonical program
    // and any attribute decoders.
    Specializations specializations = m_specializationPlan->narrowFor( cp );
    for( const CanonicalProgram* decoder : attributeDecoders )
    {
        Specializations attributeSpecializations = m_specializationPlan->narrowFor( decoder );
        specializations.mergeVariableSpecializations( attributeSpecializations );
    }

    // Filter heavyweight call sites for current canonical program.
    std::set<std::string> heavyweightCallSiteNames;
    if( inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM )
    {
        for( const CallSiteIdentifier* csId : heavyweightCallSites )
        {
            if( csId->getParent() == cp )
                heavyweightCallSiteNames.emplace( csId->getUniversallyUniqueName() );
        }
    }

    //
    // Look first in the in-memory cache.
    //
    RtcCompileOptions optionsClone = options;
    if( stype != ST_ANY_HIT && stype != ST_CLOSEST_HIT && stype != ST_INTERSECTION )
    {
        // If the program is a non-hitgroup program clear numMemoryAttributeScalars to
        // avoid unnecessary cache misses.  (http://lwbugs/2713560)
        optionsClone.numMemoryAttributeScalars = 0;
    }
    CompiledProgramCacheKey cacheKey( device, rtxParams, optionsClone, colwertToCacheKeyFormat( attributeDecoders ),
                                      colwertToCacheKeyFormat( specializations, m_context->getProgramManager() ),
                                      heavyweightCallSiteNames, stype, inheritedStype, cp, m_dimensionality,
                                      numConlwrrentLaunchDevices, pagingMode );
    ModuleEntryRefPair cacheEntry;
    if( m_compiledProgramCache->find( cacheKey, cacheEntry ) )
    {
        llog( 13 ) << "RTXPLAN: Function: " << cp->getInputFunctionName() << " with ST: " << semanticTypeToString( stype )
                   << " found in compiled module cache! Skipping compile.\n";
        return cacheEntry;
    }

    //
    // Next, look in the on-disk cache.
    //
    std::ostringstream diskCacheKey;
    if( m_context->getDiskCache()->isActive() && k_cacheEnabled.get() )
    {
        std::unique_ptr<PersistentStream> hasher = m_context->getDiskCache()->createHasher();
        readOrWrite( hasher.get(), &cacheKey, "cacheKey" );
        std::string digest = hasher->getDigestString();

        // Build the cache key name
        int smVersion = device->computeCapability().version();
        diskCacheKey << "rtx-" << cp->getUniversallyUniqueName() << "-key" << digest << "-sm_" << smVersion << "-drv"
                     << LWML::driverVersion();
        llog( 13 ) << "DiskCache: RTX module cache key string: " << diskCacheKey.str() << '\n';
        lif_active( 13 )
        {
            PrintStream ps( llog_stream( 13 ) );
            readOrWrite( &ps, &cacheKey, "cacheKey" );
        }

        // Attempt to load from cache
        std::unique_ptr<optix::PersistentStream> cached;
        optix_exp::ErrorDetails                  errDetails;
        if( m_context->getDiskCache()->find( diskCacheKey.str(), cached, m_context->getDeviceContextLogger(), errDetails ) )
        {
            llog( 2 ) << "DiskCache: " << errDetails.m_description;
        }
        else
        {
            if( cached )
            {
                ModuleEntryRefPair cachedModule;
                if( loadModuleFromDiskCache( cached.get(), cachedModule, cacheKey ) )
                {
                    llog( 13 ) << "DiskCache: RTX module hit\n";
                    ureport2( m_context->getUsageReport(), "INFO" ) << "Module cache HIT  : " << diskCacheKey.str() << std::endl;
                    return cachedModule;
                }
                else
                {
                    llog( 13 ) << "DiskCache: RTX module miss due to failed read\n";
                }
            }
            else
            {
                llog( 13 ) << "DiskCache: RTX module miss\n";
            }
            ureport2( m_context->getUsageReport(), "INFO" ) << "Module cache miss : " << diskCacheKey.str() << std::endl;
        }
    }

    //
    // No luck, compile
    //
    llog( 13 ) << "RTXPLAN: Function: " << cp->getInputFunctionName() << " with ST: " << semanticTypeToString( stype )
               << " NOT found in compiled module cache, compiling...\n";

    corelib::timerTick t0 = corelib::getTimerTick();

    RTXCompile::Options rtxOptions{device,
                                   rtxParams,
                                   optionsClone,
                                   attributeDecoders,
                                   heavyweightCallSiteNames,
                                   specializations,
                                   stype,
                                   inheritedStype,
                                   numConlwrrentLaunchDevices,
                                   pagingMode,
                                   cp};
    ModuleEntryRefPair compilerOutput = compileProgramToRTX( rtxOptions, mutex );

    // Put the module in the memory cache
    m_compiledProgramCache->emplace( cacheKey, compilerOutput );

    // And in the disk cache
    if( m_context->getDiskCache()->isActive() && k_cacheEnabled.get() )
    {
        std::unique_ptr<PersistentStream> save;
        optix_exp::ErrorDetails           errDetails;
        if( m_context->getDiskCache()->insert( diskCacheKey.str(), save, errDetails ) )
        {
            llog( 2 ) << "DiskCache: " << errDetails.m_description;
        }
        else if( save )
        {
            saveModuleToDiskCache( save.get(), &compilerOutput, cacheKey );
            save->flush( m_context->getDeviceContextLogger() );
        }
    }

    ureport2( m_context->getUsageReport(), "INFO" ) << "Module compile time (ms) : " << cp->getInputFunctionName()
                                                    << " : " << corelib::getDeltaMilliseconds( t0 ) << std::endl;

    return compilerOutput;
}

static const unsigned int* version = getOptixUUID();

bool RTXPlan::loadModuleFromDiskCache( PersistentStream* stream, ModuleEntryRefPair& cachedModule, const CompiledProgramCacheKey& cacheKey ) const
{
    // Read function index and blob
    stream->readOrWriteObjectVersion( version );
#if RTCORE_API_VERSION >= 25
    Rtlw32 entryIndex;
    readOrWrite( stream, &entryIndex, "entryIndex" );
#else
    readOrWrite( stream, &cachedModule.second, "functionName" );
#endif
    std::vector<char> blob;
    readOrWrite( stream, &blob, "blob" );

    // Read options that were used to compile this module (for
    // forensic use). This could be skipped if it impacts perf.
    CompiledProgramCacheKey originalKey;
    readOrWrite( stream, &originalKey, "originalKey" );

    // Check version again and fail if any errors oclwrred.
    stream->readOrWriteObjectVersion( version );
    if( stream->error() )
    {
        llog( 13 ) << "DiskCache: RTX module cache miss due to failed deserialization";
        return false;
    }

    // Reconstitute compiled module from the blob
    RtcCompiledModule rtcModule;
    RtcResult         res;
    m_context->getRTCore()->compiledModuleFromCachedBlob( cacheKey.device->rtcContext(), &blob[0], blob.size(),
                                                          &rtcModule, &res );
    if( res != RTC_SUCCESS )
    {
        llog( 13 ) << "DiskCache: RTX module cache miss due to failed deserialization (rtcore)";
        return false;
    }

#if RTCORE_API_VERSION >= 25
    cachedModule = deduplicateRtcModule( cacheKey.device, rtcModule, entryIndex );
#else
    RTCore* rtcore = m_context->getRTCore();
    auto deleteCompiledModule = [rtcore]( RtcCompiledModule cm ) { rtcore->compiledModuleDestroy( cm ); };
    cachedModule.first = CompiledModuleHandle( rtcModule, deleteCompiledModule );
#endif

    // Save the module in the memory-based cache
    m_compiledProgramCache->emplace( cacheKey, cachedModule );

    return true;
}

void RTXPlan::saveModuleToDiskCache( PersistentStream* stream, ModuleEntryRefPair* compiledModule, const CompiledProgramCacheKey& cacheKey ) const
{
    // Get module from rtcore
    Rtlw64 blobSize = 0;
    m_context->getRTCore()->compiledModuleGetCachedBlob( compiledModule->first.get(), 0, nullptr, &blobSize );

    std::vector<char> blob( blobSize );
    Rtlw64            checkSize = 0;
    m_context->getRTCore()->compiledModuleGetCachedBlob( compiledModule->first.get(), blobSize, &blob[0], &checkSize );
    if( blobSize != checkSize )
        throw prodlib::CompileError( RT_EXCEPTION_INFO, "Cached blob size mismatch" );

    // Write function name and blob
    stream->readOrWriteObjectVersion( version );
#if RTCORE_API_VERSION >= 25
    readOrWrite( stream, &compiledModule->second, "entryIndex" );
#else
    readOrWrite( stream, &compiledModule->second, "functionName" );
#endif
    readOrWrite( stream, &blob, "blob" );

    // Write options used to compile this module.
    readOrWrite( stream, deconst( &cacheKey ), "cacheKey" );

    // Write version again
    stream->readOrWriteObjectVersion( version );
}

#if RTCORE_API_VERSION >= 25
ModuleEntryRefPair RTXPlan::deduplicateRtcModule( const LWDADevice* device, RtcCompiledModule newModule, Rtlw32 entryIndex ) const
{
    // Deduplicate SASS module
    Rtlw64 rtcHash;
    m_context->getRTCore()->compiledModuleGetHash( newModule, &rtcHash );

    CompiledModuleHandle handle;
    if( !m_compiledProgramCache->find( device, rtcHash, handle ) )
    {
        // Create and insert the handle
        RTCore* rtcore               = m_context->getRTCore();
        auto    deleteCompiledModule = [rtcore]( RtcCompiledModule cm ) { rtcore->compiledModuleDestroy( cm ); };
        handle = CompiledModuleHandle( newModule, deleteCompiledModule );
        m_compiledProgramCache->emplace( device, rtcHash, handle );
    }
    else
    {
        m_context->getRTCore()->compiledModuleDestroy( newModule );
    }

    return std::make_pair( handle, entryIndex );
}
#endif

ModuleEntryRefPair RTXPlan::compileProgramToRTX( const RTXCompile::Options& options, std::mutex& mutex ) const
{
    TIMEVIZ_SCOPE( "compileProgram" );

    // Collect all functions needed. Cast const away because 1. we are
    // cloning, and 2. the linker does not have const inputs.
    Function* entry                    = nullptr;
    bool      useGeneratedIntersection = options.cp->isUsedAsSemanticType( ST_INTERSECTION )
                                    && !options.cp->isBuiltInIntersection() && options.cp->callsReportIntersection();
    entry = const_cast<Function*>( useGeneratedIntersection ? options.cp->llvmIntersectionFunction() : options.cp->llvmFunction() );

    RTXCompile::AttributeDecoderList attributeDecoders;
    for( const CanonicalProgram* acp : options.attributeDecoders )
    {
        RT_ASSERT( acp->isUsedAsSingleSemanticType() );


        if( acp->isUsedAsSemanticType( ST_ATTRIBUTE ) )
        {
            // Add attribute program directly as the attribute decoder
            attributeDecoders.push_back( {const_cast<Function*>( acp->llvmFunction() ), acp->get32bitAttributeKind(), ST_ATTRIBUTE} );
        }
        else if( acp->isUsedAsSemanticType( ST_INTERSECTION ) )
        {
            // Add the automatically generated attribute decoder if it exists
            if( acp->callsReportIntersection() )
                attributeDecoders.push_back( {const_cast<Function*>( acp->llvmAttributeDecoder() ),
                                              acp->get32bitAttributeKind(), ST_INTERSECTION} );
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "Invalid attribute decoder" );
        }
    }

    // For parallel compilation we need to use a separate LLVM context
    // and clone all modules into that context.
    std::unique_ptr<llvm::LLVMContext>   newContext;
    std::vector<Module*> modulesToLink;
    std::unique_ptr<Module>              newModule;

    {
        TIMEVIZ_SCOPE( "Cloning modules" );
        newContext.reset( new llvm::LLVMContext() );
        LLVMContext& llvmContext = *newContext.get();

        // Start with a fresh copy of the runtime.  This ensures that struct type names are resolved
        // properly during linking.  Note that struct types receive unique ("dotted") names when
        // bitcode is loaded in an LLVM context in which its struct types are already defined.
        // Dotted struct names link properly when linking into a module with undotted names, but not
        // vice versa. In this case the RTXRuntime module has undotted names because it's
        // deserialized from scratch in a fresh LLVM context. The entry module might have dotted
        // names if it's loaded from the disk cache, because that happens in a shared LLVM context
        // in which various struct types have already been created.
        newModule = LLVMManager::getRTXRuntime( llvmContext );
        newModule->setDataLayout( optix::createDataLayoutForLwrrentProcess().getStringRepresentation() );

        addMissingLineInfoAndDump( newModule.get(), k_saveLLVM.get(), "RTXRuntime", 1, m_kernelLaunchCounter );

        // Enable/disable motion blur transforms
        setInt32GlobalInitializer( "RTX_hasMotionTransforms", m_hasMotionBlur, newModule.get() );

        // Prepare to link in the module for the entry point.
        Module* newEntryModule( corelib::cloneModuleBySerialization( entry->getParent(), llvmContext ) );
        entry = newEntryModule->getFunction( entry->getName() );
        RT_ASSERT( entry != nullptr );
        modulesToLink.push_back( std::move( newEntryModule ) );

        // Prepare to link the attribute decoders.
        for( RTXCompile::AttributeDecoderData& decoderData : attributeDecoders )
        {
            llvm::Module* decoderModule(
                corelib::cloneModuleBySerialization( decoderData.decoder->getParent(), llvmContext ) );
            decoderData.decoder = decoderModule->getFunction( decoderData.decoder->getName() );
            RT_ASSERT( decoderData.decoder != nullptr );
            modulesToLink.push_back( std::move( decoderModule ) );
        }
    }

    // End critical section
    ReverseLockGuard<std::mutex> guard( mutex );

    // Instrument the function with exception checks
    {
        const uint64_t exceptionFlags     = m_context->getExceptionFlags();
        const uint64_t maxPayloadSize     = options.compileParams.maxPayloadSize;
        const bool     payloadInRegisters = options.compileParams.payloadInRegisters;
        TIMEVIZ_SCOPE( "ExceptionInstrumenter" );
        {
            RTXExceptionInstrumenter exceptionInstrumenter( options.stype, exceptionFlags, maxPayloadSize,
                                                            payloadInRegisters, m_kernelLaunchCounter );
            exceptionInstrumenter.runOnFunction( entry );
        }

        // Also instrument the attribute decoders
        for( RTXCompile::AttributeDecoderData& decoderData : attributeDecoders )
        {
            // Note that an exception instrumenter cannot be used on more than one module because
            // it caches LLVM values internally (this arose in bug 2701705).
            RTXExceptionInstrumenter exceptionInstrumenter( options.stype, exceptionFlags, maxPayloadSize,
                                                            payloadInRegisters, m_kernelLaunchCounter );
            exceptionInstrumenter.runOnFunction( decoderData.decoder );
        }
    }

    // Specialize the function
    if( !k_skipSpecialization.get() )
    {
        TIMEVIZ_SCOPE( "Specializer" );
        const bool deviceSupportsLDG = LWDADevice::smVersionSupportsLDG( options.rtcoreCompileOptions.smVersion );
        const bool objectRecordInConstMemory = options.compileParams.constMemAllocFlags.objectRecordInConstMemory;
        RTXVariableSpecializer specializer( options.specializations, entry, options.stype, options.inheritedStype,
                                            deviceSupportsLDG, objectRecordInConstMemory,
                                            m_context->getProgramManager(), m_kernelLaunchCounter );
        std::string dumpName = RTXSpecializer::computeDumpName( options.stype, entry->getName().str() );
        specializer.runOnModule( entry->getParent(), dumpName );

        for( RTXCompile::AttributeDecoderData& decoderData : attributeDecoders )
        {
            Function*              decoderFunction = decoderData.decoder;
            SemanticType           stype           = decoderData.stype;
            RTXVariableSpecializer attributeSpecializer( options.specializations, decoderFunction, stype, stype,
                                                         deviceSupportsLDG, objectRecordInConstMemory,
                                                         m_context->getProgramManager(), m_kernelLaunchCounter );
            attributeSpecializer.runOnModule( decoderFunction->getParent(),
                                              dumpName + "-" + decoderData.decoder->getName().str() );
        }
    }

    {
        // Link functions together
        TIMEVIZ_SCOPE( "Link functions" );
        Linker linker( *newModule.get() );
        verifyModule( *newModule.get() );

        // Note: linking destroys the modules
        std::string entryName = entry->getName();
        for( Module* module : modulesToLink )
        {
            linkOrThrow( linker, module, /*preserveModule=*/true, "Error linking atribute decoder" );
        }
        entry = newModule->getFunction( entryName );
        RT_ASSERT_MSG( entry != nullptr, "Cannot find function after link" );

        // Get decoders from the new module
        for( RTXCompile::AttributeDecoderData& decoderData : attributeDecoders )
        {
            decoderData.decoder = newModule->getFunction( decoderData.decoder->getName() );
            RT_ASSERT_MSG( decoderData.decoder != nullptr, "Cannot find function after link" );
        }
    }

    {
        addMissingLineInfoAndDump( entry->getParent(), k_saveLLVM.get(), "RTXDemandBuffer", 1, m_kernelLaunchCounter );
        RTXDemandBufferSpecializer specializer;
        specializer.runOnFunction( entry );
    }

    RTXGlobalSpecializer globalSpecializer( m_dimensionality, options.specializations.m_minTransformDepth,
                                            options.specializations.m_maxTransformDepth,
                                            options.specializations.m_printEnabled, m_kernelLaunchCounter );
    globalSpecializer.runOnModule( entry->getParent(), RTXSpecializer::computeDumpName( options.stype, entry->getName().str() ) );

    // Compile the function to lwvm-rt
    std::string entryName;
    bool        fellBackToLWPTX = false;
    {
        TIMEVIZ_SCOPE( "Compile" );
        RTXCompile compile( options, attributeDecoders, m_context->getProgramManager(), m_kernelLaunchCounter );
        entryName = compile.runOnFunction( entry, fellBackToLWPTX );
    }

    // Compile the function from lwvm-rt to SASS
    ModuleEntryRefPair result;
    {
        TIMEVIZ_SCOPE( "rtcCompileModule" );
#if 0
        newModule->setDataLayout( optix::createDataLayoutForLwrrentProcess() );

        // TODO(Kincaid): LWVM 70 fails if a function doesn't have an
        // associated DISubprogram. The null program doesn't seem to be created
        // with one. Is this a bug in LWVM or should we be setting this
        // somewhere?
        // Extra weirdness: There is a subprogram for the function (it's
        // referenced indirectly as a scope by the ret instruction), but it's
        // not set as the subprogram for the function.
        for( auto func_it = newModule->begin(), end = newModule->end(); func_it != end; ++func_it )
        {
            if( func_it->getName().find( "null_program" ) != std::string::npos )
            {
                /* DIBuilder dbuilder( *newModule ); */
                /* DIFile* genFile = dbuilder.createFile( "internal", "internal" ); */
                /* DICompileUnit* compUnit = dbuilder.createCompileUnit( llvm::dwarf::DW_LANG_C_plus_plus, genFile, "optix", true, "", 0 ); */
                /* DIModule*      mod = dbuilder.createModule( compUnit, "internal", "internal", "internal", "internal" ); */
                /* std::vector<Metadata*> nullArgs = {}; */
                /* DITypeRefArray paramTypes = dbuilder.getOrCreateTypeArray( nullArgs ); */
                /* DISubroutineType* nullFuncType = dbuilder.createSubroutineType( paramTypes ); */
                /* DISubprogram* nullSP = dbuilder.createFunction( mod, "null_program", "global", genFile, 0, nullFuncType, false, true, 0 ); */
                /* func_it->setSubprogram( nullSP ); */
                for( auto bb_it = func_it->begin(), bb_end = func_it->end(); bb_it != bb_end; ++bb_it )
                {
                    for( auto inst_it = bb_it->begin(), inst_end = bb_it->end(); inst_it != inst_end; ++inst_it )
                    {
                        DISubprogram* subprog = dyn_cast<DISubprogram>( inst_it->getDebugLoc()->getScope()->getScope() );
                        func_it->setSubprogram( subprog );
                    }
                }
            }
        }
#endif
        std::string       serializedModuleBuffer = corelib::serializeModule( newModule.get() );
        RtcCompileOptions rtcOptionsCopy         = options.rtcoreCompileOptions;
        // the result of hasLineInfo() is already computed in RTXCompile::runOnFunction, but there's lwrrently no way
        // to return that information. We might want to change that.
        // Additionally, the options.rtcoreCompileOptions passed to this function are used by the caller
        // to construct the CompiledProgramCacheKey, which SHOULD contain this debugLevel, but unfortunately the settings
        // are determined for all modules globally and before any module is even accessible for us, so it can't be set
        // per-module at the moment.
        if( hasLineInfo( newModule.get() ) )
        {
            rtcOptionsCopy.debugLevel = max( rtcOptionsCopy.debugLevel, 1 );
        }

        if( fellBackToLWPTX )
            rtcOptionsCopy.useLWPTX = true;

        if( k_recordCompileCalls.get() )
            recordcompile::recordCompileCall( entryName, rtcOptionsCopy, serializedModuleBuffer );

        RtcCompiledModule rtcModule = nullptr;
        m_context->getRTCore()->compileModule( options.device->rtcContext(), &rtcOptionsCopy,
                                               serializedModuleBuffer.c_str(), serializedModuleBuffer.size(), &rtcModule );
#if RTCORE_API_VERSION >= 25
        Rtlw32 entryIndex = ~0;
        m_context->getRTCore()->compiledModuleGetEntryFunctionIndex( rtcModule, entryName.c_str(), &entryIndex );

        result = deduplicateRtcModule( options.device, rtcModule, entryIndex );
#else
        RTCore* rtcore = m_context->getRTCore();
        auto deleteCompiledModule = [rtcore]( RtcCompiledModule cm ) { rtcore->compiledModuleDestroy( cm ); };
        result = std::make_pair( CompiledModuleHandle( rtcModule, deleteCompiledModule ), entryName );
#endif
    }

    // Clean up explicitly such that it is still done in the parallel section before the lock is re-acquired
    {
        TIMEVIZ_SCOPE( "Destroying cloned module" );
        newModule.reset();
        newContext.reset();
    }

    return result;
}


void RTXPlan::eventContextSetAttributeStackSize( const size_t oldContinuationStackSize,
                                                 const size_t oldDirectCallableStackSizeFromTraversal,
                                                 const size_t oldDirectCallableStackSizeFromState,
                                                 const size_t newContinuationStackSize,
                                                 const size_t newDirectCallableStackSizeFromTraversal,
                                                 const size_t newDirectCallableStackSizeFromState )
{
    if( !isValid() )
        return;

    const bool useContextAttributesForStackSize = newContinuationStackSize != 0 || newDirectCallableStackSizeFromTraversal != 0
                                                  || newDirectCallableStackSizeFromState != 0;

    if( useContextAttributesForStackSize != m_useContextAttributesForStackSize )
    {
        ilwalidatePlan();
        return;
    }

    if( !useContextAttributesForStackSize )
        return;


    // Ilwalidate the plan if one of the stack sizes does not match its corresponding 'newSize'.
    // 'old*Size*' is ignored since the plan knows with which stack size it was created.
    if( m_continuationStackSize != newContinuationStackSize || m_directCallableStackSizeFromTraversal != newDirectCallableStackSizeFromTraversal
        || m_directCallableStackSizeFromState != newDirectCallableStackSizeFromState )
    {
        ilwalidatePlan();
    }
}

void RTXPlan::eventContextSetMaxCallableProgramDepth( const unsigned int oldMaxDepth, const unsigned int newMaxDepth )
{
    if( m_useContextAttributesForStackSize )
        return;

    // Ilwalidate the plan if the maximum callable program call depth does not match the 'newMaxDepth'.
    // 'oldMaxDepth' is ignored since the plan knows with which maximum callable program call depth it was created.
    if( m_maxCallableProgramDepth != newMaxDepth )
        ilwalidatePlan();
}

void RTXPlan::eventContextSetMaxTraceDepth( const unsigned int oldMaxDepth, const unsigned int newMaxDepth )
{
    // Ilwalidate the plan if the maximum trace depth does not match the 'newMaxDepth'.
    // 'oldMaxDepth' is ignored since the plan knows with which maximum trace depth it was created.
    if( m_maxTraceDepth != newMaxDepth )
        ilwalidatePlan();
}

void RTXPlan::eventContextMaxTransformDepthChanged( int oldDepth, int newDepth )
{
    // No need to ilwalidate the plan when the transform depth changes.
    if( m_maxTransformHeight != newDepth )
    {
        m_maxTransformHeight = newDepth;

        // Update the traversable graph depth of the frame task.
        FrameTask*    frameTask    = getTask();
        RTXFrameTask* rtxFrameTask = dynamic_cast<RTXFrameTask*>( frameTask );
        if( rtxFrameTask )
            rtxFrameTask->setTraversableGraphDepth( m_maxTransformHeight + m_maxAccelerationHeight );
    }
}

void RTXPlan::eventContextMaxAccelerationHeightChanged( int oldValue, int newValue )
{
    // No need to ilwalidate the plan when the transform depth changes.
    if( m_maxAccelerationHeight != newValue )
    {
        m_maxAccelerationHeight = newValue;

        // Update the traversable graph depth of the frame task.
        FrameTask*    frameTask    = getTask();
        RTXFrameTask* rtxFrameTask = dynamic_cast<RTXFrameTask*>( frameTask );
        if( rtxFrameTask )
            rtxFrameTask->setTraversableGraphDepth( m_maxTransformHeight + m_maxAccelerationHeight );
    }
}

void RTXPlan::eventContextNeedsUniversalTraversalChanged( bool needsUniversalTraversal )
{
    if( m_needsUniversalTraversal != needsUniversalTraversal )
        ilwalidatePlan();
}

void RTXPlan::eventContextHasMotionBlurChanged( bool hasMotionBlur )
{
    if( m_hasMotionBlur != hasMotionBlur )
        ilwalidatePlan();
}

void RTXPlan::eventPagingModeDidChange( PagingMode newMode )
{
    if( m_pagingMode != newMode )
        ilwalidatePlan();
}

void RTXPlan::createPlan( unsigned int entry, int dimensionality )
{
    TIMEVIZ_FUNC;

    m_entry            = entry;
    m_dimensionality   = dimensionality;
    MemoryManager*  mm = m_context->getMemoryManager();
    ProgramManager* pm = m_context->getProgramManager();

    Context::AttributeStackSize attrStackSize;
    m_context->getAttributeStackSize( attrStackSize );
    m_useContextAttributesForStackSize = attrStackSize.continuation != 0 || attrStackSize.direct_from_traversal != 0
                                         || attrStackSize.direct_from_state != 0;
    if( m_useContextAttributesForStackSize )
    {
        m_continuationStackSize                = attrStackSize.continuation;
        m_directCallableStackSizeFromTraversal = attrStackSize.direct_from_traversal;
        m_directCallableStackSizeFromState     = attrStackSize.direct_from_state;
    }

    m_maxCallableProgramDepth = m_context->getMaxCallableProgramDepth();
    m_maxTraceDepth           = m_context->getMaxTraceDepth();
    m_maxTransformHeight      = m_context->getBindingManager()->getMaxTransformHeight();
    m_maxAccelerationHeight   = m_context->getBindingManager()->getMaxAccelerationHeight();

    m_moduleCache.reset( new prodlib::ModuleCache() );

    // Create plan for each unique devices
    DeviceManager* dm = m_context->getDeviceManager();
    DeviceSet      udevices;
    for( int allDeviceListIndex : m_devices )
    {
        Device* device     = dm->allDevices()[allDeviceListIndex];
        int     udeviceIdx = device->uniqueDeviceListIndex();
        Device* udevice    = dm->uniqueActiveDevices()[udeviceIdx];
        udevices.insert( udevice );
    }
    m_perUniqueDevice.resize( udevices.count() );

    ConstantMemoryPlan::PerDeviceProgramPlan perDevicePrograms( udevices.count() );

    for( int allDeviceListIndex : udevices )
    {
        LWDADevice* device = deviceCast<LWDADevice>( dm->allDevices()[allDeviceListIndex] );
        RT_ASSERT_MSG( device != nullptr, "Non-lwca device supplied to RTXPlan" );

        // Fill in the device
        const int        uniqueDeviceIdx = device->uniqueDeviceListIndex();
        PerUniqueDevice& pud             = m_perUniqueDevice[uniqueDeviceIdx];
        pud.m_archetype                  = device;

        // Determine the number of bound textures.
        const BitSet&      bound         = mm->getAssignedTexReferences( device );
        const unsigned int numHWTex      = 1 + bound.findLast( true );
        const int          boundTextures = std::min( numHWTex, device->maximumBoundTextures() );
        pud.m_numBoundTextures = device->supportsHWBindlessTexture() ? boundTextures : device->maximumBoundTextures();

        // Find the canonical programs to use for this entry point
        pud.m_programPlan.reset( new ProgramPlan( this, m_context, entry, pud.m_archetype ) );

        // Determine the maximum size of the payload and if promotion to registers is possible.
        for( const CanonicalProgramID cpID : pud.m_programPlan->getAllReachablePrograms() )
        {
            const CanonicalProgram* cp    = pm->getCanonicalProgramById( cpID );
            pud.m_maxPayloadRegisterCount = std::max( pud.m_maxPayloadRegisterCount, cp->getMaxPayloadRegisterCount() );
            pud.m_canPromotePayload &=
                ( !cp->canPayloadPointerEscape() && !cp->hasDynamicAccessesToPayload()
                  && !( cp->isUsedAsSemanticType( ST_BOUND_CALLABLE_PROGRAM ) && cp->hasPayloadAccesses() ) );
        }

        // Assign attribute order & determine number of attribute registers.
        Rtlw64 maxAttributeRegisters = 0;
        m_context->getRTCore()->deviceContextGetLimit( device->rtcContext(), RTC_LIMIT_MAX_ATTRIBUTE_REGISTERS, &maxAttributeRegisters );

        if( k_maxAttributeRegCount.get() != -1 )
            maxAttributeRegisters = std::min( maxAttributeRegisters, (Rtlw64)k_maxAttributeRegCount.get() );

        const CPIDSet& isectPrograms = pud.m_programPlan->getReachablePrograms( ST_INTERSECTION );
        computeAttributeDataSizes( isectPrograms, (int)maxAttributeRegisters, pud.m_maxAttributeRegisterCount,
                                   pud.m_maxAttributeMemoryCount );

        // Decoders are made up of attribute programs and intersection programs.  It is
        // important to make the order list consistent from run to run, so sort the list
        // based on the 32 bit attribute kind.
        const CPIDSet& attrPrograms = pud.m_programPlan->getReachablePrograms( ST_ATTRIBUTE );
        pud.m_attributeDecoders.reserve( isectPrograms.size() + attrPrograms.size() );
        for( const CanonicalProgramID& cpid : isectPrograms )
        {
            const CanonicalProgram* cp = m_context->getProgramManager()->getCanonicalProgramById( cpid );
            if( cp->isBuiltInIntersection() )
                continue;
            pud.m_attributeDecoders.push_back( cp );
        }
        for( const CanonicalProgramID& cpid : attrPrograms )
            pud.m_attributeDecoders.push_back( m_context->getProgramManager()->getCanonicalProgramById( cpid ) );
        algorithm::sort( pud.m_attributeDecoders, []( const CanonicalProgram* left, const CanonicalProgram* right ) -> bool {
            return left->get32bitAttributeKind() < right->get32bitAttributeKind();
        } );

        // Determine the maximum size of callable program parameters.
        for( const CanonicalProgramID cpID : pud.m_programPlan->getReachablePrograms( ST_BINDLESS_CALLABLE_PROGRAM ) )
        {
            const int numCallableProgramRegisters = getMaxNumCallableProgramRegisters( cpID, false );

            pud.m_maxCallableProgramParamRegisterCount =
                std::max( pud.m_maxCallableProgramParamRegisterCount, numCallableProgramRegisters );
        }
        for( const CanonicalProgramID cpID : pud.m_programPlan->getReachablePrograms( ST_BOUND_CALLABLE_PROGRAM ) )
        {
            const int numCallableProgramRegisters = getMaxNumCallableProgramRegisters( cpID, true );

            pud.m_maxCallableProgramParamRegisterCount =
                std::max( pud.m_maxCallableProgramParamRegisterCount, numCallableProgramRegisters );
        }

        // Also consider the maximum size of bounding box program parameters.
        {
            LLVMContext& llvmContext = m_context->getLLVMManager()->llvmContext();
            Type*        i32Ty       = Type::getInt32Ty( llvmContext );
            Type*        i64Ty       = Type::getInt64Ty( llvmContext );

            // i32 for geometry instance handle, i32 for primitive index, i32 for motion index, i64 for AABB pointer
            int numBoundingBoxProgramParamRegisters = 0;
            numBoundingBoxProgramParamRegisters += corelib::getNumRequiredRegisters( i32Ty );
            numBoundingBoxProgramParamRegisters += corelib::getNumRequiredRegisters( i32Ty );
            numBoundingBoxProgramParamRegisters += corelib::getNumRequiredRegisters( i32Ty );
            numBoundingBoxProgramParamRegisters += corelib::getNumRequiredRegisters( i64Ty );
            pud.m_maxCallableProgramParamRegisterCount =
                std::max( pud.m_maxCallableProgramParamRegisterCount, numBoundingBoxProgramParamRegisters );
        }

        pud.m_programPlan->computeHeavyweightBindlessPrograms();

        perDevicePrograms[uniqueDeviceIdx] = std::make_pair( pud.m_archetype, pud.m_programPlan.get() );
    }

    // Compute variable and other specializations once for all devices.
    m_specializationPlan.reset( new SpecializationPlan( this, m_context, perDevicePrograms ) );

    // Reserve 2244 bytes of memory for RTCore. This includes space for:
    // - coreParams (192 bytes)
    // - maxAttributeFrameSize (4 bytes)
    // - stateFunPtrsInConstant (2048 bytes)
    //
    // RTCore dynamically sizes stateFunPtrsInConstant up to 16K.
    // However, OptiX only reserves enough for 256 states, to better
    // utilize constant memory in cases where there aren't that many
    // states. A state is generated any time there is a continuation
    // callable or heavyweight callsite (a program that might call a
    // bindless program containing an rtTrace call).
    size_t constBytesToReserve = 2244u;

    // If universal traversal is enabled, RTCore allocates an
    // additional 1024 bytes.
    if( m_context->RtxUniversalTraversalEnabled() )
        constBytesToReserve += 1024u;

    m_constantMemoryPlan.reset( new ConstantMemoryPlan( this, m_context, perDevicePrograms, constBytesToReserve,
                                                        /*deduplicateConstants*/ false ) );
}
// -----------------------------------------------------------------------------
int RTXPlan::getMaxNumCallableProgramRegisters( const CanonicalProgramID cpID, bool isBound ) const
{
    ProgramManager*         pm = m_context->getProgramManager();
    const CanonicalProgram* cp = pm->getCanonicalProgramById( cpID );
    const Function*         fn = cp->llvmFunction();

    // Accumulate the number of required registers for each CP parameter.
    Function::const_arg_iterator A  = fn->arg_begin();
    Function::const_arg_iterator AE = fn->arg_end();
    ++A;  // Skip CanonicalState* param.
    int numCallableProgramParamRegisters = 0;
    for( ; A != AE; ++A )
        numCallableProgramParamRegisters += corelib::getNumRequiredRegisters( A->getType() );

    if( isBound )
    {
        // Add parameters that will be added to bound callable programs during compile
        LLVMContext& llvmContext = m_context->getLLVMManager()->llvmContext();
        Type*        i64Ty       = Type::getInt64Ty( llvmContext );
        // The caller's SBT data pointer
        numCallableProgramParamRegisters += corelib::getNumRequiredRegisters( i64Ty );
        if( cp->isUsedAsInheritedSemanticTypes( {ST_MISS, ST_INTERSECTION, ST_CLOSEST_HIT, ST_ANY_HIT} ) )
        {
            // The bound callable program state pointer
            numCallableProgramParamRegisters += corelib::getNumRequiredRegisters( i64Ty );
        }
    }

    // Get the number of required registers for the result(s). The number of
    // "parameter" registers is the maximum of this number and the number of
    // required registers for the parameters.
    Type*     retTy                             = fn->getReturnType();
    const int numCallableProgramReturnRegisters = corelib::getNumRequiredRegisters( retTy );

    return std::max( numCallableProgramParamRegisters, numCallableProgramReturnRegisters );
}
// -----------------------------------------------------------------------------
RTXPlan::PerUniqueDevice::PerUniqueDevice( PerUniqueDevice&& other )
    : m_archetype( other.m_archetype )
    , m_numBoundTextures( other.m_numBoundTextures )
    , m_programPlan( std::move( other.m_programPlan ) )
    , m_canPromotePayload( other.m_canPromotePayload )
    , m_maxPayloadRegisterCount( other.m_maxPayloadRegisterCount )
    , m_maxAttributeRegisterCount( other.m_maxAttributeRegisterCount )
    , m_maxAttributeMemoryCount( other.m_maxAttributeMemoryCount )
    , m_attributeDecoders( std::move( other.m_attributeDecoders ) )
    , m_maxCallableProgramParamRegisterCount( other.m_maxCallableProgramParamRegisterCount )
{
}

// -----------------------------------------------------------------------------
bool RTXPlan::PerUniqueDevice::isCompatibleWith( const PerUniqueDevice& other ) const
{
    if( m_archetype != other.m_archetype )
        return false;
    if( m_numBoundTextures != other.m_numBoundTextures )
        return false;
    if( m_maxPayloadRegisterCount != other.m_maxPayloadRegisterCount )
        return false;
    if( m_canPromotePayload != other.m_canPromotePayload )
        return false;
    if( m_maxAttributeRegisterCount != other.m_maxAttributeRegisterCount )
        return false;
    if( m_maxAttributeMemoryCount != other.m_maxAttributeMemoryCount )
        return false;
    if( m_maxCallableProgramParamRegisterCount != other.m_maxCallableProgramParamRegisterCount )
        return false;

    if( m_attributeDecoders != other.m_attributeDecoders )
        return false;

    return m_programPlan->isCompatibleWith( *other.m_programPlan );
}
