/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#define OPTIX_DEFINE_ABI_VERSION_ONLY
#include <optix_function_table.h>
#undef OPTIX_DEFINE_ABI_VERSION_ONLY

#include <optix_types.h>

#include <exp/accel/ExtendedAccelHeader.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/OptixResultOneShot.h>
#include <exp/functionTable/compileOptionsTranslate.h>
#include <exp/pipeline/Compile.h>
#include <exp/pipeline/O7Runtime.h>
#include <exp/pipeline/O7Runtime_bin.h>
#include <exp/pipeline/TextureFootprintHW_ptx_bin.h>
#include <exp/pipeline/TextureFootprintSW_ptx_bin.h>

#include <rtcore/interface/types.h>

#include <Compile/UnnamedToGlobalPass.h>
#include <ExelwtionStrategy/RTX/RTXCompile.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/PTX/DataLayout.h>
#include <FrontEnd/PTX/LinkPTXFrontEndIntrinsics.h>
#include <FrontEnd/PTX/PTXtoLLVM.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Regex.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/DJB.h>

// Temporary include until we've fully moved over to lwvm70
#include <lwvm/Support/APIUpgradeUtilities.h>

#include <algorithm>
#include <array>
#include <initializer_list>

#define STR( x ) #x
#define STRINGIFY( x ) STR( x )

#define RTC_NUM_EXCEPTION_DETAILS 23

// NOTE: When adding knobs that influence the compilation result, do not use their value directly during
//       compile. Add their value to the InternalCompileParameters struct instead, set them in
//       setInternalCompileOptions and read from CompilationUnit::compileParams.
//       Also make sure to update the hashInternalCompileParameters function in Module.cpp, so they
//       get added to the cache key.
namespace {
// clang-format off
Knob<int>          k_maxParamRegCount( RT_DSTRING( "o7.compile.maxParamRegCount" ), 4, RT_DSTRING( "Maximum registers for callable program arguments. Arguments beyond the limit are passed on the stack. Defaults to 4 except when no-inline is enabled, then the default is 18. -1 = use rtcore limit." ) );
Knob<int>          k_maxRegCount( RT_DSTRING( "o7.compile.maxRegCount" ), 0, RT_DSTRING( "Maximum register count passed to rtcore.  Takes precedence when set." ) );
Knob<std::string>  k_saveLLVM( RT_DSTRING( "o7.compile.saveLLVM" ), "", RT_DSTRING( "Save LLVM stages during compilation." ) );
Knob<std::string>  k_saveLLVMFunctions( RT_DSTRING( "o7.compile.saveLLVMFunctions" ), "", RT_DSTRING( "Comma separated list of functions to limit LLVM dumps to." ) );
Knob<std::string>  k_enableProfiling( RT_DSTRING( "o7.enableProfiling" ), "", RT_DSTRING( "String when non-empty enables profiling and stores the result to the prefix defined here" ) );
Knob<std::string>  k_overrideAbiVariant( RT_DSTRING( "o7.overrideAbiVariant" ), "", RT_DSTRING( "String when non-empty overrides the default ABI variant [ttu_a|utrav]" ) );
Knob<int>          k_inlineUpperCallLimit( RT_DSTRING( "o7.compile.inlineUpperCallLimit" ), -1, RT_DSTRING( "Functions which are called more often than this value are force-inlined and removed from the module even if they have external linkage. Default -1 means disable. Only in effect when Noinline compilation is enabled." ) );
Knob<int>          k_inlineLowerCallLimit( RT_DSTRING( "o7.compile.inlineLowerCallLimit" ), -1, RT_DSTRING( "Functions which are called less often than this value are force-inlined and removed from the module even if they have external linkage. Default -1 means disable. Only in effect when Noinline compilation is enabled." ) );
Knob<int>          k_inlineInstructionLimit( RT_DSTRING( "o7.compile.inlineInstructionLimit" ), -1, RT_DSTRING( "Functions which would be inlined based on \"o7.compile.inlineUpperCallLimit\" or \"o7.compile.inlineLowerCallLimit\" but contain more instructions than this number are not inlined. Default -1 means disable. Only in effect when Noinline compilation is enabled." ) );
Knob<bool>         k_removeUnusedNoinlineFunctions( RT_DSTRING( "o7.compile.removeUnusedNoinlineFunctions" ), false, RT_DSTRING( "When set to true Noinline functions that are never called are removed even if they have external linkage. Only in effect when Noinline compilation is enabled." ) );
Knob<std::string>  k_forceInlineSet( RT_DSTRING( "o7.compile.forceInlineSet"), "", RT_DSTRING( "Set of functions that will be forcibly inlined.  Comma separated.  Can be filtered by callers with callee->caller syntax.  Names are matched as substrings, so assume *callee* as the matching expression" ) );
Knob<std::string>  k_disableNoinlineFunc( RT_DSTRING( "o7.compile.disableNoinlineFunc" ), "", RT_DSTRING( "Comma separated list of LWCA functions (substring matching) for which not to enable calling OptiX API functions. Only in effect when Noinline compilation is enabled." ) );
Knob<bool>         k_enableAllExceptions( RT_DSTRING( "o7.enableAllExceptions" ), false, RT_DSTRING( "When set to true all exception types are enabled. Default false means disabled." ) );
Knob<bool>         k_allowIndirectFunctionCalls( RT_DSTRING( "o7.compile.allowIndirectFunctionCalls" ), false, RT_DSTRING( "When set to true the validation check for indirect function calls is skipped." ) );
Knob<bool>         k_disableActiveMaskCheck( RT_DSTRING( "o7.compile.disableActiveMaskCheck" ), false, RT_DSTRING( "When set to true the validation of warp wide synchronization intrinsics is skipped." ) );
Knob<bool>         k_enableSpecializations( RT_DSTRING( "o7.compile.enableLaunchParamSpecialization" ), true, RT_DSTRING( "Enable the use of specialized launch params." ) );
Knob<bool>         k_enableLwstomABIProcessing( RT_DSTRING( "o7.compile.enableLwstomABIProcessing" ), true, RT_DSTRING( "Enable automatic translating certain functions to use a custom ABI" ) );
Knob<int>          k_numAdditionalABIScratchRegs( RT_DSTRING( "o7.compile.numAdditionalABIScratchRegs" ), 2, RT_DSTRING( "Number of additional scratch registers to use when using custom ABI" ) );
Knob<bool>         k_enableCoroutines( RT_DSTRING( "compile.enableCoroutines" ), false, RT_DSTRING( "Enable coroutines.") );
Knob<bool>         k_disableAllExceptions( RT_DSTRING( "o7.disableAllExceptions" ), false, RT_DSTRING( "When set to true all exception types are disabled. Default false means normal set is enabled (but not all)." ) );
Knob<bool>         k_hideInternalModules( RT_PUBLIC_DSTRING( "o7.hideInternalModules" ), true, RT_PUBLIC_DSTRING( "Hide functions in internal modules in public lwpu tools." ) );
Knob<bool>         k_allowOldIntrinsics( RT_PUBLIC_DSTRING( "o7.allowOldIntrinsics" ), false, RT_PUBLIC_DSTRING( "Allow old intrinsics with new ABIs" ) );
Knob<unsigned int> k_splitModuleLogLevel( RT_DSTRING( "o7.splitModuleLogLevel" ), 30, RT_DSTRING( "Log level to start printing information about the module splitting" ) );
HiddenPublicKnob<bool> k_useSoftwareTextureFootprint( RT_PUBLIC_DSTRING( "o7.compile.useSoftwareTextureFootprint" ), false, RT_PUBLIC_DSTRING( "Use software emulation of texture footprint intrinsic." ) );
Knob<bool>         k_forceDebugMode( RT_DSTRING( "o7.forceDebugMode" ), false, RT_DSTRING( "Force OptixModuleCompileOptions::debugLevel to OPTIX_COMPILE_DEBUG_LEVEL_FULL and OptixModuleCompileOptions::optLevel to OPTIX_COMPILE_OPTIMIZATION_LEVEL_0." ) );
Knob<int>          k_overwriteOptLevel( RT_DSTRING( "o7.overwriteOptLevel" ), -1, RT_DSTRING( "Overwrite the OptixModuleCompileOptions::optLevel. Valid values: -1 no overwrite (default), 0, 1, 2, 3" ) );
Knob<int>          k_overwriteDbgLevel( RT_DSTRING( "o7.overwriteDebugLevel" ), -1, RT_DSTRING( "Overwrite the OptixModuleCompileOptions::debugLevel. Valid values: -1 no overwrite (default), 0, 1, 2, 3" ) );
Knob<bool>         k_serializeModuleId( RT_DSTRING( "o7.compile.serializeModuleId" ), false, RT_DSTRING( "When creating mangled names, use a serial number instead of a hash of the input. Useful for diffing runs." ) );

// clang-format on
}  // namespace

namespace optix_exp {

// Before compilation we build up a lookup array with the llvm::Functions
// of all intrinsics that we are going to handle.
enum IntrinsicIndex
{
    optix_trace_0 = 0,
    optix_trace_1,
    optix_trace_2,
    optix_trace_3,
    optix_trace_4,
    optix_trace_5,
    optix_trace_6,
    optix_trace_7,
    optix_trace_8,
    optix_trace_32,
    optix_trace_typed_32,
    optix_get_payload_0,
    optix_get_payload_1,
    optix_get_payload_2,
    optix_get_payload_3,
    optix_get_payload_4,
    optix_get_payload_5,
    optix_get_payload_6,
    optix_get_payload_7,
    optix_get_payload,
    optix_set_payload_0,
    optix_set_payload_1,
    optix_set_payload_2,
    optix_set_payload_3,
    optix_set_payload_4,
    optix_set_payload_5,
    optix_set_payload_6,
    optix_set_payload_7,
    optix_set_payload,
    optix_set_payload_types,
    optix_undef_value,
    optix_get_world_ray_origin_x,
    optix_get_world_ray_origin_y,
    optix_get_world_ray_origin_z,
    optix_get_world_ray_direction_x,
    optix_get_world_ray_direction_y,
    optix_get_world_ray_direction_z,
    optix_get_object_ray_origin_x,
    optix_get_object_ray_origin_y,
    optix_get_object_ray_origin_z,
    optix_get_object_ray_direction_x,
    optix_get_object_ray_direction_y,
    optix_get_object_ray_direction_z,
    optix_get_ray_tmin,
    optix_get_ray_tmax,
    optix_get_ray_time,
    optix_get_ray_flags,
    optix_get_ray_visibility_mask,
    optix_get_instance_traversable_from_ias,
    optix_get_triangle_vertex_data,
    optix_get_linear_lwrve_vertex_data,
    optix_get_quadratic_bspline_vertex_data,
    optix_get_lwbic_bspline_vertex_data,
    optix_get_catmullrom_vertex_data,
    optix_get_sphere_data,
    optix_get_gas_traversable_handle,
    optix_get_gas_motion_time_begin,
    optix_get_gas_motion_time_end,
    optix_get_gas_motion_step_count,
    optix_get_gas_ptr,
    optix_get_transform_list_size,
    optix_get_transform_list_handle,
    optix_get_transform_type_from_handle,
    optix_get_static_transform_from_handle,
    optix_get_srt_motion_transform_from_handle,
    optix_get_matrix_motion_transform_from_handle,
    optix_get_instance_id_from_handle,
    optix_get_instance_child_from_handle,
    optix_get_instance_transform_from_handle,
    optix_get_instance_ilwerse_transform_from_handle,
    optix_report_intersection_0,
    optix_report_intersection_1,
    optix_report_intersection_2,
    optix_report_intersection_3,
    optix_report_intersection_4,
    optix_report_intersection_5,
    optix_report_intersection_6,
    optix_report_intersection_7,
    optix_report_intersection_8,
    optix_get_attribute_0,
    optix_get_attribute_1,
    optix_get_attribute_2,
    optix_get_attribute_3,
    optix_get_attribute_4,
    optix_get_attribute_5,
    optix_get_attribute_6,
    optix_get_attribute_7,
    optix_terminate_ray,
    optix_ignore_intersection,
    optix_read_primitive_idx,
    optix_read_prim_va,
    optix_read_key_time,
    optix_read_sbt_gas_idx,
    optix_read_instance_id,
    optix_read_instance_idx,
    optix_get_hit_kind,
    optix_get_primitive_type_from_hit_kind,
    optix_is_hitkind_backface,
    optix_get_triangle_barycentrics,
    optix_get_launch_index_x,
    optix_get_launch_index_y,
    optix_get_launch_index_z,
    optix_get_launch_dimension_x,
    optix_get_launch_dimension_y,
    optix_get_launch_dimension_z,
    optix_get_sbt_data_ptr_64,
    optix_throw_exception_0,
    optix_throw_exception_1,
    optix_throw_exception_2,
    optix_throw_exception_3,
    optix_throw_exception_4,
    optix_throw_exception_5,
    optix_throw_exception_6,
    optix_throw_exception_7,
    optix_throw_exception_8,
    optix_get_exception_code,
    optix_get_exception_detail_0,
    optix_get_exception_detail_1,
    optix_get_exception_detail_2,
    optix_get_exception_detail_3,
    optix_get_exception_detail_4,
    optix_get_exception_detail_5,
    optix_get_exception_detail_6,
    optix_get_exception_detail_7,
    optix_get_exception_ilwalid_traversable,
    optix_get_exception_ilwalid_sbt_offset,
    optix_get_exception_ilwalid_ray,
    optix_get_exception_parameter_mismatch,
    optix_get_exception_line_info,
    optix_call_direct_callable,
    optix_call_continuation_callable,
    optix_tex_footprint_2d,
    optix_tex_footprint_2d_grad,
    optix_tex_footprint_2d_lod,
    optix_tex_footprint_2d_v2,
    optix_tex_footprint_2d_grad_v2,
    optix_tex_footprint_2d_lod_v2,
    optix_private_get_compile_time_constant,
    intrinsicCount
};

// clang-format off
#define ASSERT_ENUM_CONTINUITY_7( prefix )                    \
    static_assert( prefix##_0 == (prefix##_1 - 1) &&          \
                   prefix##_0 == (prefix##_2 - 2) &&          \
                   prefix##_0 == (prefix##_3 - 3) &&          \
                   prefix##_0 == (prefix##_4 - 4) &&          \
                   prefix##_0 == (prefix##_5 - 5) &&          \
                   prefix##_0 == (prefix##_6 - 6) &&          \
                   prefix##_0 == (prefix##_7 - 7),            \
                   "Continuity error in enum values for " #prefix )

#define ASSERT_ENUM_CONTINUITY_8( prefix )                    \
    static_assert( prefix##_0 == (prefix##_1 - 1) &&          \
                   prefix##_0 == (prefix##_2 - 2) &&          \
                   prefix##_0 == (prefix##_3 - 3) &&          \
                   prefix##_0 == (prefix##_4 - 4) &&          \
                   prefix##_0 == (prefix##_5 - 5) &&          \
                   prefix##_0 == (prefix##_6 - 6) &&          \
                   prefix##_0 == (prefix##_7 - 7) &&          \
                   prefix##_0 == (prefix##_8 - 8),            \
                   "Continuity error in enum values for " #prefix )
// clang-format on

ASSERT_ENUM_CONTINUITY_8( optix_trace );
ASSERT_ENUM_CONTINUITY_8( optix_report_intersection );
ASSERT_ENUM_CONTINUITY_8( optix_throw_exception );
ASSERT_ENUM_CONTINUITY_7( optix_get_payload );
ASSERT_ENUM_CONTINUITY_7( optix_set_payload );
ASSERT_ENUM_CONTINUITY_7( optix_get_attribute );
ASSERT_ENUM_CONTINUITY_7( optix_get_exception_detail );


// Information about OptiX PTX intrinsics
struct IntrinsicInfo
{
    // The minimum ABI version needed for use of this intrinsic being valid
    OptixABI minimumAbiVersion;
    // The maximum ABI version needed for use of this intrinsic being valid
    OptixABI maximumAbiVersion;
    // Intrinsic name as defined in PTX
    std::string intrinsicName;
    // Name of the API function that uses this PTX intrinsic (e.g. used for semantic type error messages)
    std::string apiName;
    // Signature of the API function that uses this intrinsic (used for ABI version mismatch error message)
    std::string signature;
};

using IntrinsicInfos = std::array<IntrinsicInfo, intrinsicCount>;

static const std::string& intrinsicName( IntrinsicIndex index );
static const std::string& apiName( IntrinsicIndex index );
static const std::string& signature( IntrinsicIndex index );
static OptixABI minimumAbiVersion( IntrinsicIndex index );
static OptixABI maximumAbiVersion( IntrinsicIndex index );

class InitialCompilationUnit
{
public:
    Module* optixModule = nullptr;
    llvm::Module* llvmModule = nullptr;
    const InternalCompileParameters& compileParams;

    InitialCompilationUnit( optix_exp::Module* optixModuleP, llvm::Module* module, const InternalCompileParameters& compileParams )
        : optixModule( optixModuleP )
        , llvmModule( module )
        , compileParams( compileParams )
    {}

    // The llvm::Functions for the Optix intrinsics
    std::array<llvm::Function*, intrinsicCount> llvmIntrinsics;

    OptixResult initAndVerifyIntrinsics( ErrorDetails& errDetails )
    {
        OptixResultOneShot result;
        for( int i = 0; i < intrinsicCount; ++i )
        {
            llvmIntrinsics[i] = llvmModule->getFunction( intrinsicName( static_cast<IntrinsicIndex>( i ) ) );
            result += verifyIntrinsicAbiVersion( static_cast<IntrinsicIndex>( i ), errDetails );
        }
        return result;
    }

    static const IntrinsicInfos& intrinsicsInfo();

  private:
    OptixResult verifyIntrinsicAbiVersion( IntrinsicIndex intrinsic, ErrorDetails& errDetails );

    static IntrinsicInfos initializeIntrinsicsInfo();
};

class CompilationUnit
{
  public:
    int  usedPayloadValues     = 0;
    int  usedAttributeValues   = 0;

    // maps caller's function names to number of calls
    std::map<std::string, unsigned int> traceCalls;
    std::map<std::string, unsigned int> dcCalls;
    std::map<std::string, unsigned int> ccCalls;

    void recordTraceCall( llvm::Function* caller )
    {
        auto it = traceCalls.insert( {caller->getName().str(), 1} );
        if( !it.second )
            ++it.first->second;
    }
    void recordCallableCall( llvm::Function* caller, IntrinsicIndex intrinsic )
    {
        if( intrinsic == optix_call_continuation_callable )
        {
            auto it = ccCalls.insert( {caller->getName().str(), 1} );
            if( !it.second )
                ++it.first->second;
        }
        else if( intrinsic == optix_call_direct_callable )
        {
            auto it = dcCalls.insert( {caller->getName().str(), 1} );
            if( !it.second )
                ++it.first->second;
        }
    }

    SubModule* subModule = nullptr;
    // The module to compile
    llvm::Module* llvmModule = nullptr;

    const InternalCompileParameters& compileParams;

    std::map<std::string, llvm::Constant*> m_stringsCache;

    // The llvm::Functions for the Optix intrinsics
    std::array<llvm::Function*, intrinsicCount> llvmIntrinsics;

    CompilationUnit( optix_exp::SubModule* subModuleP, llvm::Module* module, const InternalCompileParameters& compileParams )
        : subModule( subModuleP )
        , llvmModule( module )
        , compileParams( compileParams )
    {
        addNamedConstantName( compileParams.pipelineLaunchParamsVariableName );
    }

    OptixResult init( ErrorDetails& errDetails )
    {
        OptixResultOneShot result;
        for( int i = 0; i < intrinsicCount; ++i )
        {
            llvmIntrinsics[i] = llvmModule->getFunction( intrinsicName( static_cast<IntrinsicIndex>( i ) ) );
        }
        return result;
    }

    // If there are no remaining calls to the function, remove it from the module and set
    // the entry in llvmIntrinsics to nullptr.
    //
    // No error handling here if there are any uses. This should only happen after
    // previous errors. And if not it will fail during linking anyway.
    void eraseIntrinsic( IntrinsicIndex index )
    {
        llvm::Function*& intrinsic = llvmIntrinsics[index];
        if( intrinsic && intrinsic->use_empty() )
            intrinsic->eraseFromParent();
        intrinsic = nullptr;
    }

    void addNamedConstantName( const std::string& name )
    {
        if( !name.empty() )
            namedConstantsNames.push_back( name );
    }
    const std::vector<std::string>& getNamedConstantsNames() const { return namedConstantsNames; }

  private:
    std::vector<std::string> namedConstantsNames;
};

// Little helper class for being verbose about the ABI selection.
class ABIDecisionLogger
{
  public:
    void addDecision( const std::string& decision )
    {
        if( !m_decisions.empty() )
            m_decisions += ", ";
        m_decisions += decision;
    }

    void setAbiAndPrint( RtcAbiVariant abi )
    {
        setAbi( abi );
        printABISelectionLogic();
    }

  private:
    std::string m_abi = "default";
    std::string m_decisions;

    void setAbi( RtcAbiVariant abi )
    {
        switch( abi )
        {
            case RTC_ABI_VARIANT_DEFAULT:
                m_abi = "default";
                break;
            case RTC_ABI_VARIANT_TTU_A:
                m_abi = RT_DSTRING( "ttu_a" );
                break;
            case RTC_ABI_VARIANT_TTU_B:
                m_abi = RT_DSTRING( "ttu_b" );
                break;
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
            case RTC_ABI_VARIANT_TTU_D:
                m_abi = RT_DSTRING( "ttu_d" );
                break;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
            case RTC_ABI_VARIANT_UTRAV:
                m_abi = "utrav";
                break;
            case RTC_ABI_VARIANT_MTTU:
                m_abi = RT_DSTRING( "mttu" );
                break;
            default:
                m_abi = "unhandled";
                break;
        }
    }
    void printABISelectionLogic() const { llog( 9 ) << m_abi << " chosen due to " << m_decisions << std::endl; }
};

OptixResult InitialCompilationUnit::verifyIntrinsicAbiVersion( IntrinsicIndex intrinsic, ErrorDetails& errDetails )
{
    OptixResult result = OPTIX_SUCCESS;
    if( llvmIntrinsics[intrinsic] && !llvmIntrinsics[intrinsic]->use_empty() )
    {
        if( compileParams.abiVersion < minimumAbiVersion( intrinsic ) )
        {
            errDetails.m_compilerFeedback << "Error: Function \"" << signature( intrinsic )
                                          << "\" (PTX: " << intrinsicName( intrinsic )
                                          << ") needs at least ABI version " << (int)minimumAbiVersion( intrinsic )
                                          << " (current: " << (int)compileParams.abiVersion << ").";

            result = OPTIX_ERROR_ILWALID_FUNCTION_USE;
        }
        else if( compileParams.abiVersion > maximumAbiVersion( intrinsic ) && !k_allowOldIntrinsics.get() )
        {
            errDetails.m_compilerFeedback
                << "Error: Function \"" << signature( intrinsic ) << "\" (PTX: " << intrinsicName( intrinsic )
                << ") is not supported with an ABI version higher than " << (int)maximumAbiVersion( intrinsic )
                << " (current: " << (int)compileParams.abiVersion
                << ").  Please recompile your PTX with a newer version of OptiX.";
            result = OPTIX_ERROR_ILWALID_FUNCTION_USE;
        }
        if( result )
        {
            errDetails.m_compilerFeedback << apiName( intrinsic ) << " is used in: ";
            bool first = true;
            for( llvm::CallInst* call : corelib::getCallsToFunction( llvmIntrinsics[intrinsic] ) )
            {
                if( !first )
                    errDetails.m_compilerFeedback << ", ";
                first                  = false;
                llvm::Function* caller = call->getParent()->getParent();
                errDetails.m_compilerFeedback << caller->getName().str();
            }
            errDetails.m_compilerFeedback << "\n";
        }
    }
    return result;
}

IntrinsicInfos InitialCompilationUnit::initializeIntrinsicsInfo()
{
    // This function is only called once by intrinsicsInfo().

    // It only exists to ensure correct order of the entries in the function info.
    // We could initialize them inline in intrinsicsInfo() using initializer lists,
    // but this way each entry is explicitly tied to the index variable to avoid
    // errors in the order.
    IntrinsicInfos info;

    // clang-format off
    //                                                         min/max ABI version                          PTX intrinsic name                                     API function name                               API function signature
    info[optix_trace_0]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_0",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_trace_1]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_1",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int& )" },
    info[optix_trace_2]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_2",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int& )" },
    info[optix_trace_3]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_3",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_4]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_4",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_5]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_5",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_6]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_6",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_7]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_7",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_8]                                      = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_trace_8",                                      "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_32]                                     = { OptixABI::ABI_46,   OptixABI::ABI_LWRRENT, "_optix_trace_32",                                     "optixTrace",                                   "void optixTrace( OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_trace_typed_32]                               = { OptixABI::ABI_47,   OptixABI::ABI_LWRRENT, "_optix_trace_typed_32",                               "optixTrace",                                   "void optixTrace( OptixPayloadTypeID, OptixTraversableHandle, float3, float3, float, float, float, OptixVisibilityMask, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int&, unsigned int& )" },
    info[optix_get_payload_0]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_0",                                "optixGetPayload_0",                            "unsigned int optixGetPayload_0()" },
    info[optix_get_payload_1]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_1",                                "optixGetPayload_1",                            "unsigned int optixGetPayload_1()" },
    info[optix_get_payload_2]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_2",                                "optixGetPayload_2",                            "unsigned int optixGetPayload_2()" },
    info[optix_get_payload_3]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_3",                                "optixGetPayload_3",                            "unsigned int optixGetPayload_3()" },
    info[optix_get_payload_4]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_4",                                "optixGetPayload_4",                            "unsigned int optixGetPayload_4()" },
    info[optix_get_payload_5]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_5",                                "optixGetPayload_5",                            "unsigned int optixGetPayload_5()" },
    info[optix_get_payload_6]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_6",                                "optixGetPayload_6",                            "unsigned int optixGetPayload_6()" },
    info[optix_get_payload_7]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_get_payload_7",                                "optixGetPayload_7",                            "unsigned int optixGetPayload_7()" },
    info[optix_get_payload]                                  = { OptixABI::ABI_46,   OptixABI::ABI_LWRRENT, "_optix_get_payload",                                  "optixGetPayload",                              "unsigned int optixGetPayload()" },
    info[optix_set_payload_0]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_0",                                "optixSetPayload_0",                            "void optixSetPayload_0( unsigned int )" },
    info[optix_set_payload_1]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_1",                                "optixSetPayload_1",                            "void optixSetPayload_1( unsigned int )" },
    info[optix_set_payload_2]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_2",                                "optixSetPayload_2",                            "void optixSetPayload_2( unsigned int )" },
    info[optix_set_payload_3]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_3",                                "optixSetPayload_3",                            "void optixSetPayload_3( unsigned int )" },
    info[optix_set_payload_4]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_4",                                "optixSetPayload_4",                            "void optixSetPayload_4( unsigned int )" },
    info[optix_set_payload_5]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_5",                                "optixSetPayload_5",                            "void optixSetPayload_5( unsigned int )" },
    info[optix_set_payload_6]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_6",                                "optixSetPayload_6",                            "void optixSetPayload_6( unsigned int )" },
    info[optix_set_payload_7]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_45,      "_optix_set_payload_7",                                "optixSetPayload_7",                            "void optixSetPayload_7( unsigned int )" },
    info[optix_set_payload]                                  = { OptixABI::ABI_46,   OptixABI::ABI_LWRRENT, "_optix_set_payload",                                  "optixSetPayload",                              "void optixSetPayload( unsigned int, unsigned int )" },
    info[optix_set_payload_types]                            = { OptixABI::ABI_47,   OptixABI::ABI_LWRRENT, "_optix_set_payload_types",                            "optixSetPayloadTypes",                         "void optixPayloadTypes( unsigned int )" },
    info[optix_undef_value]                                  = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_undef_value",                                  "optixUndefinedValue",                          "unsigned int optixUndefinedValue()" },
    info[optix_get_world_ray_origin_x]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_origin_x",                       "optixGetWorldRayOrigin",                       "float3 optixGetWorldRayOrigin()" },
    info[optix_get_world_ray_origin_y]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_origin_y",                       "optixGetWorldRayOrigin",                       "float3 optixGetWorldRayOrigin()" },
    info[optix_get_world_ray_origin_z]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_origin_z",                       "optixGetWorldRayOrigin",                       "float3 optixGetWorldRayOrigin()" },
    info[optix_get_world_ray_direction_x]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_direction_x",                    "optixGetWorldRayDirection",                    "float3 optixGetWorldRayDirection()" },
    info[optix_get_world_ray_direction_y]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_direction_y",                    "optixGetWorldRayDirection",                    "float3 optixGetWorldRayDirection()" },
    info[optix_get_world_ray_direction_z]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_world_ray_direction_z",                    "optixGetWorldRayDirection",                    "float3 optixGetWorldRayDirection()" },
    info[optix_get_object_ray_origin_x]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_origin_x",                      "optixGetObjectRayOrigin",                      "float3 optixGetObjectRayOrigin()" },
    info[optix_get_object_ray_origin_y]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_origin_y",                      "optixGetObjectRayOrigin",                      "float3 optixGetObjectRayOrigin()" },
    info[optix_get_object_ray_origin_z]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_origin_z",                      "optixGetObjectRayOrigin",                      "float3 optixGetObjectRayOrigin()" },
    info[optix_get_object_ray_direction_x]                   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_direction_x",                   "optixGetObjectRayDirection",                   "float3 optixGetObjectRayDirection()" },
    info[optix_get_object_ray_direction_y]                   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_direction_y",                   "optixGetObjectRayDirection",                   "float3 optixGetObjectRayDirection()" },
    info[optix_get_object_ray_direction_z]                   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_object_ray_direction_z",                   "optixGetObjectRayDirection",                   "float3 optixGetObjectRayDirection()" },
    info[optix_get_ray_tmin]                                 = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_ray_tmin",                                 "optixGetRayTmin",                              "float optixGetRayTmin()" },
    info[optix_get_ray_tmax]                                 = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_ray_tmax",                                 "optixGetRayTmax",                              "float optixGetRayTmax()" },
    info[optix_get_ray_time]                                 = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_ray_time",                                 "optixGetRayTime",                              "float optixGetRayTime()" },
    info[optix_get_ray_flags]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_ray_flags",                                "optixGetRayFlags",                             "unsigned int optixGetRayFlags()" },
    info[optix_get_ray_visibility_mask]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_ray_visibility_mask",                      "optixGetRayVisibilityMask",                    "unsigned int optixGetRayVisibilityMask()" },
    info[optix_get_instance_traversable_from_ias]            = { OptixABI::ABI_42,   OptixABI::ABI_LWRRENT, "_optix_get_instance_traversable_from_ias",            "optixGetInstanceTraversableFromIAS",           "OptixTraversableHandle optixGetInstanceTraversableFromIAS( OptixTraversableHandle ias, unsigned int instIdx )" },
    info[optix_get_triangle_vertex_data]                     = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_triangle_vertex_data",                     "optixGetTriangleVertexData",                   "void optixGetTriangleVertexData( OptixTraversableHandle, unsigned int, unsigned int, float, float3[3] )" },
    info[optix_get_linear_lwrve_vertex_data]                 = { OptixABI::ABI_30,   OptixABI::ABI_LWRRENT, "_optix_get_linear_lwrve_vertex_data",                 "optixGetLinearLwrveVertexData",                "void optixGetLinearLwrveVertexData( OptixTraversableHandle, unsigned int, unsigned int, float, float4[2] )" },
    info[optix_get_quadratic_bspline_vertex_data]            = { OptixABI::ABI_27,   OptixABI::ABI_LWRRENT, "_optix_get_quadratic_bspline_vertex_data",            "optixGetQuadraticBSplineVertexData",           "void optixGetQuadraticBSplineVertexData( OptixTraversableHandle, unsigned int, unsigned int, float, float4[3] )" },
    info[optix_get_lwbic_bspline_vertex_data]                = { OptixABI::ABI_27,   OptixABI::ABI_LWRRENT, "_optix_get_lwbic_bspline_vertex_data",                "optixGetLwbicBSplineVertexData",               "void optixGetLwbicBSplineVertexData( OptixTraversableHandle, unsigned int, unsigned int, float, float4[4] )" },
    info[optix_get_catmullrom_vertex_data]                   = { OptixABI::ABI_52,   OptixABI::ABI_LWRRENT, "_optix_get_catmullrom_vertex_data",                   "optixGetCatmullRomVertexData",                 "void optixGetCatmullRomVertexData( OptixTraversableHandle, unsigned int, unsigned int, float, float4[4] )" },
    info[optix_get_sphere_data]                              = { OptixABI::ABI_58,   OptixABI::ABI_LWRRENT, "_optix_get_sphere_data",                              "optixGetSphereData",                           "void optixGetSphereData( OptixTraversableHandle, unsigned int, unsigned int, float, float4[1] )" },
    info[optix_get_gas_traversable_handle]                   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_gas_traversable_handle",                   "optixGetGASTraversableHandle",                 "OptixTraversableHandle optixGetGASTraversableHandle()" },
    info[optix_get_gas_motion_time_begin]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_gas_motion_time_begin",                    "optixGetGASMotionTimeBegin",                   "float optixGetGASMotionTimeBegin( OptixTraversableHandle )" },
    info[optix_get_gas_motion_time_end]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_gas_motion_time_end",                      "optixGetGASMotionTimeEnd",                     "float optixGetGASMotionTimeEnd( OptixTraversableHandle )" },
    info[optix_get_gas_motion_step_count]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_gas_motion_step_count",                    "optixGetGASMotionStepCount",                   "unsigned int optixGetGASMotionStepCount()" },
    info[optix_get_gas_ptr]                                  = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_gas_ptr",                                  "optixGetGASPtr",                               "const char* optixGetGASPtr( OptixTraversableHandle )" },
    info[optix_get_transform_list_size]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_transform_list_size",                      "optixGetTransformListSize",                    "unsigned int optixGetTransformListSize()" },
    info[optix_get_transform_list_handle]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_transform_list_handle",                    "optixGetTransformListHandle",                  "OptixTraversableHandle optixGetTransformListHandle( unsigned int )" },
    info[optix_get_transform_type_from_handle]               = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_transform_type_from_handle",               "optixGetTransformTypeFromHandle",              "OptixTransformType optixGetTransformTypeFromHandle( OptixTraversableHandle )" },
    info[optix_get_static_transform_from_handle]             = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_static_transform_from_handle",             "optixGetStaticTransformFromHandle",            "const OptixStaticTransform* optixGetStaticTransformFromHandle( OptixTraversableHandle )" },
    info[optix_get_srt_motion_transform_from_handle]         = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_srt_motion_transform_from_handle",         "optixGetSRTMotionTransformFromHandle",         "const OptixSRTMotionTransform* optixGetSRTMotionTransformFromHandle( OptixTraversableHandle )" },
    info[optix_get_matrix_motion_transform_from_handle]      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_matrix_motion_transform_from_handle",      "optixGetMatrixMotionTransformFromHandle",      "const OptixMatrixMotionTransform* optixGetMatrixMotionTransformFromHandle( OptixTraversableHandle )" },
    info[optix_get_instance_id_from_handle]                  = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_instance_id_from_handle",                  "optixGetInstanceIdFromHandle",                 "unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle )" },
    info[optix_get_instance_child_from_handle]               = { OptixABI::ABI_42,   OptixABI::ABI_LWRRENT, "_optix_get_instance_child_from_handle",               "optixGetInstanceChildFromHandle",              "OptixTraversableHandle optixGetInstanceChildFromHandle( OptixTraversableHandle )" },
    info[optix_get_instance_transform_from_handle]           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_instance_transform_from_handle",           "optixGetInstanceTransformFromHandle",          "const float4* optixGetInstanceTransformFromHandle( OptixTraversableHandle )" },
    info[optix_get_instance_ilwerse_transform_from_handle]   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_instance_ilwerse_transform_from_handle",   "optixGetInstanceIlwerseTransformFromHandle",   "const float4* optixGetInstanceIlwerseTransformFromHandle( OptixTraversableHandle )" },
    info[optix_report_intersection_0]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_0",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int )" },
    info[optix_report_intersection_1]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_1",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int )" },
    info[optix_report_intersection_2]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_2",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_3]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_3",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_4]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_4",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_5]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_5",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_6]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_6",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_7]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_7",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_report_intersection_8]                        = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_report_intersection_8",                        "optixReportIntersection",                      "bool optixReportIntersection( float, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_get_attribute_0]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_0",                              "optixGetAttribute_0",                          "unsigned int optixGetAttribute_0()" },
    info[optix_get_attribute_1]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_1",                              "optixGetAttribute_1",                          "unsigned int optixGetAttribute_1()" },
    info[optix_get_attribute_2]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_2",                              "optixGetAttribute_2",                          "unsigned int optixGetAttribute_2()" },
    info[optix_get_attribute_3]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_3",                              "optixGetAttribute_3",                          "unsigned int optixGetAttribute_3()" },
    info[optix_get_attribute_4]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_4",                              "optixGetAttribute_4",                          "unsigned int optixGetAttribute_4()" },
    info[optix_get_attribute_5]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_5",                              "optixGetAttribute_5",                          "unsigned int optixGetAttribute_5()" },
    info[optix_get_attribute_6]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_6",                              "optixGetAttribute_6",                          "unsigned int optixGetAttribute_6()" },
    info[optix_get_attribute_7]                              = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_attribute_7",                              "optixGetAttribute_7",                          "unsigned int optixGetAttribute_7()" },
    info[optix_terminate_ray]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_terminate_ray",                                "optixTerminateRay",                            "void optixTerminateRay()" },
    info[optix_ignore_intersection]                          = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_ignore_intersection",                          "optixIgnoreIntersection",                      "void optixIgnoreIntersection()" },
    info[optix_read_primitive_idx]                           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_read_primitive_idx",                           "optixGetPrimitiveIndex",                       "unsigned int optixGetPrimitiveIndex()" },
    // This was added in ABI version 31, but min version is intentionally set to ABI_MIN because otherwise the lwrve intersector program would fail to compile.
    info[optix_read_prim_va]                                 = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_read_prim_va",                                 "optixGetPrimVA",                               "unsigned long long optixGetPrimVA()" },
    info[optix_read_key_time]                                = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_read_key_time",                                "optixGetKeyTime",                              "float optixGetKeyTime()" },
    info[optix_read_sbt_gas_idx]                             = { OptixABI::ABI_22,   OptixABI::ABI_LWRRENT, "_optix_read_sbt_gas_idx",                             "optixGetSbtGASIndex",                          "unsigned int optixGetSbtGASIndex()" },
    info[optix_read_instance_id]                             = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_read_instance_id",                             "optixGetInstanceId",                           "unsigned int optixGetInstanceId()" },
    info[optix_read_instance_idx]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_read_instance_idx",                            "optixGetInstanceIndex",                        "unsigned int optixGetInstanceIndex()" },
    info[optix_get_hit_kind]                                 = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_hit_kind",                                 "optixGetHitKind",                              "unsigned int optixGetHitKind()" },
    info[optix_get_primitive_type_from_hit_kind]             = { OptixABI::ABI_27,   OptixABI::ABI_LWRRENT, "_optix_get_primitive_type_from_hit_kind",             "optixGetPrimitiveType",                        "OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind)" },
    info[optix_is_hitkind_backface]                          = { OptixABI::ABI_27,   OptixABI::ABI_LWRRENT, "_optix_get_backface_from_hit_kind",                   "optixIsBackFaceHit",                           "bool optixIsBackFaceHit(unsigned int hitKind)" },
    info[optix_get_triangle_barycentrics]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_triangle_barycentrics",                    "optixGetTriangleBarycentrics",                 "float2 optixGetTriangleBarycentrics()" },
    info[optix_get_launch_index_x]                           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_index_x",                           "optixGetLaunchIndex",                          "uint3 optixGetLaunchIndex()" },
    info[optix_get_launch_index_y]                           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_index_y",                           "optixGetLaunchIndex",                          "uint3 optixGetLaunchIndex()" },
    info[optix_get_launch_index_z]                           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_index_z",                           "optixGetLaunchIndex",                          "uint3 optixGetLaunchIndex()" },
    info[optix_get_launch_dimension_x]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_dimension_x",                       "optixGetLaunchDimensions",                     "uint3 optixGetLaunchDimensions()" },
    info[optix_get_launch_dimension_y]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_dimension_y",                       "optixGetLaunchDimensions",                     "uint3 optixGetLaunchDimensions()" },
    info[optix_get_launch_dimension_z]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_launch_dimension_z",                       "optixGetLaunchDimensions",                     "uint3 optixGetLaunchDimensions()" },
    info[optix_get_sbt_data_ptr_64]                          = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_sbt_data_ptr_64",                          "optixGetSbtDataPointer",                       "LWdeviceptr optixGetSbtDataPointer()" },
    info[optix_throw_exception_0]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_0",                           "optixThrowException",                          "void optixThrowException( int )" },
    info[optix_throw_exception_1]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_1",                           "optixThrowException",                          "void optixThrowException( int, unsigned int )" },
    info[optix_throw_exception_2]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_2",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int )" },
    info[optix_throw_exception_3]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_3",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int )" },
    info[optix_throw_exception_4]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_4",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_throw_exception_5]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_5",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_throw_exception_6]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_6",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_throw_exception_7]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_7",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_throw_exception_8]                            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_throw_exception_8",                           "optixThrowException",                          "void optixThrowException( int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int )" },
    info[optix_get_exception_code]                           = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_code",                           "optixGetExceptionCode",                        "int optixGetExceptionCode()" },
    info[optix_get_exception_detail_0]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_0",                      "optixGetExceptionDetail_0", "unsigned int optixGetExceptionDetail_0()" },
    info[optix_get_exception_detail_1]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_1",                      "optixGetExceptionDetail_1", "unsigned int optixGetExceptionDetail_1()" },
    info[optix_get_exception_detail_2]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_2",                      "optixGetExceptionDetail_2", "unsigned int optixGetExceptionDetail_2()" },
    info[optix_get_exception_detail_3]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_3",                      "optixGetExceptionDetail_3", "unsigned int optixGetExceptionDetail_3()" },
    info[optix_get_exception_detail_4]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_4",                      "optixGetExceptionDetail_4", "unsigned int optixGetExceptionDetail_4()" },
    info[optix_get_exception_detail_5]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_5",                      "optixGetExceptionDetail_5", "unsigned int optixGetExceptionDetail_5()" },
    info[optix_get_exception_detail_6]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_6",                      "optixGetExceptionDetail_6", "unsigned int optixGetExceptionDetail_6()" },
    info[optix_get_exception_detail_7]                       = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_detail_7",                      "optixGetExceptionDetail_7", "unsigned int optixGetExceptionDetail_7()" },
    info[optix_get_exception_ilwalid_traversable]            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_ilwalid_traversable",            "optixGetExceptionIlwalidTraversable",          "OptixTraversableHandle optixGetExceptionIlwalidTraversable()" },
    info[optix_get_exception_ilwalid_sbt_offset]             = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_ilwalid_sbt_offset",             "optixGetExceptionIlwalidSbtOffset",            "int optixGetExceptionIlwalidSbtOffset()" },
    // This was added in ABI version 31, but min version is intentionally set to ABI_MIN because otherwise the default exception program would fail to compile.
    info[optix_get_exception_ilwalid_ray]                    = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_ilwalid_ray",                    "optixGetExceptionIlwalidRay",                  "OptixIlwalidRayExceptionDetails optixGetExceptionIlwalidRay()" },
    info[optix_get_exception_parameter_mismatch]             = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_parameter_mismatch",            "optixGetExceptionParameterMismatch",           "OptixParameterMismatchExceptionDetails optixGetExceptionParameterMismatch()" },
    info[optix_get_exception_line_info]                      = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_get_exception_line_info",                     "optixGetExceptionLineInfo",                    "char* optixGetExceptionLineInfo()" },
    info[optix_call_direct_callable]                         = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_call_direct_callable",                         "optixDirectCall",                              "ReturnT optixDirectCall( unsigned int, ArgTypes... )" },
    info[optix_call_continuation_callable]                   = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_call_continuation_callable",                   "optixContinuationCall",                        "ReturnT optixContinuationCall( unsigned int, ArgTypes... )" };
    // These initial versions of the texture footprint intrinsics are no longer supported, so an invalid version range is specified.
    info[optix_tex_footprint_2d]                             = { OptixABI::ABI_43,   OptixABI::ABI_42, "_optix_tex_footprint_2d",                             "optixTexFootprint2D",                          "uint4 optixTexFootprint2D( unsigned long, float, float, unsigned int, unsigned int, unsigned int* )" };
    info[optix_tex_footprint_2d_grad]                        = { OptixABI::ABI_43,   OptixABI::ABI_42, "_optix_tex_footprint_2d_grad",                        "optixTexFootprint2DGrad",                      "uint4 optixTexFootprint2DGrad( unsigned long, float, float, float, float, float, float, unsigned int, unsigned int, unsigned int* )" };
    info[optix_tex_footprint_2d_lod]                         = { OptixABI::ABI_43,   OptixABI::ABI_42, "_optix_tex_footprint_2d_lod",                         "optixTexFootprint2DLod",                       "uint4 optixTexFootprint2DLod( unsigned long, float, float, float, unsigned int, unsigned int, unsigned int* )" };
    info[optix_tex_footprint_2d_v2]                          = { OptixABI::ABI_47,   OptixABI::ABI_LWRRENT, "_optix_tex_footprint_2d_v2",                             "optixTexFootprint2D",                          "uint4 optixTexFootprint2D    ( unsigned long long tex, unsigned int texInfo, float x, float y, unsigned long long singleMipLevelPtr, unsigned long long resultPtr )" };
    info[optix_tex_footprint_2d_grad_v2]                     = { OptixABI::ABI_47,   OptixABI::ABI_LWRRENT, "_optix_tex_footprint_2d_grad_v2",                        "optixTexFootprint2DGrad",                      "uint4 optixTexFootprint2DGrad( unsigned long long tex, unsigned int texInfo, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, unsigned int coarse, unsigned long long singleMipLevelPtr, unsigned long long resultPtr)" };
    info[optix_tex_footprint_2d_lod_v2]                      = { OptixABI::ABI_47,   OptixABI::ABI_LWRRENT, "_optix_tex_footprint_2d_lod_v2",                         "optixTexFootprint2DLod",                       "uint4 optixTexFootprint2DLod ( unsigned long long tex, unsigned int texInfo, float x, float y, float lod, unsigned int coarse, unsigned long long singleMipLevelPtr, unsigned long long resultPtr )" };
    // This was added in ABI version 56 (ABI_56), but min version is intentionally set to ABI_MIN because otherwise the lwrve intersector program would fail to compile.
    info[optix_private_get_compile_time_constant]            = { OptixABI::ABI_MIN,  OptixABI::ABI_LWRRENT, "_optix_private_get_compile_time_constant",            "optixPrivateGetCompileTimeConstant",           "unsigned int optixPrivateGetCompileTimeConstant ( unsigned int index )" };
    // clang-format on

    return info;
}

const IntrinsicInfos& InitialCompilationUnit::intrinsicsInfo()
{
    static IntrinsicInfos info = initializeIntrinsicsInfo();
    return info;
}

static const std::string& intrinsicName( IntrinsicIndex index )
{
    return InitialCompilationUnit::intrinsicsInfo()[index].intrinsicName;
}
static const std::string& apiName( IntrinsicIndex index )
{
    return InitialCompilationUnit::intrinsicsInfo()[index].apiName;
}
static const std::string& signature( IntrinsicIndex index )
{
    return InitialCompilationUnit::intrinsicsInfo()[index].signature;
}
static OptixABI minimumAbiVersion( IntrinsicIndex index )
{
    return InitialCompilationUnit::intrinsicsInfo()[index].minimumAbiVersion;
}
static OptixABI maximumAbiVersion( IntrinsicIndex index )
{
    return InitialCompilationUnit::intrinsicsInfo()[index].maximumAbiVersion;
}

static int getCallableParamRegCount( int defaultNumCallablePramRegCount, bool noinlineEnabled, bool enableCallableParamCheck, int paramCheckExceptionRegisters )
{
    int maxParamRegisters = defaultNumCallablePramRegCount;
    int paramRegisters    = noinlineEnabled ? 18 : 4;
    if( k_maxParamRegCount.isSet() )
        paramRegisters = k_maxParamRegCount.get();
    if( paramRegisters == -1 || paramRegisters > maxParamRegisters )
        paramRegisters = maxParamRegisters;
    if( enableCallableParamCheck )
    {
        // If the parameter check for callable programs is enabled, there will be compileParams.paramCheckExceptionRegisters
        // i32 parameters added to each callable program. We need to make sure that none of those ever gets spilled, so we need
        // at least (compileParams.paramCheckExceptionRegisters + 1) parameter registers.
        // Otherwise the following may happen (assuming compileParams.paramCheckExceptionRegisters == 4):
        //  - Callable takes one parameter, with the additional 4 we have 5 which will (by default)
        //    cause spilling in the callee. Especially the last of the additional exception parameters
        //    will be spilled, so its value will try to be loaded from the spill address.
        //  - Assume the callable is called without any parameters. With the additional 4 parameters that
        //    are added to the call, we are still under the spilling limit, so no parameters will
        //    be spilled and no spill address will passed.
        //  - The exception will be triggered and evil things happen when the callable now tries to load
        //    the fourth exception parameter.
        paramRegisters = std::max( paramRegisters, paramCheckExceptionRegisters + 1 );
    }
    return paramRegisters;
}

CompilePayloadType::CompilePayloadType( const OptixPayloadType& type )
{
    semantics.resize( type.numPayloadValues );
    memcpy( semantics.data(), type.payloadSemantics, sizeof(unsigned int) * type.numPayloadValues );

    // construct string uniquely identifying the payload type definition
    std::stringstream ss;
    int      bits = 0;
    uint64_t encoding = 0;
    for( const auto& semantic : semantics )
    {
        // tightly pack the semantics to keep the identifier short-ish
        if( bits + 10 >= 64 )
        {
            ss << std::hex << "0x" << encoding;
            bits = 0;
        }
        encoding |= (semantic << bits);
        bits += 10;
    }
    // add the size to guarentee uniqueness when a payload is padded with values with 0(zero) semantics.
    ss << std::hex << semantics.size() << "x" << encoding;
    mangledName = ss.str();
}

// Check if OptixPayloadSemantics are consistent
static OptixResult validatePayloadSemantics( unsigned int  optixPayloadSemantics,
                                             unsigned int  payloadTypeIdx,
                                             unsigned int  payloadValueIdx,
                                             unsigned int  usesPrimitiveTypeFlags,
                                             ErrorDetails& errDetails )
{
    // pipeline supports custom primitives (triangles and lwrves don't touch payload)
    const bool hasIs = ( usesPrimitiveTypeFlags & OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM ) == OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM;

    const bool traceIn  = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE ) != 0;
    const bool traceOut = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ  ) != 0;
    const bool isWrite  = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_IS_WRITE ) != 0 && hasIs;
    const bool ahWrite  = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_AH_WRITE ) != 0;
    const bool msWrite  = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_MS_WRITE ) != 0;
    const bool chWrite  = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_CH_WRITE ) != 0;
    const bool isRead   = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_IS_READ ) != 0 && hasIs;
    const bool ahRead   = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_AH_READ ) != 0;
    const bool msRead   = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_MS_READ ) != 0;
    const bool chRead   = ( optixPayloadSemantics & OPTIX_PAYLOAD_SEMANTICS_CH_READ ) != 0;

    const bool writtenInOrBeforeTraversal = traceIn || isWrite || ahWrite;
    const bool readAfterTraversal         = traceOut || chRead || msRead;
    const bool read                       = traceOut || isRead || ahRead || chRead || msRead;
    const bool write                      = traceIn || isWrite || ahWrite || chWrite || msWrite;

    if( !write )
    {
        // never written
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Invalid payload semantics for payload value " + std::to_string( payloadValueIdx )
                                          + " in payload type " + std::to_string( payloadTypeIdx )
                                          + ". Payload value never written." );
    }

    if( !read )
    {
        // never read
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Invalid payload semantics for payload value " + std::to_string( payloadValueIdx )
                                          + " in payload type " + std::to_string( payloadTypeIdx )
                                          + ". Payload value never used." );
    }

    if( ( isRead || ahRead || chRead || msRead ) && !( traceIn || isWrite || ahWrite ) )
    {
        // use before write
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Invalid payload semantics for payload value " + std::to_string( payloadValueIdx )
                                          + " in payload type " + std::to_string( payloadTypeIdx )
                                          + ". Payload use before write." );
    }

    if( ( chWrite || msWrite ) && !traceOut )
    {
        // write without use
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Invalid payload semantics for payload value " + std::to_string( payloadValueIdx )
                                          + " in payload type " + std::to_string( payloadTypeIdx )
                                          + ". Payload write without use." );
    }

    return OPTIX_SUCCESS;
}

OptixResult setInternalCompileOptions( InternalCompileParameters&         compileParams,
                                       const OptixModuleCompileOptions*   moduleCompileOptions,
                                       const OptixPipelineCompileOptions* pipelineCompileOptions,
                                       const DeviceContext*               context,
                                       const bool                         isBuiltinModule,
                                       const bool                         enableLwstomPrimitiveVA,
                                       const bool                         useD2IR,
                                       const std::vector<unsigned int>&   privateCompileTimeConstants,
                                       ErrorDetails&                      errDetails )
{
    compileParams                               = {};
    compileParams.maxSmVersion                  = context->getComputeCapability();
    compileParams.abiVersion                    = context->getAbiVersion();
    compileParams.sbtHeaderSize                 = context->getSbtHeaderSize();
    compileParams.noinlineEnabled               = context->isNoInlineEnabled();
    compileParams.validationModeDebugExceptions = context->hasValidationModeDebugExceptions();

    //
    if( moduleCompileOptions )
    {
        compileParams.maxRegisterCount = moduleCompileOptions->maxRegisterCount == 0 ? 128 : moduleCompileOptions->maxRegisterCount;
        if( k_maxRegCount.isSet() )
            compileParams.maxRegisterCount = k_maxRegCount.get();

        compileParams.optLevel   = moduleCompileOptions->optLevel;
        compileParams.debugLevel = moduleCompileOptions->debugLevel;
        if( !isBuiltinModule )
        {
            if( k_forceDebugMode.get() )
            {
                compileParams.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                compileParams.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            }
            else
            {
                if( k_overwriteDbgLevel.get() != -1 )
                {
                    switch( k_overwriteDbgLevel.get() )
                    {
                        case 0:
                            compileParams.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
                            break;
                        case 1:
                            compileParams.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
                            break;
                        case 2:
                            compileParams.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
                            break;
                        case 3:
                            compileParams.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
                            break;
                        default:
                            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                          "Invalid value for knob o7.overwriteDebugLevel. Valid values "
                                                          "are in the range of [ -1; 3 ]" );
                    }
                }
                if( k_overwriteOptLevel.get() != -1 )
                {
                    switch( k_overwriteOptLevel.get() )
                    {
                        case 0:
                            compileParams.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
                            break;
                        case 1:
                            compileParams.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
                            break;
                        case 2:
                            compileParams.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
                            break;
                        case 3:
                            compileParams.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
                            break;
                        default:
                            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                          "Invalid value for knob o7.overwriteOptLevel. Valid values "
                                                          "are in the range of [ -1; 3 ]" );
                    }
                }
            }
        }

        if( k_enableSpecializations.get() )
        {
            compileParams.specializedLaunchParam.resize( moduleCompileOptions->numBoundValues );
            for( unsigned int i = 0; i < moduleCompileOptions->numBoundValues; ++i )
            {
                const OptixModuleCompileBoundValueEntry& modEntry = moduleCompileOptions->boundValues[i];
                CompileBoundValueEntry&                  entry    = compileParams.specializedLaunchParam[i];
                entry.offset                                      = modEntry.pipelineParamOffsetInBytes;
                entry.value.resize( modEntry.sizeInBytes );
                memcpy( entry.value.data(), modEntry.boundValuePtr, entry.value.size() );
                if( modEntry.annotation )
                    entry.annotation = modEntry.annotation;
                else
                    entry.annotation = "No annotation";
            }
            std::sort( compileParams.specializedLaunchParam.begin(), compileParams.specializedLaunchParam.end(),
                       []( const CompileBoundValueEntry& a, const CompileBoundValueEntry& b ) {
                           return a.offset < b.offset;
                       } );
        }

        if( moduleCompileOptions->numPayloadTypes != 0 )
        {
            if( moduleCompileOptions->payloadTypes == 0 )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "OptixModuleCompileOptions::payloadTypes must not be zero when OptixModuleCompileOptions::numPayloadTypes is non-zero" );

            if( pipelineCompileOptions->numPayloadValues != 0 )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                    "OptixPipelineCompileOptions::numPayloadValues must be zero when "
                    "OptixModuleCompileOptions::numPayloadTypes is not zero" );

            if( moduleCompileOptions->numPayloadTypes > OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT )
            {
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                    "OptixModuleCompileOptions::numPayloadTypes must be less than or equal to "
                    + std::to_string( OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT ) + ": "
                    + std::to_string( moduleCompileOptions->numPayloadTypes ) );
            }

            // deep-copy payload types
            OptixResultOneShot result;
            for( unsigned int i = 0; i < moduleCompileOptions->numPayloadTypes; ++i )
            {
                result += validatePayloadType( moduleCompileOptions->payloadTypes[i], errDetails );
                if( result != OPTIX_SUCCESS )
                    return result;

                compileParams.payloadTypes.push_back( CompilePayloadType( moduleCompileOptions->payloadTypes[i] ) );
            }

            if( result != OPTIX_SUCCESS )
                return result;
        }
        else
        {
            if( pipelineCompileOptions->numPayloadValues > OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                    "OptixPipelineCompileOptions::numPayloadValues exceeds "
                    + std::to_string( OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT ) );

            // generate single default payload type
            const unsigned int defaultOptixSemantics = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE
                | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE
                | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE;
            compileParams.payloadTypes.resize( 1 );
            compileParams.payloadTypes[0].semantics.resize( pipelineCompileOptions->numPayloadValues, defaultOptixSemantics );
        }
    }
    else
    {
        // Note that these options get used for the rtcore nop programs. Be aware when changing these.
        compileParams.maxRegisterCount = 128;
        compileParams.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
        compileParams.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    }

    compileParams.usesMotionBlur = pipelineCompileOptions->usesMotionBlur == 1 ? true : false;

    int numAttributeValues = pipelineCompileOptions->numAttributeValues;
    if( numAttributeValues > 8 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "OptixPipelineCompileOptions::numAttributeValues must be less than or equal to "
                                      "8: "
                                          + std::to_string( numAttributeValues ) );
    compileParams.numAttributeValues = std::max( 2, numAttributeValues );

    compileParams.exceptionFlags = getExceptionFlags( pipelineCompileOptions, context->hasValidationModeDebugExceptions() );
    compileParams.traversableGraphFlags = pipelineCompileOptions->traversableGraphFlags;

    compileParams.usesPrimitiveTypeFlags = 0;

    // Field was introduced with ABI 27.
    if( context->getAbiVersion() >= OptixABI::ABI_27 )
        compileParams.usesPrimitiveTypeFlags = pipelineCompileOptions->usesPrimitiveTypeFlags;

    // By default support triangles and custom primitives.
    if( compileParams.usesPrimitiveTypeFlags == 0 )
        compileParams.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    // Validate payload semantics
    for( unsigned int i = 0; i < compileParams.payloadTypes.size(); ++i )
    {
        OptixResultOneShot result;
        for( unsigned int j = 0; j < compileParams.payloadTypes[i].semantics.size(); ++j )
        {
            result += validatePayloadSemantics( compileParams.payloadTypes[i].semantics[j],
                                                i, j, compileParams.usesPrimitiveTypeFlags, errDetails );
        }

        if( result != OPTIX_SUCCESS )
            return result;
    }

    if( pipelineCompileOptions->pipelineLaunchParamsVariableName )
        compileParams.pipelineLaunchParamsVariableName = pipelineCompileOptions->pipelineLaunchParamsVariableName;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    compileParams.allowVisibilityMaps = pipelineCompileOptions->allowVisibilityMaps;
    compileParams.allowDisplacedMicromeshes = pipelineCompileOptions->allowDisplacedMicromeshes;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    compileParams.enableCallableParamCheck     = compileParams.exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG;
    compileParams.paramCheckExceptionRegisters = 4;

    // Make sure the values in compileParams have been set before using them here
    compileParams.callableParamRegCount =
        getCallableParamRegCount( context->getCallableParamRegCount(), compileParams.noinlineEnabled,
                                  compileParams.enableCallableParamCheck, compileParams.paramCheckExceptionRegisters );

    compileParams.inlineCallLimitHigh           = k_inlineUpperCallLimit.get();
    compileParams.inlineCallLimitLow            = k_inlineLowerCallLimit.get();
    compileParams.inlineInstructionLimit        = k_inlineInstructionLimit.get();
    compileParams.removeUnusedNoinlineFunctions = k_removeUnusedNoinlineFunctions.get();
    compileParams.forceInlineSet                = k_forceInlineSet.get();
    compileParams.disableNoinlineFunc           = k_disableNoinlineFunc.get();
    compileParams.allowIndirectFunctionCalls    = k_allowIndirectFunctionCalls.get();
    compileParams.disableActiveMaskCheck        = k_disableActiveMaskCheck.get();
    compileParams.enableLwstomABIProcessing     = k_enableLwstomABIProcessing.get();
    compileParams.numAdditionalABIScratchRegs   = k_numAdditionalABIScratchRegs.get();
    compileParams.enableCoroutines              = k_enableCoroutines.get();
    compileParams.enableProfiling               = !k_enableProfiling.get().empty();
    compileParams.useSoftwareTextureFootprint   = k_useSoftwareTextureFootprint.get();
    compileParams.splitModuleMinBinSize         = context->getSplitModuleMinBinSize();
    compileParams.serializeModuleId             = k_serializeModuleId.get();

    compileParams.useD2IR = useD2IR;
    if( useD2IR && ( compileParams.debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_MODERATE
                     || compileParams.debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_FULL ) )
    {
        lwarn
            << "Disabling D2IR due to OPTIX_COMPILE_DEBUG_LEVEL_MODERATE or OPTIX_COMPILE_DEBUG_LEVEL_FULL\n";
        compileParams.useD2IR = false;
    }

    compileParams.enableLWPTXFallback      = !isBuiltinModule && context->isLWPTXFallbackEnabled();
    compileParams.addBuiltinPrimitiveCheck = !isBuiltinModule && ( compileParams.exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG );
    compileParams.isBuiltinModule          = isBuiltinModule;
    compileParams.enableLwstomPrimitiveVA  = enableLwstomPrimitiveVA;
    compileParams.elideUserThrow           = compileParams.abiVersion >= OptixABI::ABI_36;
    compileParams.hideModule               = isBuiltinModule && k_hideInternalModules.get();

    compileParams.privateCompileTimeConstants = privateCompileTimeConstants;

    // Make sure this is last as it reads a few of the values in compileParams
    ABIDecisionLogger decisionLogger;
    compileParams.abiVariant = getRtcAbiVariant( compileParams, context, &decisionLogger );
    decisionLogger.setAbiAndPrint( compileParams.abiVariant );

    return OPTIX_SUCCESS;
}

// This function should not read anything except values from compileParams. Nothing from
// DeviceContext or knobs. Otherwise the cache will not recognize the changes.
OptixResult setRtcCompileOptions( RtcCompileOptions&               compileOptions,
                                  const InternalCompileParameters& compileParams,
                                  ErrorDetails&                    errDetails )
{
    compileOptions.abiVariant                = compileParams.abiVariant;
    compileOptions.numPayloadRegisters       = ~0u; // indicates we are using payload type annotation
    compileOptions.numAttributeRegisters     = compileParams.numAttributeValues;
    compileOptions.numCallableParamRegisters = compileParams.callableParamRegCount;
    compileOptions.numMemoryAttributeScalars = 0;  // no memory attributes for now
    compileOptions.smVersion                 = compileParams.maxSmVersion;
    compileOptions.maxRegisterCount          = compileParams.maxRegisterCount;
    compileOptions.enableLwstomPrimitiveVA   = compileParams.enableLwstomPrimitiveVA;
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    if( compileParams.allowVisibilityMaps )
        compileOptions.compileFlags |= RTC_COMPILE_FLAG_ALLOW_VISIBILITY_MAPS;
    if( compileParams.allowDisplacedMicromeshes )
        compileOptions.compileFlags |= RTC_COMPILE_FLAG_ALLOW_DISPLACED_MICROMESHES;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    switch( compileParams.optLevel )
    {
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:
            compileOptions.optLevel = 0;
            break;
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1:
            compileOptions.optLevel = 1;
            break;
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2:
            compileOptions.optLevel = 2;
            break;
        case OPTIX_COMPILE_OPTIMIZATION_DEFAULT:
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                          "Internal error: OPTIX_COMPILE_OPTIMIZATION_DEFAULT should not be seen at "
                                          "this point\n" );
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3:
            compileOptions.optLevel = 3;
            break;
    }

    switch( compileParams.debugLevel )
    {
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:
            compileOptions.debugLevel = 0;
            break;
        case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                          "OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT should not be seen at this point\n" );
        case OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:
        case OPTIX_COMPILE_DEBUG_LEVEL_MODERATE:
            compileOptions.debugLevel = 1;
            break;
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:
            compileOptions.debugLevel = 2;
            break;
    }

    if( compileParams.enableProfiling )
        compileOptions.enabledTools = RTC_TOOLS_FLAG_PROFILING | RTC_TOOLS_FLAG_DETAILED_SHADER_INFO;
    else
        compileOptions.enabledTools = RTC_TOOLS_FLAG_NONE;

    {
        // compileOptions.exceptionFlags
        const int exceptionFlags = compileParams.exceptionFlags;

        compileOptions.exceptionFlags = RTC_EXCEPTION_FLAG_ILWOKE_EX;
        if( exceptionFlags & OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW )
            compileOptions.exceptionFlags |= RTC_EXCEPTION_FLAG_STACK_OVERFLOW;
        if( exceptionFlags & OPTIX_EXCEPTION_FLAG_TRACE_DEPTH )
            compileOptions.exceptionFlags |= RTC_EXCEPTION_FLAG_TRACE_DEPTH;
        if( exceptionFlags & OPTIX_EXCEPTION_FLAG_USER )
            compileOptions.exceptionFlags |= RTC_EXCEPTION_FLAG_PRODUCT_SPECIFIC;

        // For the inbuilt optix side exceptions OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE
        // and OPTIX_EXCEPTION_CODE_ILWALID_RAY we need to enable product specific exceptions for rtcore
        // when debug exceptions are enabled because they use lw.rt.throw.
        // Note that lowerThrowException produces an error for uses of optixThrowException if
        // OPTIX_EXCEPTION_FLAG_USER is not enabled, so this does not break correctness.
        if( exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG )
            compileOptions.exceptionFlags |= RTC_EXCEPTION_FLAG_DEBUG | RTC_EXCEPTION_FLAG_PRODUCT_SPECIFIC;
    }

    compileOptions.dumpModuleLwbin         = false;
    compileOptions.printContinuationSpills = false;
    compileOptions.smemSpillPolicy         = RTC_SMEM_SPILLING_DISABLED;
    {
        compileOptions.traversableGraphFlags = 0;
        // The scene graph has no instances
        if( compileParams.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS )
        {
            compileOptions.traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_HAS_NO_TOP_LEVEL_ACCEL;
        }

        // The scene allows single level GAS traversal
        if( compileParams.traversableGraphFlags != OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING )
        {
            compileOptions.traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_BLAS;
        }

        // The scene graph may contain motion AS
        if( compileParams.usesMotionBlur )
        {
            compileOptions.traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_MOTION_ACCEL;
        }

        // The scene graph may contain transform traversables
        if( compileParams.traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY )
        {
            compileOptions.traversableGraphFlags |= RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_STATIC_TRANSFORM_TRAVERSABLE;

            // The scene graph may contain motion transform traversables
            if( compileParams.usesMotionBlur )
                compileOptions.traversableGraphFlags |= ( RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_MATRIX_MOTION_TRANSFORM_TRAVERSABLE
                                                          | RTC_TRAVERSABLE_GRAPH_FLAG_ALLOW_SRT_MOTION_TRANSFORM_TRAVERSABLE );
        }
    }
    compileOptions.numRgContinuationRegisters   = 0;
    compileOptions.useLWPTX                     = !compileParams.useD2IR;
    compileOptions.compileForFastLinking        = false;
    compileOptions.targetSharedMemoryBytesPerSM = -1;  // deprecated and ignored by rtcore
    compileOptions.hideModule                   = compileParams.hideModule;
    //compileOptions.preferredScheduler; // deprecated and ignored by rtcore
    compileOptions.enableCoroutines    = compileParams.enableCoroutines;
    compileOptions.warpsPerCtaOverride = 0;  // zero means default
    compileOptions.usesTraversables    = true;
    compileOptions.compileForLwda      = true;

    return OPTIX_SUCCESS;
}

int getExceptionFlags( const OptixPipelineCompileOptions* pipelineCompileOptions, bool enableAll )
{
    if( k_disableAllExceptions.get() )
        return OPTIX_EXCEPTION_FLAG_NONE;

    int exceptionFlags = pipelineCompileOptions->exceptionFlags;
    if( k_enableAllExceptions.get() || enableAll )
        exceptionFlags |= OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER
               | OPTIX_EXCEPTION_FLAG_DEBUG;
    if( !k_enableProfiling.get().empty() )
    {
        if( exceptionFlags & ( OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH ) )
        {
            lwarn << "Disabling user and debug exceptions to enable profiling\n";
        }
        exceptionFlags &= ~( OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH );
    }

    return exceptionFlags;
}

RtcAbiVariant getRtcAbiVariant( const InternalCompileParameters& compileParams, const DeviceContext* context, ABIDecisionLogger* decisionLog )
{
    const bool enableDebugExceptions = ( compileParams.exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG ) != 0;

    if( !k_overrideAbiVariant.isDefault() )
    {
        if( decisionLog )
            decisionLog->addDecision( "knobOverride" );
        if( context->hasTTU() && k_overrideAbiVariant.get() == "ttu_a" )
            return RTC_ABI_VARIANT_TTU_A;
        else if( context->hasTTU() && k_overrideAbiVariant.get() == "ttu_b" )
            return RTC_ABI_VARIANT_TTU_B;
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
        else if( k_overrideAbiVariant.get() == "ttu_d" )
            return RTC_ABI_VARIANT_TTU_D;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
        else if( k_overrideAbiVariant.get() == "utrav" )
            return RTC_ABI_VARIANT_UTRAV;
        else if( k_overrideAbiVariant.get() == "mttu" )
            return RTC_ABI_VARIANT_MTTU;
        else
        {
            lwarn << "Ignoring the unknown abi variant override '" << k_overrideAbiVariant.get() << "'.\n";
        }
    }

    bool hasTTU         = context->hasTTU();
    bool hasMTTU        = context->hasMotionTTU();
    bool triPrimsOnly   = ( compileParams.usesPrimitiveTypeFlags == OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE );
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    bool hasDisplacedMmTTU   = context->hasDisplacedMicromeshTTU();
    bool allowDisplacedMm    = compileParams.allowDisplacedMicromeshes;
    bool allowVisibilityMaps = compileParams.allowVisibilityMaps;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    bool singleIAS      = ( compileParams.traversableGraphFlags & OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ) != 0;
    bool singleGAS      = ( compileParams.traversableGraphFlags & OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ) != 0;
    bool usesMotionBlur = compileParams.usesMotionBlur;

#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    if( allowVisibilityMaps )
    {
        if( !hasTTU )
        {
            lerr << "Usage of visibility maps is not supported on non-RTX cards.\n";
            return RTC_ABI_VARIANT_ILWALID;
        }
    }
    if( allowDisplacedMm )
    {
        if( !hasTTU )
        {
            lerr << "Usage of displaced micromesh is not supported on non-RTX cards.\n";
            return RTC_ABI_VARIANT_ILWALID;
        }
        if( usesMotionBlur )
        {
            lerr << "Usage of displaced micromesh does not supported motion.\n";
            return RTC_ABI_VARIANT_ILWALID;
        }
    }
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    if( enableDebugExceptions )
    {
        if( decisionLog )
            decisionLog->addDecision( "debugExceptions" );
        return RTC_ABI_VARIANT_UTRAV;
    }
    else
    {
        if( decisionLog )
            decisionLog->addDecision( "noDebugExceptions" );
    }

#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    if( allowDisplacedMm )
    {
        if( decisionLog )
            decisionLog->addDecision( "displacedMicromesh" );

        if( hasDisplacedMmTTU && singleIAS && !singleGAS )
        {
            if( decisionLog )
                decisionLog->addDecision( "displacedMicromeshTTU" );
            return RTC_ABI_VARIANT_TTU_D;
        }
        else
        {
            if( decisionLog )
                decisionLog->addDecision( "noDisplacedMicromeshTTU" );
            // utrav, ttu_a, and mttu support the dmm sw fallback
        }
    }
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    if( usesMotionBlur )
    {
        if( decisionLog )
        {
            decisionLog->addDecision( "motionBlur" );
            if( hasMTTU )
                decisionLog->addDecision( RT_DSTRING( "hasMTTU" ) );
            else
                decisionLog->addDecision( RT_DSTRING( "noMTTU" ) );
        }
        return hasMTTU ? RTC_ABI_VARIANT_MTTU : RTC_ABI_VARIANT_UTRAV;
    }
    else
    {
        if( decisionLog )
            decisionLog->addDecision( "noMotionBlur" );
    }

    if( !hasTTU )
    {
        if( decisionLog )
            decisionLog->addDecision( RT_DSTRING( "noTTU" ) );
        return RTC_ABI_VARIANT_UTRAV;
    }
    else
    {
        if( decisionLog )
            decisionLog->addDecision( RT_DSTRING( "hasTTU" ) );
    }

    // At this point we know we have a TTU, No motion blur and no exceptions.
    if( decisionLog )
    {
        if( singleIAS )
            decisionLog->addDecision( "singleIAS" );
        if( singleGAS )
            decisionLog->addDecision( "singleGAS" );
        if( !singleIAS && !singleGAS )
            decisionLog->addDecision( "any" );
    }

    if( singleIAS && !singleGAS )
    {
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
        // this can only be reached if !hasDisplacedMmTTU
        // we need dmm emulation using item ranges, hence, cannot use TTU_B
        if( allowDisplacedMm )
            return RTC_ABI_VARIANT_TTU_A;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
        if( decisionLog )
        {
            if( triPrimsOnly )
                decisionLog->addDecision( "TriPrimsOnly" );
            else
                decisionLog->addDecision( "noTriPrimsOnly" );
        }
        return triPrimsOnly ? RTC_ABI_VARIANT_TTU_B : RTC_ABI_VARIANT_TTU_A;
    }

    if( singleIAS && singleGAS )
        return RTC_ABI_VARIANT_MTTU;

    if( decisionLog )
    {
        if( hasMTTU )
            decisionLog->addDecision( RT_DSTRING( "hasMTTU" ) );
        else
            decisionLog->addDecision( RT_DSTRING( "noMTTU" ) );
    }

    // Note: we could also enable mttu on Turing for multi-level traversal
    // ( !singleIAS && singleGAS ) || ( !singleIAS && !singleGAS )
    return hasMTTU ? RTC_ABI_VARIANT_MTTU : RTC_ABI_VARIANT_UTRAV;
}


std::string serializeLLVMModule( const llvm::Module* module )
{
    return std::string();
}

static bool isInternalFunction( const llvm::StringRef& functionName )
{
    llvm::Regex blackList(
        "^(_rt.+|_lw_optix_.+|_optix_.+|optix\\.ptx\\..+|optix\\.lwvm\\..+|llvm\\..+|lw\\.rt\\..+|vprintf|"
        "__assertfail|lwdaMalloc|lwdaFuncGetAttributes|lwdaDeviceGetAttribute|lwdaGetDevice|"
        "lwdaOclwpancyMaxActiveBlocksPerMultiprocessor|lwdaOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags)$" );
    return blackList.match( functionName );
}

std::string semanticTypeToString( SemanticType stype )
{
    switch( stype )
    {
        case ST_RAYGEN:
            return "RAYGEN";
        case ST_EXCEPTION:
            return "EXCEPTION";
        case ST_MISS:
            return "MISS";
        case ST_INTERSECTION:
            return "INTERSECTION";
        case ST_CLOSESTHIT:
            return "CLOSESTHIT";
        case ST_ANYHIT:
            return "ANYHIT";
        case ST_DIRECT_CALLABLE:
            return "DIRECT_CALLABLE";
        case ST_CONTINUATION_CALLABLE:
            return "CONTINUATION_CALLABLE";
        case ST_NOINLINE:
            return "NOINLINE";
        case ST_ILWALID:
            return "INVALID";
            // default case intentionally omitted
    }
    return "INVALID";
}

std::string semanticTypeToAbbreviationString( SemanticType stype )
{
    switch( stype )
    {
        case ST_RAYGEN:
            return "RG";
        case ST_EXCEPTION:
            return "EX";
        case ST_MISS:
            return "MS";
        case ST_INTERSECTION:
            return "IS";
        case ST_CLOSESTHIT:
            return "CH";
        case ST_ANYHIT:
            return "AH";
        case ST_DIRECT_CALLABLE:
            return "DC";
        case ST_CONTINUATION_CALLABLE:
            return "CC";
        case ST_NOINLINE:
            return "NI";
        case ST_ILWALID:
            return "INVALID";
            // default case intentionally omitted
    }
    return "INVALID";
}

optix_exp::SemanticType getSemanticTypeForFunctionName( const llvm::StringRef& functionName, bool noInlineEnabled, const std::string& disableNoinlineFunc )
{
    static std::vector<std::pair<const char*, optix_exp::SemanticType>> semanticTypeIdentifiers = {
        {"__raygen__", optix_exp::ST_RAYGEN},
        {"__miss__", optix_exp::ST_MISS},
        {"__closesthit__", optix_exp::ST_CLOSESTHIT},
        {"__anyhit__", optix_exp::ST_ANYHIT},
        {"__intersection__", optix_exp::ST_INTERSECTION},
        {"__exception__", optix_exp::ST_EXCEPTION},
        {"__direct_callable__", optix_exp::ST_DIRECT_CALLABLE},
        {"__continuation_callable__", optix_exp::ST_CONTINUATION_CALLABLE}};

    for( const std::pair<const char*, optix_exp::SemanticType>& identifier : semanticTypeIdentifiers )
    {
        if( functionName.startswith( identifier.first ) )
        {
            return identifier.second;
        }
    }

    if( noInlineEnabled )
    {
        if( !disableNoinlineFunc.empty() )
        {
            std::stringstream tokens( disableNoinlineFunc );
            std::string       token;
            while( std::getline( tokens, token, ',' ) )
            {
                if( functionName.find( token ) != std::string::npos )
                {
                    return optix_exp::ST_ILWALID;
                }
            }
        }

        if( !isInternalFunction( functionName ) )
        {
            return optix_exp::ST_NOINLINE;
        }
    }
    return optix_exp::ST_ILWALID;
}

static optix_exp::SemanticType getSemanticTypeForFunction( llvm::Function* f, bool noInlineEnabled, const std::string& disableNoinlineFunc )
{
    return getSemanticTypeForFunctionName( f->getName(), noInlineEnabled, disableNoinlineFunc );
}

static const char* getRtcAnnotationForSemanticType( optix_exp::SemanticType stype )
{
    switch( stype )
    {
        case ST_RAYGEN:
            return "raygen";
        case ST_EXCEPTION:
            return "exception";
        case ST_MISS:
            return "miss";
        case ST_INTERSECTION:
            return "intersection";
        case ST_CLOSESTHIT:
            return "closesthit";
        case ST_ANYHIT:
            return "anyhit";
        case ST_DIRECT_CALLABLE:
        case ST_NOINLINE:
            return "directcallable";
        case ST_CONTINUATION_CALLABLE:
            return "continuationcallable";
        case ST_ILWALID:
            return "INVALID";
            // default case intentionally omitted
    }
    return "INVALID";
}

static void getBasicBlockAndInstructionCount( llvm::Module* module, const std::string& name, size_t& basicBlockCount, size_t& instructionCount )
{
    basicBlockCount  = 0;
    instructionCount = 0;

    llvm::Function* F = module->getFunction( name );
    if( !F )
        return;

    for( llvm::Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB )
    {
        ++basicBlockCount;
        for( llvm::BasicBlock::iterator I = BB->begin(), IE = BB->end(); I != IE; ++I )
            ++instructionCount;
    }
}

static OptixResult getPayloadTypeAnnotation( llvm::Function* inFunction, CompilationUnit& module, optix_exp::SemanticType stype, unsigned int& payloadTypeID, ErrorDetails& errDetails )
{
    payloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT;
    if( stype == ST_MISS || stype == ST_CLOSESTHIT || stype == ST_ANYHIT || stype == ST_INTERSECTION )
    {
        llvm::Module* inModule  = inFunction->getParent();
        llvm::MDNode* types     = inFunction->getMetadata( "optix.payload.types" );

        if( types )
        {
            bool isConstant = corelib::getConstantValue( UseMdAsValue( module.llvmModule->getContext(), types->getOperand( 0 ) ), payloadTypeID );
            if( !isConstant )
                return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }

        // when the default payload type is specified on a function with a semantic types that require a payload type, all payload types are supported.
        if( payloadTypeID == OPTIX_PAYLOAD_TYPE_DEFAULT )
        {
            payloadTypeID = ( ( 1u << module.compileParams.payloadTypes.size() ) - 1 );
        }
    }

    return OPTIX_SUCCESS;
}

static std::string getSourceLocation( llvm::Module* module, const llvm::Instruction* instruction );
static std::string getSourceLocation( llvm::Instruction* instruction );

static OptixResult getPayloadTypeIndexFromID( unsigned int&      typeIndex,
                                              unsigned int       typeID,
                                              llvm::Instruction* I,
                                              CompilationUnit&   module,
                                              ErrorDetails&      errDetails )
{
    // only one type bit may be set
    if( typeID == 0 || ( typeID & ( typeID - 1 ) ) != 0 )
    {
        errDetails.m_compilerFeedback << "Error: Invalid payload type ID " << typeID << " (OptixPayloadTypeID)\n"
                                      << "\t" << getSourceLocation( I ) << "\n";
        return OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID;
    }

    // colwert ID to index by scanning for the set bit
    typeIndex = 0;
    while( ( typeID >>= 1 ) != 0 )
        typeIndex++;

    if( typeIndex >= module.compileParams.payloadTypes.size() )
    {
        errDetails.m_compilerFeedback << "Error: Invalid payload type ID " << typeID << ", payload index out of bounds (OptixPayloadTypeID)\n"
                                      << "\t" << getSourceLocation( I ) << "\n";
        return OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID;
    }

    return OPTIX_SUCCESS;
}

static OptixResult addSemanticTypeAnnotations( optix_exp::SubModule* subModule, CompilationUnit& module, ErrorDetails& errDetails )
{
    bool                            addedAnnotation = false;
    SubModule::NonEntryFunctionInfo nonEntryFunctionInfo;

    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );

    // Attach rtc payload types as metadata to the module
    std::vector<std::vector<unsigned int>> rtcPayloadTypeSemantics;

    // check if the optix and rtx semantic flags bitwise match
    static_assert( (int)RTC_PAYLOAD_SEMANTICS_CALLER_WRITE == (int)OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_CALLER_READ == (int)OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ
                && (int)RTC_PAYLOAD_SEMANTICS_CALLER_READWRITE == (int)OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_CH_READ == (int)OPTIX_PAYLOAD_SEMANTICS_CH_READ
                && (int)RTC_PAYLOAD_SEMANTICS_CH_WRITE == (int)OPTIX_PAYLOAD_SEMANTICS_CH_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_CH_READWRITE == (int)OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_MS_READ == (int)OPTIX_PAYLOAD_SEMANTICS_MS_READ
                && (int)RTC_PAYLOAD_SEMANTICS_MS_WRITE == (int)OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_MS_READWRITE == (int)OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_AH_READ == (int)OPTIX_PAYLOAD_SEMANTICS_AH_READ
                && (int)RTC_PAYLOAD_SEMANTICS_AH_WRITE == (int)OPTIX_PAYLOAD_SEMANTICS_AH_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_AH_READWRITE == (int)OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_IS_READ == (int)OPTIX_PAYLOAD_SEMANTICS_IS_READ
                && (int)RTC_PAYLOAD_SEMANTICS_IS_WRITE == (int)OPTIX_PAYLOAD_SEMANTICS_IS_WRITE
                && (int)RTC_PAYLOAD_SEMANTICS_IS_READWRITE == (int)OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,
            "OptixPayloadSemantics and RtcPayloadSemantics out of sync." );

    // Generate per payload Type value semantic
    for( auto& payloadType : module.compileParams.payloadTypes )
    {
        // Generate per payload value semantic
        std::vector<unsigned int> rtcPayloadValueSemantics;
        for( size_t i = 0; i < payloadType.semantics.size(); ++i )
            rtcPayloadValueSemantics.push_back( payloadType.semantics[i] );

        rtcPayloadTypeSemantics.emplace_back( rtcPayloadValueSemantics );
    }

    corelib::addLwvmRtPayloadMetadata( module.llvmModule, rtcPayloadTypeSemantics );

    for( const auto& F : corelib::getFunctions( module.llvmModule ) )
    {
        optix_exp::SemanticType stype =
            getSemanticTypeForFunction( F, module.compileParams.noinlineEnabled, module.compileParams.disableNoinlineFunc );
        if( F->isDeclaration() )
        {
            if( stype == optix_exp::ST_NOINLINE )
            {
                const char* semanticTypeForAnnotation = getRtcAnnotationForSemanticType( stype );
                corelib::addLwvmRtAnnotationMetadata( F, semanticTypeForAnnotation );
            }
            continue;
        }

        std::string name = F->getName();

        size_t basicBlockCount;
        size_t instructionCount;
        getBasicBlockAndInstructionCount( module.llvmModule, name, basicBlockCount, instructionCount );

        if( stype != optix_exp::ST_ILWALID )
        {
            // get mask with supported payload types
            unsigned int payloadTypeMask;
            if( OptixResult result = getPayloadTypeAnnotation( F, module, stype, payloadTypeMask, errDetails ) )
                return result;
            EntryFunctionSemantics entryFunctionSemantics( payloadTypeMask );

            addedAnnotation = true;

            const char* semanticTypeForAnnotation = getRtcAnnotationForSemanticType( stype );
            if( payloadTypeMask == 0 )
            {
                // case for shaders that don't support payload types (RG, EX, DC, CC)
                corelib::addLwvmRtAnnotationMetadata( F, semanticTypeForAnnotation );

                // set mangled name for the original function
                const std::string& mangledNamed = subModule->m_parentModule->getMangledName( name, OPTIX_PAYLOAD_TYPE_DEFAULT, stype );
                corelib::renameFunction( F, mangledNamed, /*changeDiLinkageNameOnly=*/true );
            }
            else
            {
                // case for shaders that require a specific payload type (IS, AH, MS, CH)

                // scan through all supported payload types.
                // duplicate entry function for each extra payload types beyond the first supported type.
                // the final module has a unique clone of the function for each supported payload type.
                OptixPayloadTypeID firstID = OPTIX_PAYLOAD_TYPE_DEFAULT;
                OptixPayloadTypeID lwrrentID = OPTIX_PAYLOAD_TYPE_ID_0;
                while( payloadTypeMask )
                {
                    if( payloadTypeMask & 0x1 )
                    {
                        unsigned int typeIndex = ~0u;
                        OptixResult  result =
                            getPayloadTypeIndexFromID( typeIndex, lwrrentID, F->getEntryBlock().getFirstNonPHI(), module, errDetails );
                        if( result != OPTIX_SUCCESS )
                            return result;

                        llvm::Function* typedFunction = F;
                        if( firstID == OPTIX_PAYLOAD_TYPE_DEFAULT )
                        {
                            // first supported payload type found
                            firstID = (OptixPayloadTypeID)lwrrentID;
                            corelib::addLwvmRtAnnotationMetadata( F, semanticTypeForAnnotation, typeIndex );

                            // set mangled name for the original function
                            const std::string& mangledNamed = subModule->m_parentModule->getMangledName( name, firstID, stype );
                            corelib::renameFunction( F, mangledNamed, /*changeDiLinkageNameOnly=*/true );
                        }
                        else
                        {
                            // duplicate the function for any supported secondary payload types
                            const std::string& mangledNamed = subModule->m_parentModule->getMangledName( name, lwrrentID, stype );

                            // the mangled name may not be unique if the payload type semantics are not unique.
                            // we can skip duplication if the duplicate already exists.
                            if( !F->getParent()->getFunction( mangledNamed ) )
                            {
                                llvm::ValueToValueMapTy VMap;
                                typedFunction = llvm::CloneFunction( F, VMap );
                                // The function is added to the module by llvm::CloneFunction
                                corelib::renameFunction( typedFunction, mangledNamed, /*changeDiLinkageNameOnly=*/true );
                                corelib::addLwvmRtAnnotationMetadata( typedFunction, semanticTypeForAnnotation, typeIndex );
                            }
                        }
                    };
                    lwrrentID = ( OptixPayloadTypeID )( (unsigned int)lwrrentID << 1 );
                    payloadTypeMask >>= 1;
                };
            }

            SubModule::EntryFunctionInfo entryFunctionInfo( module.traceCalls[name], module.ccCalls[name],
                                                            module.dcCalls[name], basicBlockCount, instructionCount );

            if( stype == optix_exp::ST_NOINLINE )
            {
                if( entryFunctionInfo.m_traceCallCount != 0 || entryFunctionInfo.m_directCallableCallCount != 0 )
                {
                    errDetails.m_compilerFeedback
                        << "Info: Function \"" << name << "\" has " << entryFunctionInfo.m_traceCallCount << " trace call(s), "
                        << entryFunctionInfo.m_directCallableCallCount << " direct callable call(s)\n";
                }
                ++nonEntryFunctionInfo.m_count;
                nonEntryFunctionInfo.m_basicBlockCount += basicBlockCount;
                nonEntryFunctionInfo.m_instructionCount += instructionCount;
            }
            else
            {
                // clang-format off
                errDetails.m_compilerFeedback
                    << "Info: Entry function \"" << name << "\" with semantic type "
                    << optix_exp::semanticTypeToString( stype ) << " has "
                    << entryFunctionInfo.m_traceCallCount << " trace call(s), "
                    << entryFunctionInfo.m_continuationCallableCallCount << " continuation callable call(s), "
                    << entryFunctionInfo.m_directCallableCallCount << " direct callable call(s), "
                    << entryFunctionInfo.m_basicBlockCount << " basic block(s), "
                    << entryFunctionInfo.m_instructionCount << " instruction(s)\n";
                // clang-format on
            }
            subModule->registerEntryFunction( name, std::move( entryFunctionInfo ), entryFunctionSemantics, stype );
        }
        else
        {
            ++nonEntryFunctionInfo.m_count;
            nonEntryFunctionInfo.m_basicBlockCount += basicBlockCount;
            nonEntryFunctionInfo.m_instructionCount += instructionCount;
        }
    }

    // clang-format off
    errDetails.m_compilerFeedback
        << "Info: " << nonEntryFunctionInfo.m_count << " non-entry function(s) have "
        << nonEntryFunctionInfo.m_basicBlockCount << " basic block(s), "
        << nonEntryFunctionInfo.m_instructionCount << " instruction(s)\n";
    // clang-format on

    subModule->setNonEntryFunctionInfo( std::move( nonEntryFunctionInfo ) );

    if( !addedAnnotation )
    {
        // make sure there is an lwvm.rt.annotations node, to signal to rtcore that we know there aren't any entry points.
        llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
        llvm::MDNode*      mdNode      = llvm::MDNode::get( llvmContext, {llvm::MDString::get( llvmContext, "empty" )} );
        llvm::NamedMDNode* annotations = module.llvmModule->getOrInsertNamedMetadata( "lwvm.rt.annotations" );
        annotations->addOperand( mdNode );
    }

    return OPTIX_SUCCESS;
}

static OptixResult validateInputPtx( InitialCompilationUnit& module, ErrorDetails& errDetails );
static OptixResult specializeLaunchParam( CompilationUnit& module, std::set<llvm::Function*>& specializedFunctions, ErrorDetails& errDetails );
static OptixResult lowerPrivateCompileTimeConstant( CompilationUnit&           module,
                                                    std::set<llvm::Function*>& specializedFunctions,
                                                    ErrorDetails&              errDetails );
static OptixResult optimizeSpecializedFunctions( CompilationUnit&                 module,
                                                 const std::set<llvm::Function*>& specializedFunctions,
                                                 ErrorDetails&                    errDetails );
static OptixResult handleNoInlineFunctions( InitialCompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetHitKind( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetBuiltinBackfaceFromHitKind( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetBuiltinTypeFromHitKind( CompilationUnit&   module,
                                                   const unsigned int exceptionFlags,
                                                   const unsigned int usesPrimitiveTypeFlags,
                                                   ErrorDetails&      errDetails );
static OptixResult lowerGetPrimitiveIndex( CompilationUnit& module, const unsigned int usesPrimitiveTypeFlags, ErrorDetails& errDetails );
static OptixResult lowerUndefinedValue( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetRayFlags( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetRayVisibilityMask( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetRayTime( CompilationUnit& module, bool motionEnabled, ErrorDetails& errDetails );
static OptixResult doFunctionReplacements( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult rewriteSpecialRegisters( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetSBTData( CompilationUnit& module, int headerSize, ErrorDetails& errDetails );
static OptixResult lowerGetPayloadLegacy( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerSetPayloadLegacy( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetPayload( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerSetPayload( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerSetPayloadTypes( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerReportIntersection( CompilationUnit& module, int numAttributeValues, ErrorDetails& errDetails );
static OptixResult lowerTraceLegacy( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails );
static OptixResult lowerTrace( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails );
static OptixResult lowerGetAttribute( CompilationUnit& module, int numAttributeValues, ErrorDetails& errDetails );
static OptixResult lowerGetTriangleBarycentrics( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerThrowException( CompilationUnit& module, unsigned int exceptionFlags, ErrorDetails& errDetails );
static OptixResult lowerGetExceptionDetails( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetExceptionLineInfo( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerInbuiltExceptionDetails( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult lowerGetBuiltinISData( CompilationUnit&        module,
                                          const unsigned int      exceptionFlags,
                                          const unsigned int      usesPrimitiveTypeFlags,
                                          const bool              usesMotionBlur,
                                          OptixPrimitiveTypeFlags primitiveType,
                                          IntrinsicIndex          intrinsicIndex,
                                          const char*             name,
                                          ErrorDetails&           errDetails );

static OptixResult generateThrowOptixException( CompilationUnit&    module,
                                                OptixExceptionCodes exceptionCode,
                                                llvm::Value*        details,
                                                llvm::Instruction*  insertBefore,
                                                ErrorDetails&       errDetails );
static OptixResult generateConditionalThrowOptixException( CompilationUnit&    module,
                                                           llvm::Value*        condition,
                                                           OptixExceptionCodes exceptionCode,
                                                           llvm::Instruction*& insertBefore,
                                                           llvm::Value*        details,
                                                           ErrorDetails&       errDetails );


static OptixResult extractPipelineParamsSize( InitialCompilationUnit& module, const std::string& paramName, ErrorDetails& errDetails );
static OptixResult lowerCalls( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails );
static OptixResult optimizeModule( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails );

static OptixResult rewriteCallablesAndNoInline( CompilationUnit&                 module,
                                                const InternalCompileParameters& compileParams,
                                                ErrorDetails&                    errDetails );
static OptixResult rewriteFunctionsForLwstomABI( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult deleteDeadLwstomABIAnnotations( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult deleteDeadDirectCallableAnnotations( CompilationUnit& module, ErrorDetails& errDetails );
static OptixResult instrumentExceptionProgramsForValidationMode( CompilationUnit& module, ErrorDetails& errDetails );

static llvm::Constant* getSourceLocatiolwalue( CompilationUnit& module, llvm::Instruction* instruction );

static llvm::Constant* addModuleString( CompilationUnit& module, const std::string& str );

static llvm::Value* insertExceptionSourceLocation( CompilationUnit& module, llvm::Value* details, llvm::Instruction* instruction );

// Semantic Type checks

// Utility function to compare the semantic type of a caller with an arbitrary number
// of given semantic types and produce an error if none of the given types match
// the caller's type.
static OptixResult doSemanticTypeCheck( llvm::CallInst*                                       CI,
                                        const InternalCompileParameters&                      compileParams,
                                        ErrorDetails&                                         errDetails,
                                        const std::string&                                    apiFunctionName,
                                        const std::initializer_list<optix_exp::SemanticType>& stypes )
{
    llvm::Function*               f = CI->getParent()->getParent();
    const optix_exp::SemanticType stype =
        getSemanticTypeForFunction( f, compileParams.noinlineEnabled, compileParams.disableNoinlineFunc );

    if( optix::algorithm::find( stypes, stype ) == stypes.end() )
    {
        errDetails.m_compilerFeedback << "Error: Illegal call to " << apiFunctionName << " in function " << f->getName().str()
                                      << " with semantic type " << optix_exp::semanticTypeToString( stype ) << ": "
                                      << getSourceLocation( f->getParent(), CI ) << "\n";
        return OPTIX_ERROR_ILWALID_FUNCTION_USE;
    }
    return OPTIX_SUCCESS;
}

static OptixResult isTraceCallLegal( llvm::CallInst*                  CI,
                                     const InternalCompileParameters& compileParams,
                                     ErrorDetails&                    errDetails,
                                     const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_RAYGEN, ST_CLOSESTHIT, ST_MISS, ST_CONTINUATION_CALLABLE, ST_DIRECT_CALLABLE, ST_NOINLINE} );
}

static OptixResult isPayloadAccessLegal( llvm::CallInst*                  CI,
                                         const InternalCompileParameters& compileParams,
                                         ErrorDetails&                    errDetails,
                                         const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_MISS} );
}

static OptixResult isWorldRayAccessLegal( llvm::CallInst*                  CI,
                                          const InternalCompileParameters& compileParams,
                                          ErrorDetails&                    errDetails,
                                          const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_MISS} );
}

static OptixResult isObjectRayAccessLegal( llvm::CallInst*                  CI,
                                           const InternalCompileParameters& compileParams,
                                           ErrorDetails&                    errDetails,
                                           const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION, ST_ANYHIT} );
}

static OptixResult isRayAccessLegal( llvm::CallInst*                  CI,
                                     const InternalCompileParameters& compileParams,
                                     ErrorDetails&                    errDetails,
                                     const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_MISS} );
}

static OptixResult isTransformListAccessLegal( llvm::CallInst*                  CI,
                                               const InternalCompileParameters& compileParams,
                                               ErrorDetails&                    errDetails,
                                               const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_EXCEPTION} );
}

static OptixResult isAttributeAccessLegal( llvm::CallInst*                  CI,
                                           const InternalCompileParameters& compileParams,
                                           ErrorDetails&                    errDetails,
                                           const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT} );
}

static OptixResult isGetPrimitiveIndexLegal( llvm::CallInst*                  CI,
                                             const InternalCompileParameters& compileParams,
                                             ErrorDetails&                    errDetails,
                                             const std::string&               apiFunctionName )
{
    // No longer officially supported in EX since ABI VERSION 32.
    // However, we have to accept it here for backwards compatibility.
    // TODO: conditionally make usage in EX illegal at the next ABI function table change.
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_EXCEPTION} );
}

static OptixResult isGetSbtGasIndexLegal( llvm::CallInst*                  CI,
                                          const InternalCompileParameters& compileParams,
                                          ErrorDetails&                    errDetails,
                                          const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_EXCEPTION} );
}

static OptixResult isGetInstanceIdLegal( llvm::CallInst*                  CI,
                                         const InternalCompileParameters& compileParams,
                                         ErrorDetails&                    errDetails,
                                         const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT} );
}

static OptixResult isGetInstanceIdxLegal( llvm::CallInst*                  CI,
                                          const InternalCompileParameters& compileParams,
                                          ErrorDetails&                    errDetails,
                                          const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT} );
}

static OptixResult isGetInstanceFlagsLegal( llvm::CallInst*                  CI,
                                            const InternalCompileParameters& compileParams,
                                            ErrorDetails&                    errDetails,
                                            const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION} );
}

static OptixResult isGetHitKindLegal( llvm::CallInst*                  CI,
                                      const InternalCompileParameters& compileParams,
                                      ErrorDetails&                    errDetails,
                                      const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_ANYHIT, ST_CLOSESTHIT} );
}

static OptixResult isGetTriangleBarycentricsLegal( llvm::CallInst*                  CI,
                                                   const InternalCompileParameters& compileParams,
                                                   ErrorDetails&                    errDetails,
                                                   const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_ANYHIT, ST_CLOSESTHIT} );
}

static OptixResult isReportIntersectionLegal( llvm::CallInst*                  CI,
                                              const InternalCompileParameters& compileParams,
                                              ErrorDetails&                    errDetails,
                                              const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_INTERSECTION} );
}

static OptixResult isTerminateRayLegal( llvm::CallInst*                  CI,
                                        const InternalCompileParameters& compileParams,
                                        ErrorDetails&                    errDetails,
                                        const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_ANYHIT} );
}

static OptixResult isIgnoreIntersectionLegal( llvm::CallInst*                  CI,
                                              const InternalCompileParameters& compileParams,
                                              ErrorDetails&                    errDetails,
                                              const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_ANYHIT} );
}

static OptixResult isThrowExceptionLegal( llvm::CallInst*                  CI,
                                          const InternalCompileParameters& compileParams,
                                          ErrorDetails&                    errDetails,
                                          const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_RAYGEN, ST_MISS, ST_CLOSESTHIT, ST_ANYHIT, ST_INTERSECTION, ST_DIRECT_CALLABLE,
                                 ST_CONTINUATION_CALLABLE} );
}

static OptixResult isGetExceptionCodeLegal( llvm::CallInst*                  CI,
                                            const InternalCompileParameters& compileParams,
                                            ErrorDetails&                    errDetails,
                                            const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_EXCEPTION} );
}

static OptixResult isGetExceptionDetailLegal( llvm::CallInst*                  CI,
                                              const InternalCompileParameters& compileParams,
                                              ErrorDetails&                    errDetails,
                                              const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName, {ST_EXCEPTION} );
}

static OptixResult isDirectCallableCallLegal( llvm::CallInst*                  CI,
                                              const InternalCompileParameters& compileParams,
                                              ErrorDetails&                    errDetails,
                                              const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_RAYGEN, ST_MISS, ST_CLOSESTHIT, ST_ANYHIT, ST_INTERSECTION, ST_DIRECT_CALLABLE,
                                 ST_CONTINUATION_CALLABLE, ST_NOINLINE} );
}

static OptixResult isContinuationCallableCallLegal( llvm::CallInst*                  CI,
                                                    const InternalCompileParameters& compileParams,
                                                    ErrorDetails&                    errDetails,
                                                    const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_RAYGEN, ST_CLOSESTHIT, ST_MISS, ST_CONTINUATION_CALLABLE} );
}

static OptixResult isGetGASTraversableLegal( llvm::CallInst*                  CI,
                                             const InternalCompileParameters& compileParams,
                                             ErrorDetails&                    errDetails,
                                             const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_INTERSECTION, ST_ANYHIT, ST_CLOSESTHIT, ST_EXCEPTION} );
}

static OptixResult isGetSbtDataPtrLegal( llvm::CallInst*                  CI,
                                         const InternalCompileParameters& compileParams,
                                         ErrorDetails&                    errDetails,
                                         const std::string&               apiFunctionName )
{
    return doSemanticTypeCheck( CI, compileParams, errDetails, apiFunctionName,
                                {ST_RAYGEN, ST_MISS, ST_CLOSESTHIT, ST_ANYHIT, ST_INTERSECTION, ST_EXCEPTION,
                                 ST_DIRECT_CALLABLE, ST_CONTINUATION_CALLABLE} );
}

template <typename T>
static std::string llvmToString( T* llvm )
{
    std::string              str;
    llvm::raw_string_ostream rso( str );
    llvm->print( rso );
    rso.flush();
    return str;
}

// Collect functions based on the given comma separated list of functions.
// A warning is issued if warnIfNotFound==true (only when called from getDumpFunctionName
// which is only called once in the very beginning) if the list contains functions
// that are not present in the module.
static std::vector<llvm::Function*> getFunctionsToDump( optix_exp::Module* optixModule,
                                                        llvm::Module*      module,
                                                        const std::string& knobString,
                                                        bool               warnIfNotFound )
{
    std::vector<llvm::Function*> functions;
    std::string                  functionName;
    std::stringstream            tokens( knobString );
    while( std::getline( tokens, functionName, ',' ) )
    {
        llvm::Function* function = module->getFunction( functionName );
        if( !function )
        {
            function = module->getFunction( optixModule->getMangledName( functionName, OPTIX_PAYLOAD_TYPE_DEFAULT, ST_ILWALID ) );
            if( !function )
            {
                if( warnIfNotFound )
                {
                    lwarn << "Did not find function to dump: " << functionName << " in module\n";
                }
                continue;
            }
        }
        functions.push_back( function );
    }
    return functions;
}

// Use the name of the first found entry function plus number and hash of the remaining names (if any).
// Do not use *all* names since this will result in too long file names.
static std::string getDumpFunctionName( optix_exp::Module*               optixModule,
                                        std::vector<llvm::Function*>     functions,
                                        const InternalCompileParameters& compileParams )
{

    std::string dumpName;
    size_t      otherNamesCount = 0;

    for( llvm::Function* F : functions )
    {
        if( F->isDeclaration() )
            continue;

        optix_exp::SemanticType stype =
            getSemanticTypeForFunction( F, compileParams.noinlineEnabled, compileParams.disableNoinlineFunc );
        if( stype == optix_exp::ST_ILWALID )
            continue;

        if( dumpName.empty() )
        {
            dumpName = F->getName().str();
        }
        else
        {
            ++otherNamesCount;
        }
    }

    if( otherNamesCount )
    {
        dumpName += "_and_" + std::to_string( otherNamesCount ) + "_more_";
        dumpName += "ID" + std::to_string( optixModule->getModuleId() );
    }

    return dumpName;
}

static std::string getDumpFunctionName( optix_exp::Module* optixModule, llvm::Module* module, const InternalCompileParameters& compileParams )
{
    if( k_saveLLVM.get().empty() )
        return std::string();

    std::vector<llvm::Function*> functions;
    if( !k_saveLLVMFunctions.get().empty() )
    {
        functions = getFunctionsToDump( optixModule, module, k_saveLLVMFunctions.get(), /*warnIfNotFound=*/true );
    }
    else
    {
        functions = corelib::getFunctions( module );
    }
    return getDumpFunctionName( optixModule, functions, compileParams );
}

static std::string getModuleIdentifier( optix_exp::Module* optixModule, llvm::Module* module, const InternalCompileParameters& compileParams )
{
    return getDumpFunctionName( optixModule, corelib::getFunctions( module ), compileParams );
}

static void dump( optix_exp::Module* optixModule, llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix )
{
#if defined( DEBUG ) || defined( DEVELOP )
    std::string outFilePattern = k_saveLLVM.get();
    if( !outFilePattern.empty() )
    {
        std::string filename = corelib::createDumpPath( outFilePattern, 0, dumpId, suffix, functionName );

        std::string functionNames = k_saveLLVMFunctions.get();
        if( functionNames.empty() )
        {
            lprint << "Writing LLVM IR file to: " << filename << "\n";
            corelib::saveModuleToAsmFile( module, filename );
        }
        else
        {
            std::vector<llvm::Function*> functions =
                getFunctionsToDump( optixModule, module, functionNames, /*warnIfNotFound=*/false );
            // do not create empty files.
            if( functions.empty() )
                return;
            lprint << "Writing LLVM IR file to: " << filename << "\n";
            std::ofstream file( filename );
            for( llvm::Function* function : functions )
            {
                file << llvmToString( function ) << "\n\n";
            }
            file.close();
        }
    }
#endif
}

void addGenericErrorMessage( OptixResult result, ErrorDetails& errDetails )
{
    switch( result )
    {
        case OPTIX_ERROR_ILWALID_FUNCTION_USE:
            errDetails.logDetails( OPTIX_ERROR_ILWALID_FUNCTION_USE,
                                   "Detected invalid function use. See compile details for more information." );
            break;
        case OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS:
            errDetails.logDetails( OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS,
                                   "Detected invalid function arguments. See compile details for more information." );
            break;
        case OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER:
            errDetails.logDetails( OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER,
                                   "Invalid launch parameter. See compile details for more information." );
            break;
        case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:
            errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                   "Internal compilation error. See compile details for more information." );
            break;
        case OPTIX_ERROR_ILWALID_PTX:
            errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX,
                                   "Malformed PTX input. See compile details for more information." );
            break;
        case OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS:
            errDetails.logDetails( OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS,
                                   "Detected invalid payload access. See compile details for more information." );
            break;
        case OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS:
            result = errDetails.logDetails( OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS,
                                            "Detected invalid attribute access. See compile details for more "
                                            "information." );
            break;
        case OPTIX_ERROR_UNKNOWN:
            result = errDetails.logDetails( OPTIX_ERROR_UNKNOWN,
                                            "An unknown error oclwrred. See compiler details for more information." );
        default:
            break;
    }
}

static OptixResult linkRuntimeCode( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    // link in the runtime
    llvm::Linker linker( *module.llvmModule );

    llvm::Module* runtimeModule =
        corelib::loadModuleFromBitcodeLazy( module.llvmModule->getContext(), optix::data::getO7RuntimeData(),
                                            optix::data::getO7RuntimeDataLength() );

    if( linker.linkInModule( std::unique_ptr<llvm::Module>( runtimeModule ) ) )
    {
        errDetails.m_compilerFeedback << "Failed to load optix7 runtime bitcode module\n";
        addGenericErrorMessage( OPTIX_ERROR_INTERNAL_COMPILER_ERROR, errDetails );
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }

    return OPTIX_SUCCESS;
}

static OptixResult linkFootprintIntrinsics( CompilationUnit& module, ErrorDetails& errDetails )
{
    if( !module.subModule->usesTextureIntrinsic() )
        return OPTIX_SUCCESS;

    // link in the runtime
    llvm::Linker linker( *module.llvmModule );

    // Load HW or SW footprint module PTX
    bool useHardwareFootprint = module.compileParams.maxSmVersion >= 75 && !module.compileParams.useSoftwareTextureFootprint;
    const char* footprintSource = useHardwareFootprint ? optix::data::getTextureFootprintHWSources()[1] :
                                                         optix::data::getTextureFootprintSWSources()[1];
    const size_t footprintSourceSize = useHardwareFootprint ? optix::data::getTextureFootprintHWSourceSizes()[0] :
                                                              optix::data::getTextureFootprintSWSourceSizes()[0];

    // Prepare to colwert PTX to LLVM
    std::unique_ptr<llvm::Module> footprintModule;
    llvm::DataLayout              DL( optix::createDataLayoutForLwrrentProcess() );
    optix::PTXtoLLVM              ptx2llvm( module.llvmModule->getContext(), &DL );
    try
    {
        // Colwert PTX to LLVM IR
        std::vector<prodlib::StringView> inputStrings{prodlib::StringView( footprintSource, footprintSourceSize )};
        std::string                      moduleName = "footprint-module";
        footprintModule.reset(
            ptx2llvm.translate( "footprint-intrinsics", "" /*headers*/, inputStrings, true /*parseLineNumbers*/, "footprint-intrinsics" ) );
    }
    catch( const std::exception& e )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, std::string( "Error in footprint intrinsic PTX: " ) + e.what() );
    }
    catch( ... )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_PTX, "Unknown error in footprint intrinsic PTX" );
    }

#if defined( DEBUG ) || defined( DEVELOP )
    // Dump footprint IR
    std::string outFilePattern = k_saveLLVM.get();
    if( !outFilePattern.empty() )
    {
        std::string filename =
            corelib::createDumpPath( outFilePattern, 0, 0 /*dumpId*/, "texture_footprint", "TextureFootprint" );
        lprint << "Writing LLVM IR file to: " << filename << "\n";
        corelib::saveModuleToAsmFile( footprintModule.get(), filename );
    }
#endif

    std::vector<std::string> functionsToInline;
    for( llvm::Function* funcToInline : corelib::getFunctions( footprintModule.get() ) )
        functionsToInline.push_back( funcToInline->getName().str() );

    // Link footprint module into user module.
    if( linker.linkInModule( std::move( footprintModule ) ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                      "Failed to load optix7 texture footprint bitcode module" );
    }

    // Force inline all the functions that were linked in.  This is necessary to avoid multiple
    // definition errors when multiple submodules use footprint intrinsics.
    OptixResultOneShot result;
    llvm::Module*      destModule = module.llvmModule;
    for( const std::string& funcName : functionsToInline )
    {
        // Look up the function by name in the destination module.
        llvm::Function* destFunction = destModule->getFunction( funcName );

        // Skip declarations.  (Note that the functions in the source module turn into declarations
        // after they're linked into the destination module.)
        if( destFunction->isDeclaration() )
            continue;

        // Gather callers.
        std::vector<llvm::CallInst*> toInline;
        for( llvm::CallInst* call : corelib::getCallsToFunction( destFunction ) )
        {
            toInline.push_back( call );
        }

        // Inline calls.
        if( !corelib::inlineCalls( toInline ) )
        {
            std::string msg( "Error: failed to inline function " + destFunction->getName().str() );
            result += errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR, msg );
            continue;
        }

        // Remove the function definition.
        if( !destFunction->use_empty() )
        {
            std::string msg( "Error: uses remain after inlining function " + destFunction->getName().str() );
            result += errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR, msg );
            continue;
        }
        corelib::deleteLwvmMetadata( destFunction );
        destFunction->eraseFromParent();
    }

    return result;
}

// Set runtime function linkage to internal to avoid multiple defined symbols when linking multiple modules.
// Defining them as internal to begin with would not work because DCE would remove them right away.
OptixResult fixRuntimeLinkage( llvm::Module* llvmModule, ErrorDetails& errDetails )
{
    std::array<const char*, 8> runtimeFunctionNames = {
        STRINGIFY( RUNTIME_FETCH_LWRVE4_VERTEX_DATA ),     STRINGIFY( RUNTIME_FETCH_LWRVE3_VERTEX_DATA ),
        STRINGIFY( RUNTIME_FETCH_LWRVE2_VERTEX_DATA ),     STRINGIFY( RUNTIME_FETCH_SPHERE_DATA ),
        STRINGIFY( RUNTIME_BUILTIN_TYPE_FROM_HIT_KIND ),
        STRINGIFY( RUNTIME_BUILTIN_TYPE_TO_BUILTIN_FLAG ),
        STRINGIFY( RUNTIME_IS_BUILTIN_TYPE_SUPPORTED ),    STRINGIFY( RUNTIME_IS_EXCEPTION_LINE_INFO_AVAILABLE ) };
    for( const char* name : runtimeFunctionNames )
    {
        if( llvm::Function* fn = llvmModule->getFunction( name ) )
        {
            if( !fn )
            {
                errDetails.m_compilerFeedback << "Error: Runtime function missing\n";
                return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            }

            fn->setLinkage( llvm::GlobalValue::InternalLinkage );
        }
    }
    return OPTIX_SUCCESS;
}

static llvm::BasicBlock::iterator getSafeInsertionPoint( llvm::Function* function )
{
    llvm::BasicBlock::iterator point = function->getEntryBlock().getFirstInsertionPt();
    while( llvm::isa<llvm::AllocaInst>( *point ) )
        ++point;
    return point;
}

OptixResult addBuiltinPrimitiveCheck( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails )
{
    for( const auto& F : corelib::getFunctions( module.llvmModule ) )
    {
        if( F->isDeclaration() )
            continue;

        optix_exp::SemanticType stype = getSemanticTypeForFunction( F, module.compileParams.noinlineEnabled,
                                                                    module.compileParams.disableNoinlineFunc );
        switch( stype )
        {
            case ST_INTERSECTION:
            {
                // generate exception if user IS is ilwoked for builtin primitives
                if( compileParams.addBuiltinPrimitiveCheck )
                {
                    llvm::Instruction* CI = &*getSafeInsertionPoint( F );

                    corelib::CoreIRBuilder irb{CI};

                    llvm::Type* i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
                    llvm::Type* i64Ty = llvm::Type::getInt64Ty( module.llvmModule->getContext() );

                    llvm::FunctionType* rtcBlasTraversableType = llvm::FunctionType::get( i64Ty, false );
                    llvm::Function*     rtcBlasTraversableFunc =
                        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.blas.traversable", rtcBlasTraversableType );

                    llvm::Value* gas = irb.CreateCall( rtcBlasTraversableFunc );

                    // we check the 7th bit of the traversable to see if this is an extended header.
                    // this is internal rtcore knowledge which is not part of the rtcore interface.
                    // TODO: add an rtcore intrinsic to query this from the handle
                    llvm::Value* extendedGasBit = irb.CreateAnd( gas, irb.getInt64( 0x40 ) );
                    llvm::Value* isExtendedGas  = irb.CreateICmpNE( extendedGasBit, irb.getInt64( 0 ) );

                    llvm::Value* details = llvm::UndefValue::get( llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS ) );
                    details              = insertExceptionSourceLocation( module, details, CI );

                    OptixResult ret = generateConditionalThrowOptixException(
                        module, isExtendedGas, OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH, CI, details, errDetails );

                    if( ret != OPTIX_SUCCESS )
                        return ret;
                }
            }
            break;
            default:
                break;
        }
    }

    return OPTIX_SUCCESS;
}

static OptixResult gatherImportsAndExports( CompilationUnit& lwModule )
{
    llvm::Module*                    module        = lwModule.llvmModule;
    SubModule*                       subModule     = lwModule.subModule;
    const InternalCompileParameters& compileParams = lwModule.compileParams;

    llvm::DataLayout dl( module );
    for( llvm::Function& f : *module )
    {
        // Check for functions that are declared only.
        if( f.empty() )
        {
            if( !f.use_empty() && !isInternalFunction( f.getName() ) )
            {
                size_t size = 0;
                if( !f.getReturnType()->isVoidTy() )
                    size = dl.getTypeAllocSize( f.getReturnType() );
                subModule->addImportedFunctionSymbol( f.getName().str(), size );
            }
        }
        else if( f.getLinkage() == llvm::GlobalValue::ExternalLinkage )
        {
            // Store defined functions with external linkage for resolving.
            size_t size = 0;
            if( !f.getReturnType()->isVoidTy() )
                size = dl.getTypeAllocSize( f.getReturnType() );
            subModule->addExportedFunctionSymbol( f.getName().str(), size );
        }
    }

    for( auto globalIt = module->global_begin(), globalE = module->global_end(); globalIt != globalE; ++globalIt )
    {
        llvm::GlobalVariable* var  = &( *globalIt );
        size_t                size = dl.getTypeAllocSize( var->getType()->getPointerElementType() );
        if( var->isDeclaration() )
        {
            // Ignore the pipelineLaunchParam and unused imports incl additionally added named constants
            if( !var->use_empty()
                && optix::algorithm::find( lwModule.getNamedConstantsNames(), var->getName() )
                       == lwModule.getNamedConstantsNames().end() )
            {
                subModule->addImportedDataSymbol( var->getName().str(), size );
            }
        }
        else if( var->getLinkage() == llvm::GlobalValue::ExternalLinkage )
        {
            subModule->addExportedDataSymbol( var->getName().str(), size );
        }
    }

    return OPTIX_SUCCESS;
}

struct SplitModuleFunctionInfo
{
    std::string  name;
    unsigned int cost          = 0;
    unsigned int binAssignment = -1;

    bool operator<( const SplitModuleFunctionInfo& rhs )
    {
        // We want biggest first
        return cost > rhs.cost;
    }
};

static OptixResult replaceFunctionWithDeclaration( const llvm::StringRef& name, llvm::Module* llvmModule, ErrorDetails& errDetails )
{
    llvm::Function* function = llvmModule->getFunction( name );
    if( !function )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                      "Expected to find function " + name.str() + ", but didn't" );
    if( function->isDeclaration() )
        return OPTIX_SUCCESS;
    function->deleteBody();
    //corelib::removeAnnotation( function, "kernel" );
    corelib::deleteLwvmMetadata( function );
    return OPTIX_SUCCESS;
}

static OptixResult splitModule( InitialCompilationUnit& module, std::unique_ptr<llvm::LLVMContext>&& llvmContext, unsigned int maxNumAdditionalTasks, ErrorDetails& errDetails )
{
    // Determine how to split the module. All global variables can go into a single module
    // regardless of size. The functions need to be divided into pieces. Most compilation
    // seems to scale based on the number of basic blocks, so let's use that as the basis
    // for now.

    // Create a data structure with all the functions that are candidates for splitting
    // along with relevant data to determine each function's cost (see above). Then assign
    // each function to a "bin" where each bin will be compiled by a single SubModule.

    std::vector<SplitModuleFunctionInfo> functionInfos;
    unsigned int                         totalCost = 0;
    for( llvm::Module::iterator F = module.llvmModule->begin(), FE = module.llvmModule->end(); F != FE; ++F )
    {
        // ltemp << "function: " << F->getName().str() << ": "
        //       << ( F->isDeclaration() ? "declaration" : "definition" ) << "\n";
        if( F->isDeclaration() )
            continue;
        if( isInternalFunction( F->getName() ) )
            continue;
        if( F->getLinkage() != llvm::GlobalValue::ExternalLinkage )
        {
            // We need to share the functions between submodules, so we don't duplicate
            // the functions. In order to avoid collisions with other modules we will need
            // to mangle the function names. We will do this regardless of whether we are
            // splitting in order to reduce differences when splitting.
            std::string oldName( F->getName().str() );
            std::string newName = oldName;
            if( module.compileParams.serializeModuleId )
                newName = oldName + "_" + std::to_string( module.optixModule->getModuleId() );
            else
                newName = oldName + "_" + module.optixModule->getPtxHash();
            corelib::renameFunction( &*F, newName, /*changeDiLinkageNameOnly=*/true );
            F->setLinkage( llvm::GlobalValue::ExternalLinkage );
            llog(30) << "Renaming and setting linkage to ExternalLinkage: " << oldName << ". New name: " << newName << "\n";
        }
        unsigned int cost = 0;
        for( llvm::Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB )
            cost++;
        totalCost += cost;

        SplitModuleFunctionInfo info;
        info.name = F->getName().str();
        info.cost = cost;
        functionInfos.push_back( info );
    }

    unsigned int perBinThreshold = module.compileParams.splitModuleMinBinSize;
    // Callwlate the number of bins if we put in splitModuleMinBinSize items in each bin
    unsigned int numBins = std::max( 1u, static_cast<unsigned int>( ceilf( (float)totalCost / perBinThreshold ) ) );
    // If we have too many bins, adjust the perBinThreshold higher to match the number of bins.
    if( numBins > maxNumAdditionalTasks )
    {
        numBins         = maxNumAdditionalTasks;
        perBinThreshold = totalCost / numBins;
    }

    if( numBins == 1 || functionInfos.size() < 2 )
    {
        std::unique_ptr<SubModule> initialSubModule( new SubModule );
        initialSubModule->m_parentModule = module.optixModule;
        initialSubModule->m_llvmContext  = std::move( llvmContext );
        initialSubModule->m_llvmModule   = module.llvmModule;
        module.optixModule->addSubModule( initialSubModule.release() );

        return OPTIX_SUCCESS;
    }


    // Callwlate bin assignments
    std::vector<unsigned int> binTotals;
    std::sort( functionInfos.begin(), functionInfos.end() );
    for( SplitModuleFunctionInfo& functionInfo : functionInfos )
    {
        // find bin that fits
        auto bin = binTotals.begin();
        for( ; bin != binTotals.end(); ++bin )
        {
            if( functionInfo.cost + *bin <= perBinThreshold )
                break;
        }
        if( bin == binTotals.end() )
        {
            // Ran out of bins. If there's room, add a bin, otherwise find the smallest
            // bin. This will allow bins to have more than perBinThreshold items, but
            // that's OK.
            if( binTotals.size() < numBins )
                bin = binTotals.emplace( binTotals.end(), 0 );
            else
                bin = std::min_element( binTotals.begin(), binTotals.end() );
        }
        *bin += functionInfo.cost;
        functionInfo.binAssignment = std::distance( binTotals.begin(), bin );
    }

    const int logLevel = k_splitModuleLogLevel.get();
    if( prodlib::log::active( logLevel ) )
    {
        llog( logLevel ) << "BINNING numBins = " << numBins << ", perBinThreshold = " << perBinThreshold
                         << ", totalCost = " << totalCost << "\n";
        for( unsigned int i = 0; i < binTotals.size(); ++i )
            llog( logLevel ) << corelib::stringf( "BINNING binTotals[%2u] = %5u\n", i, binTotals[i] );
        for( SplitModuleFunctionInfo& functionInfo : functionInfos )
            llog( logLevel ) << corelib::stringf( "BINNING FI binAssignment = %2u, cost = %5u, name = %s\n",
                                                  functionInfo.binAssignment, functionInfo.cost, functionInfo.name.c_str() );
    }

    std::shared_ptr<std::vector<SplitModuleFunctionInfo>> functionInfosSharedP =
        std::make_shared<std::vector<SplitModuleFunctionInfo>>( std::move( functionInfos ) );

    std::shared_ptr<std::string> serializedModuleBuffer =
        std::make_shared<std::string>( corelib::serializeModule( module.llvmModule ) );

    // We don't need to deserialize the module in the initial SubModule. Just reuse the
    // llvm module we already have.
    std::unique_ptr<SubModule> initialSubModule( new SubModule );
    initialSubModule->m_parentModule             = module.optixModule;
    initialSubModule->m_llvmContext              = std::move( llvmContext );
    initialSubModule->m_llvmModule               = module.llvmModule;
    initialSubModule->m_splitModuleFunctionInfos = functionInfosSharedP;
    initialSubModule->m_splitModuleBinID         = 0;
    module.optixModule->addSubModule( initialSubModule.release() );

    for( size_t i = 1; i < binTotals.size(); ++i )
    {
        std::unique_ptr<SubModule> subModule( new SubModule );
        subModule->m_parentModule             = module.optixModule;
        subModule->m_serializedModule         = serializedModuleBuffer;
        subModule->m_splitModuleFunctionInfos = functionInfosSharedP;
        subModule->m_splitModuleBinID         = i;
        module.optixModule->addSubModule( subModule.release() );
    }

    return OPTIX_SUCCESS;
}

static OptixResult materializeSplitModule( SubModule* subModule, ErrorDetails& errDetails )
{
    OptixResultOneShot result;

    // Deserialize module if we don't have one already
    if( !subModule->m_llvmModule )
    {
        std::unique_ptr<llvm::LLVMContext> llvmContext( new llvm::LLVMContext() );
        llvm::StringRef bitcode( subModule->m_serializedModule->data(), subModule->m_serializedModule->size() );
        std::unique_ptr<llvm::MemoryBuffer> buffer( llvm::MemoryBuffer::getMemBuffer( bitcode, "", /*null-terminated*/ false ) );
        llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrError =
            llvm::parseBitcodeFile( buffer->getMemBufferRef(), *llvmContext );
        if( llvm::Error error = moduleOrError.takeError() )
        {
            std::string              errorMessage;
            llvm::raw_string_ostream errorStream( errorMessage );
            errorStream << error;
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                          "Error deserializing SubModule with error: " + errorMessage );
        }
        llvm::Module* llvmModule = moduleOrError.get().release();
        subModule->m_llvmContext = std::move( llvmContext );
        subModule->m_llvmModule  = llvmModule;
        subModule->m_serializedModule.reset();
    }

    if( result != OPTIX_SUCCESS )
        return result;

    if( !subModule->m_splitModuleFunctionInfos )
        return result;

    for( const SplitModuleFunctionInfo& functionInfo : *subModule->m_splitModuleFunctionInfos )
    {
        if( functionInfo.binAssignment != subModule->m_splitModuleBinID )
            result += replaceFunctionWithDeclaration( functionInfo.name, subModule->m_llvmModule, errDetails );
    }

    subModule->m_splitModuleFunctionInfos.reset();

    return result;
}

OptixResult createSubModules( Module*                              optixModule,
                              std::unique_ptr<llvm::LLVMContext>&& llvmContext,
                              llvm::Module*                        llvmModule,
                              unsigned int                         maxNumAdditionalTasks,
                              ErrorDetails&                        errDetails )
{
    const InternalCompileParameters& compileParams = optixModule->getCompileParameters();
    int                dumpCount = 0;
    OptixResultOneShot result;

    optixModule->setModuleIdentifier( getModuleIdentifier( optixModule, llvmModule, compileParams ) );

    std::string dumpFunctionName = getDumpFunctionName( optixModule, llvmModule, compileParams );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "initial" );

    // Add debug information if necessary
    if( compileParams.debugLevel != OPTIX_COMPILE_DEBUG_LEVEL_NONE )
        if( !llvmModule->getModuleFlag( "Debug Info Version" ) )
            llvmModule->addModuleFlag( llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION );

    // run global DCE to reduce the number of functions to handle (especially for the noinline type)
    llvm::legacy::PassManager PM;
    PM.add( llvm::createGlobalDCEPass() );
    PM.run( *llvmModule );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterDCE" );

    InitialCompilationUnit module( optixModule, llvmModule, compileParams );
    result += module.initAndVerifyIntrinsics( errDetails );
    if( result != OPTIX_SUCCESS )
        return result;

    result += validateInputPtx( module, errDetails );

    // Make sure to return here in case of failures as long as rtcore is not protected
    // against call graph cycles (lwbugs 2701292).
    if( result != OPTIX_SUCCESS )
    {
        addGenericErrorMessage( result, errDetails );
        return result;
    }

    result += extractPipelineParamsSize( module, compileParams.pipelineLaunchParamsVariableName, errDetails );

    // Link the helper runtime functions
    result += linkRuntimeCode( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLinkRuntime" );

    result += handleNoInlineFunctions( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterHandleNoInlineFunctions" );

    result += splitModule( module, std::move( llvmContext ), maxNumAdditionalTasks, errDetails );

    return result;
}

OptixResult compileSubModule( SubModule* subModule, bool& fellBackToLWPTX, ErrorDetails& errDetails )
{
    if( OptixResult result = materializeSplitModule( subModule, errDetails ) )
        return result;

    Module*                          optixModule   = subModule->m_parentModule;
    llvm::Module*                    llvmModule    = subModule->m_llvmModule;
    const InternalCompileParameters& compileParams = subModule->m_parentModule->getCompileParameters();
    int                              dumpCount     = 10; // TODO (jbigler) figure out what the file naming should be for subModule compilation.

    std::string dumpFunctionName = getDumpFunctionName( optixModule, llvmModule, compileParams );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "initialSubModule" );

    OptixResultOneShot               result;

    CompilationUnit module( subModule, llvmModule, compileParams );
    result += module.init( errDetails );
    if( result )
        return result;

    ////////////////////////////////////////////////////////////////////////////////

    // lower payload types BEFORE lowering payload getter and setter
    // the type is used to validate the getter and setter.
    // lower payload types BEFORE any exceptions are generated, so we can validate
    // that this function is called at the top of the function.
    result += lowerSetPayloadTypes( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerSetPayloadTypes" );

    std::set<llvm::Function*> specializedFunctions;
    result += specializeLaunchParam( module, specializedFunctions, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterSpecializeLaunchParam" );

    result += lowerPrivateCompileTimeConstant( module, specializedFunctions, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerPrivateCompileTimeContant" );

    result += optimizeSpecializedFunctions( module, specializedFunctions, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterOptimizeSpecializedLaunchParamFunctions" );

    result += doFunctionReplacements( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterFunctionReplacements" );

    result += rewriteSpecialRegisters( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterRewriteSpecialRegisters" );

    result += linkFootprintIntrinsics( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLinkFootprintIntrinsics" );

    result += lowerGetHitKind( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetHitKind" );

    result += lowerGetBuiltinTypeFromHitKind( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetBuiltinType" );

    result += lowerGetBuiltinBackfaceFromHitKind( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetBuiltinBackface" );

    result += lowerGetPrimitiveIndex( module, compileParams.usesPrimitiveTypeFlags, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetPrimitiveIndex" );

    result += lowerUndefinedValue( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerUndefinedValue" );

    result += lowerGetRayVisibilityMask( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetRayVisibilityMask" );

    result += lowerGetRayFlags( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetRayFlags" );
    result += lowerGetRayTime( module, compileParams.usesMotionBlur, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetRayTime" );

    int numAttributeValues = compileParams.numAttributeValues;

    result += lowerGetBuiltinISData( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags,
                                     compileParams.usesMotionBlur, OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
                                     optix_get_linear_lwrve_vertex_data, STRINGIFY( RUNTIME_FETCH_LWRVE2_VERTEX_DATA ), errDetails );
    result += lowerGetBuiltinISData( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags,
                                     compileParams.usesMotionBlur, OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
                                     optix_get_quadratic_bspline_vertex_data,
                                     STRINGIFY( RUNTIME_FETCH_LWRVE3_VERTEX_DATA ), errDetails );
    result += lowerGetBuiltinISData( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags,
                                     compileParams.usesMotionBlur, OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE,
                                     optix_get_lwbic_bspline_vertex_data, STRINGIFY( RUNTIME_FETCH_LWRVE4_VERTEX_DATA ), errDetails );
    result += lowerGetBuiltinISData( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags,
                                     compileParams.usesMotionBlur, OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,
                                     optix_get_catmullrom_vertex_data, STRINGIFY( RUNTIME_FETCH_LWRVE4_VERTEX_DATA ), errDetails );
    result += lowerGetBuiltinISData( module, compileParams.exceptionFlags, compileParams.usesPrimitiveTypeFlags,
                                     compileParams.usesMotionBlur, OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE,
                                     optix_get_sphere_data, STRINGIFY( RUNTIME_FETCH_SPHERE_DATA ), errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetBuiltinISData" );

    result += lowerGetSBTData( module, compileParams.sbtHeaderSize, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetSBTData" );

    result += lowerGetPayloadLegacy( module, errDetails );
    result += lowerGetPayload( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetPayload" );

    result += lowerSetPayloadLegacy( module, errDetails );
    result += lowerSetPayload( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerSetPayload" );

    result += lowerReportIntersection( module, numAttributeValues, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerReportIntersection" );

    result += lowerTraceLegacy( module, compileParams, errDetails );
    result += lowerTrace( module, compileParams, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerTrace" );

    result += lowerGetAttribute( module, numAttributeValues, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetAttribute" );

    result += lowerGetTriangleBarycentrics( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetTriangleBarycentrics" );

    result += addBuiltinPrimitiveCheck( module, compileParams, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterAddBuiltinPrimitiveCheck" );

    result += lowerThrowException( module, compileParams.exceptionFlags, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterThrowException" );

    result += lowerGetExceptionDetails( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetExceptionDetails" );

    result += lowerInbuiltExceptionDetails( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerInbuiltExceptionDetails" );

    result += lowerGetExceptionLineInfo( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerGetExceptionLineInfo" );

    result += rewriteCallablesAndNoInline( module, compileParams, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterRewriteCallablesAndNoInline" );

    result += rewriteFunctionsForLwstomABI( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterRewriteFunctionsForLwstomABI" );

    result += instrumentExceptionProgramsForValidationMode( module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++,
          "afterInstrumentUserExceptionProgramsForValidationMode" );

    result += lowerCalls( module, compileParams, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterLowerCalls" );

    result += fixRuntimeLinkage( module.llvmModule, errDetails );

    errDetails.m_compilerFeedback << "Info: Module uses " << module.usedPayloadValues
                                  << " payload values.";
    errDetails.m_compilerFeedback << "Info: Module uses " << module.usedAttributeValues
                                  << " attribute values. Pipeline configuration: " << numAttributeValues
                                  << ( numAttributeValues == 2 ? " (default)" : "" ) << ".\n";

    result += addSemanticTypeAnnotations( subModule, module, errDetails );
    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "afterAddSemanticTypeAnnotations" );

    // avoid linking when there are failures already
    if( static_cast<OptixResult>( result ) != OPTIX_SUCCESS )
        return result;

    result += linkPTXFrontEndIntrinsics( module.llvmModule, compileParams.useD2IR, compileParams.enableLWPTXFallback,
                                         fellBackToLWPTX, errDetails );

    // Move any variables that haven't been given an address space to the global
    // space, for D2IR compatibility.
    optix::moveVariablesFromUnnamedToGlobalAddressSpace( module.llvmModule );

    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "beforeOpt" );

    result += optimizeModule( module, compileParams, errDetails );

    // Function annotations can be orphaned when their functions are DCE'd.
    // These annotations are ill-formed, and cause assertions to fail in
    // RTCore, so we check for and remove them here.
    result += deleteDeadLwstomABIAnnotations( module, errDetails );
    result += deleteDeadDirectCallableAnnotations( module, errDetails );

    dump( optixModule, llvmModule, dumpFunctionName, dumpCount++, "final" );

    result += gatherImportsAndExports( module );

    addGenericErrorMessage( result, errDetails );

    return result;
}

bool intrinsicRequiresInlining( IntrinsicIndex intrinsic, SemanticType stype )
{
    switch( intrinsic )
    {
        // Legal in any OptiX program, with or without noinline.
        case optix_tex_footprint_2d:
        case optix_tex_footprint_2d_grad:
        case optix_tex_footprint_2d_lod:
        case optix_tex_footprint_2d_v2:
        case optix_tex_footprint_2d_grad_v2:
        case optix_tex_footprint_2d_lod_v2:
            // Bypassing inlining of footprint-related code shouldn't have much performance impact
            // because such functions are usually quite large (i.e. they call demandLoading::tex2DGrad)
            return false;
        // Legal with noinline in any OptiX program
        case optix_get_transform_type_from_handle:
        case optix_get_instance_child_from_handle:
        case optix_get_instance_id_from_handle:
        case optix_get_instance_transform_from_handle:
        case optix_get_instance_ilwerse_transform_from_handle:
        case optix_get_static_transform_from_handle:
        case optix_get_matrix_motion_transform_from_handle:
        case optix_get_srt_motion_transform_from_handle:
        case optix_get_gas_motion_time_begin:
        case optix_get_gas_motion_time_end:
        case optix_get_gas_motion_step_count:
        case optix_get_primitive_type_from_hit_kind:
        case optix_is_hitkind_backface:  // optixIsFrontFaceHit( hitKind ), optixIsBackFraceHit( hitKind )
        case optix_get_triangle_vertex_data:
        case optix_get_linear_lwrve_vertex_data:
        case optix_get_quadratic_bspline_vertex_data:
        case optix_get_lwbic_bspline_vertex_data:
        case optix_get_catmullrom_vertex_data:
        case optix_get_sphere_data:
        case optix_get_instance_traversable_from_ias:
        case optix_get_launch_index_x:
        case optix_get_launch_index_y:
        case optix_get_launch_index_z:
        case optix_get_launch_dimension_x:
        case optix_get_launch_dimension_y:
        case optix_get_launch_dimension_z:
        // Legal with noinline in DC
        case optix_call_direct_callable:
        // Legal with noinline in DC, but intentionally not dolwmented
        case optix_trace_0:
        case optix_trace_1:
        case optix_trace_2:
        case optix_trace_3:
        case optix_trace_4:
        case optix_trace_5:
        case optix_trace_6:
        case optix_trace_7:
        case optix_trace_8:
        case optix_trace_32:
        case optix_trace_typed_32:
            // Inlined unless function is no-inline.
            return stype != ST_NOINLINE;
        // Legal in DC, but not in noinline
        case optix_throw_exception_0:
        case optix_throw_exception_1:
        case optix_throw_exception_2:
        case optix_throw_exception_3:
        case optix_throw_exception_4:
        case optix_throw_exception_5:
        case optix_throw_exception_6:
        case optix_throw_exception_7:
        case optix_throw_exception_8:
        default:
            // Always inlined.
            return true;
    }
}

// This function is almost the same as corelib::inlineAllCallersOfFunction, but we
// want to report which functions have been inlined and which functions
// have been removed. Also, we want to remove the functions regardless of their linkage.
static OptixResult inlineAllCallersOfIntrinsic( const InitialCompilationUnit& module, IntrinsicIndex intrinsicIndex, ErrorDetails& errDetails )
{
    llvm::SmallVector<llvm::Function*, 16> workList;
    llvm::SmallPtrSet<llvm::Function*, 16> visited;

    llvm::Function* function = module.llvmIntrinsics[intrinsicIndex];

    OptixResultOneShot result;
    if( !function )
        return result;

    for( llvm::CallInst* call : corelib::getCallsToFunction( function ) )
    {
        llvm::Function* caller = call->getParent()->getParent();

        if( visited.insert( caller ).second )
            workList.push_back( caller );
    }

    while( !workList.empty() )
    {
        llvm::Function* functionToInline = workList.pop_back_val();
        SemanticType stype = getSemanticTypeForFunction( functionToInline, module.compileParams.noinlineEnabled,
                                                         module.compileParams.disableNoinlineFunc );

        if( stype != ST_ILWALID && stype != ST_NOINLINE )
            continue;
        if( !intrinsicRequiresInlining( intrinsicIndex, stype ) )
            continue;

        std::vector<llvm::CallInst*> toInline;

        for( llvm::CallInst* call : corelib::getCallsToFunction( functionToInline ) )
        {
            toInline.push_back( call );

            llvm::Function* caller = call->getParent()->getParent();
            if( visited.insert( caller ).second )
                workList.push_back( caller );
        }
        if( !toInline.empty() )
            errDetails.m_compilerFeedback << "Info: Inlining all calls to " << functionToInline->getName().str()
                                          << " because it calls API function " << apiName( intrinsicIndex ) << " ("
                                          << intrinsicName( intrinsicIndex ) << ")\n";

        if( !corelib::inlineCalls( toInline ) )
        {
            errDetails.m_compilerFeedback << "Error: failed to inline function " << functionToInline->getName().str() << "\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            continue;
        }

        if( functionToInline->use_empty() )
        {
            errDetails.m_compilerFeedback << "Info: Removed function " << functionToInline->getName().str()
                                          << " after inlining.\n";
            // make sure to delete lwvm metadata (if any).
            corelib::deleteLwvmMetadata( functionToInline );
            functionToInline->eraseFromParent();
        }
        else
        {
            errDetails.m_compilerFeedback << "Error: Function " << functionToInline->getName().str()
                                          << " still in use after inlining.\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }
    }
    return result;
}

// Inlining callers of optix intrinsics for PTX special register access like optix.lwvm.read.ptx.sreg.tid.x
static OptixResult inlineAllCallersOfSRegIntrinsic( llvm::Module*                    llvmModule,
                                                    const InternalCompileParameters& compileParams,
                                                    const std::string&               intrinsicName,
                                                    ErrorDetails&                    errDetails )
{
    llvm::SmallVector<llvm::Function*, 16> workList;
    llvm::SmallPtrSet<llvm::Function*, 16> visited;

    OptixResultOneShot result;
    llvm::Function*    function = llvmModule->getFunction( intrinsicName );
    if( !function )
        return result;

    for( llvm::CallInst* call : corelib::getCallsToFunction( function ) )
    {
        llvm::Function* caller = call->getParent()->getParent();

        if( visited.insert( caller ).second )
            workList.push_back( caller );
    }

    while( !workList.empty() )
    {
        llvm::Function* functionToInline = workList.pop_back_val();
        SemanticType    stype =
            getSemanticTypeForFunction( functionToInline, compileParams.noinlineEnabled, compileParams.disableNoinlineFunc );

        if( stype != ST_ILWALID && stype != ST_NOINLINE )
            continue;
        if( stype == ST_NOINLINE )
            continue;

        std::vector<llvm::CallInst*> toInline;

        for( llvm::CallInst* call : corelib::getCallsToFunction( functionToInline ) )
        {
            toInline.push_back( call );

            llvm::Function* caller = call->getParent()->getParent();
            if( visited.insert( caller ).second )
                workList.push_back( caller );
        }
        if( !toInline.empty() )
            errDetails.m_compilerFeedback << "Info: Inlining all calls to " << functionToInline->getName().str()
                                          << " because it calls intrinsic " << intrinsicName << "\n";

        if( !corelib::inlineCalls( toInline ) )
        {
            errDetails.m_compilerFeedback << "Error: failed to inline function " << functionToInline->getName().str() << "\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            continue;
        }

        if( functionToInline->use_empty() )
        {
            errDetails.m_compilerFeedback << "Info: Removed function " << functionToInline->getName().str() << " after inlining.\n";
            // make sure to delete lwvm metadata (if any).
            corelib::deleteLwvmMetadata( functionToInline );
            functionToInline->eraseFromParent();
        }
        else
        {
            errDetails.m_compilerFeedback << "Error: Function " << functionToInline->getName().str() << " still in use after inlining.\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }
    }
    return result;
}

static OptixResult inlineCalleesToCallers( const InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    if( module.compileParams.forceInlineSet.empty() )
        return result;

    // Now parse the knob
    std::map<llvm::StringRef, std::set<llvm::StringRef>> calleeCallerSet;

    llvm::SmallVector<llvm::StringRef, 4> tokens;
    llvm::StringRef( module.compileParams.forceInlineSet ).split( tokens, ",", -1, false );
    for( const llvm::StringRef& token : tokens )
    {
        std::pair<llvm::StringRef, llvm::StringRef> calleeCallerPair = token.split( "->" );
        std::set<llvm::StringRef>& callerSet = calleeCallerSet[calleeCallerPair.first];
        if( !calleeCallerPair.second.empty() )
            callerSet.emplace( calleeCallerPair.second );
    }

    for( llvm::Function* F : corelib::getFunctions( module.llvmModule ) )
    {
        if( F->isDeclaration() )
            continue;

        // Determine if F is in our list
        llvm::Function*           callee = nullptr;
        std::set<llvm::StringRef> callers;
        for( auto& iter : calleeCallerSet )
        {
            if( F->getName().find( iter.first ) != llvm::StringRef::npos )
            {
                callee  = F;
                callers = iter.second;
                break;
            }
        }

        if( !callee )
            continue;

        // Find all callers
        std::vector<llvm::CallInst*> toInline;
        for( llvm::CallInst* call : corelib::getCallsToFunction( callee ) )
        {
            // Always inline if callers is empty
            if( callers.empty() )
            {
                toInline.push_back( call );
                continue;
            }

            for( const llvm::StringRef& caller : callers )
            {
                if( call->getParent()->getParent()->getName().find( caller ) != llvm::StringRef::npos )
                {
                    toInline.push_back( call );
                    break;
                }
            }
        }

        if( !corelib::inlineCalls( toInline ) )
        {
            errDetails.m_compilerFeedback << "Error: failed to inline function " << callee->getName().str() << "\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            continue;
        }

        // Remove the function if there are no more uses and it's internal linkage
        if( callee->use_empty() && callee->getLinkage() == llvm::GlobalValue::InternalLinkage )
        {
            errDetails.m_compilerFeedback << "Info: Removed function " << callee->getName().str()
                                          << " after being forced inlined.\n";
            // make sure to delete lwvm metadata (if any).
            corelib::deleteLwvmMetadata( callee );
            callee->eraseFromParent();
        }
    }
    return result;
}

// Helper function that extracts valEnd-valOffset bytes from specializatiolwalue starting at valOffset
// and returns those bytes as a constant int of the given elementSize shifted by shiftStart bytes.
static llvm::Constant* makeConstantInt( const std::vector<char>& specializatiolwalue,
                                        int                      valOffset,
                                        int                      valEnd,
                                        int                      shiftStart,
                                        int                      elementSize,
                                        CompilationUnit&         module )
{
    std::vector<llvm::Constant*> bytes;
    bytes.reserve( elementSize );
    llvm::Type* i8Ty = llvm::Type::getInt8Ty( module.llvmModule->getContext() );

    for( int start = valOffset, end = valEnd; start < end; ++start )
        bytes.push_back( llvm::ConstantInt::get( i8Ty, specializatiolwalue[start] ) );


    // Build an integer of the right size
    llvm::IntegerType* intType = llvm::IntegerType::get( module.llvmModule->getContext(), /* NumBits */ elementSize * 8 );
    llvm::Constant*    value   = llvm::Constant::getNullValue( intType );
    if( elementSize == 1 )
    {
        // This branch is here because getZExt doesn't like it if the type
        // sizes are the same which happens when you only have a single byte.
        value = bytes[0];
    }
    else
    {
        // Now loop over each byte and shift it into position.  LLVM will
        // construct the final type for us with the right size.
        int shift = shiftStart;
        for( llvm::Constant* byte : bytes )
        {
            // Zero extend, shift, and or in bytes
            byte  = llvm::ConstantExpr::getZExt( byte, intType );
            byte  = llvm::ConstantExpr::getShl( byte, llvm::ConstantInt::get( intType, shift ) );
            value = llvm::ConstantExpr::getOr( byte, value );
            shift += 8;
        }
    }
    return value;
}

static OptixResult validateSpecialization( const CompileBoundValueEntry& specialization, size_t launchParamSize, ErrorDetails& errDetails )
{
    if( specialization.value.size() == 0 )
    {
        errDetails.m_compilerFeedback << "Error: Found launch parameter specialization with zero size:\n"
                                      << "\t"
                                      << "annotation : " << specialization.annotation << "\n"
                                      << "\t"
                                      << "offset     : " << specialization.offset << "\n";
        return OPTIX_ERROR_ILWALID_VALUE;
    }

    if( specialization.offset > launchParamSize )
    {
        errDetails.m_compilerFeedback << "Error: Specialization offset is larger than the launch parameter size.\n"
                                      << "\t"
                                      << "annotation            : " << specialization.annotation << "\n"
                                      << "\t"
                                      << "offset                : " << specialization.offset << "\n"
                                      << "\t"
                                      << "size                  : " << specialization.value.size() << "\n"
                                      << "\t"
                                      << "launch parameter size : " << launchParamSize << "\n";

        return OPTIX_ERROR_ILWALID_VALUE;
    }

    if( specialization.offset + specialization.value.size() > launchParamSize )
    {
        errDetails.m_compilerFeedback << "Error: Specialization exceeds the launch parameter size.\n"
                                      << "\t"
                                      << "annotation            : " << specialization.annotation << "\n"
                                      << "\t"
                                      << "offset                : " << specialization.offset << "\n"
                                      << "\t"
                                      << "size                  : " << specialization.value.size() << "\n"
                                      << "\t"
                                      << "launch parameter size : " << launchParamSize << "\n";
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

static OptixResult validateSpecializations( const std::vector<CompileBoundValueEntry>& specializations,
                                            size_t                                     launchParamSize,
                                            ErrorDetails&                              errDetails )
{
    // Check if specializations are overlapping. Specializations are sorted by their start offset, so we can use that.
    const CompileBoundValueEntry& first = specializations[0];
    if( OptixResult res = validateSpecialization( first, launchParamSize, errDetails ) )
        return res;

    size_t lastEnd = first.offset + first.value.size();
    for( int i = 1; i < specializations.size(); ++i )
    {
        const CompileBoundValueEntry& current = specializations[i];
        if( OptixResult res = validateSpecialization( current, launchParamSize, errDetails ) )
            return res;

        if( lastEnd > current.offset )
        {
            const CompileBoundValueEntry& previous = specializations[i - 1];

            errDetails.m_compilerFeedback << "Error: Found overlapping launch parameter specializations:\n"
                                          << "\t"
                                          << "annotation 1 : " << previous.annotation << "\n"
                                          << "\t"
                                          << "offset     1 : " << previous.offset << "\n"
                                          << "\t"
                                          << "size       1 : " << previous.value.size() << "\n"
                                          << "\t"
                                          << "annotation 2 : " << current.annotation << "\n"
                                          << "\t"
                                          << "offset     2 : " << current.offset << "\n"
                                          << "\t"
                                          << "size       2 : " << current.value.size() << "\n";

            return OPTIX_ERROR_ILWALID_VALUE;
        }
        lastEnd = current.offset + current.value.size();
    }
    return OPTIX_SUCCESS;
}

// Helper structs for specializations
struct LoadElement
{
    LoadElement( size_t o, size_t s, llvm::Type* t, llvm::Constant* sv, const CompileBoundValueEntry* spec )
        : offset( o )
        , size( s )
        , type( t )
        , specializedValue( sv )
        , specialization( spec )
    {
    }
    size_t                        offset;
    size_t                        size;
    llvm::Type*                   type             = nullptr;
    llvm::Constant*               specializedValue = nullptr;
    const CompileBoundValueEntry* specialization   = nullptr;
    std::vector<LoadElement>      subElements;
    unsigned int                  specializedSubElements = 0;
    // For partial specialization of loads the value will be put "in the middle"
    // of the loaded value. This mask determines where it ends up.
    llvm::Constant*               partialSpecializationMask = nullptr;
};

struct PointerUse
{
    llvm::Value* pointer;
    size_t       offset;
};

static llvm::Value* createPartiallySpecializedValue( llvm::Value*            originalValue,
                                                     llvm::Value*            specializatiolwalue,
                                                     const LoadElement&      element,
                                                     llvm::LoadInst*         L,
                                                     corelib::CoreIRBuilder& irb )
{
    llvm::Value* mask = element.partialSpecializationMask;
    if( !element.type->isIntegerTy() )
    {
        llvm::Module*      module  = L->getParent()->getParent()->getParent();
        llvm::IntegerType* intType = llvm::IntegerType::get( module->getContext(), /* NumBits */ element.size * 8 );
        originalValue              = irb.CreateBitCast( originalValue, intType );
        specializatiolwalue        = irb.CreateBitCast( specializatiolwalue, intType );
    }
    llvm::Value* andVal = irb.CreateAnd( originalValue, mask, L->getName() + ".and" );
    llvm::Value* orVal  = irb.CreateOr( andVal, specializatiolwalue, L->getName() + ".or" );
    if( !element.type->isIntegerTy() )
        orVal = irb.CreateBitCast( orVal, element.type );
    return orVal;
}

// Processes the given LoadElements which are part of either a vector or struct load.
// Fills the resultValue which then contains the specialized vector or struct.
// Works relwrsively for nested structs.
static OptixResult specializeAggregateElements( const std::vector<LoadElement>& elements,
                                                llvm::Type*                     loadType,
                                                llvm::LoadInst*                 L,
                                                PointerUse                      pointerUse,
                                                unsigned int                    elementsSpecialized,
                                                corelib::CoreIRBuilder&         irb,
                                                llvm::Value*&                   resultValue,
                                                ErrorDetails&                   errDetails )
{
    bool        isVectorLoad = loadType->isVectorTy();
    unsigned    numElements  = isVectorLoad ? loadType->getVectorNumElements() : loadType->getStructNumElements();
    if( elements.size() != numElements )
    {
        errDetails.m_compilerFeedback << "Error: " << ( isVectorLoad ? "vector" : "struct" )
                                      << " load was split in a different number of elements (" << elements.size()
                                      << ") than " << ( isVectorLoad ? "vector" : "struct" ) << " elements ("
                                      << numElements << "):" << getSourceLocation( L ) << "\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }

    bool partialLoad = elements.size() != elementsSpecialized || ( elements.size() && elements[0].partialSpecializationMask );

    std::string        annotation;
    OptixResultOneShot result;
    for( int i = 0; i < elements.size(); ++i )
    {
        const LoadElement& element          = elements[i];
        llvm::Value*       specializedValue = element.specializedValue;
        if( element.type->isStructTy() )
        {
            if( isVectorLoad )
            {
                errDetails.m_compilerFeedback << "Error: Struct elements in vectors are not supported in load "
                                              << getSourceLocation( L ) << "\n";
                result += OPTIX_ERROR_ILWALID_PTX;
                continue;
            }
            specializedValue = irb.CreateExtractValue( resultValue, i );
            if( specializedValue->getType() != element.type )
            {
                errDetails.m_compilerFeedback << "Error: Wrong type for nested struct in specialization. Expected "
                                              << llvmToString( element.type ) << " but found "
                                              << llvmToString( specializedValue->getType() ) << " in load "
                                              << getSourceLocation( L ) << "\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                continue;
            }
            result += specializeAggregateElements( element.subElements, element.type, L, pointerUse,
                                                   element.specializedSubElements, irb, specializedValue, errDetails );
        }
        if( specializedValue == nullptr )
        {
            if( !partialLoad )
            {
                errDetails.m_compilerFeedback << "Error: Missing element specialization for "
                                              << ( isVectorLoad ? "vector" : "struct" )
                                              << "load that is fully specialized:" << getSourceLocation( L ) << "\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            }
            continue;
        }
        if( isVectorLoad )
        {
            int insertIdx    = element.offset - pointerUse.offset;
            insertIdx        = insertIdx / element.size;
            llvm::Value* idx = irb.getInt32( insertIdx );
            if( element.partialSpecializationMask )
            {
                llvm::Value* orgVal = irb.CreateExtractElement( resultValue, idx );
                specializedValue    = createPartiallySpecializedValue( orgVal, specializedValue, element, L, irb );
            }
            resultValue      = irb.CreateInsertElement( resultValue, specializedValue, idx );
        }
        else
        {
            if( element.partialSpecializationMask )
            {
                // The load is only partially specialized
                llvm::Value* orgVal = irb.CreateExtractValue( resultValue, i );
                specializedValue    = createPartiallySpecializedValue( orgVal, specializedValue, element, L, irb );
            }
            resultValue = irb.CreateInsertValue( resultValue, specializedValue, i );
        }
        // clang-format off
        // Struct entries for nested structs do not have their own specialization, they are specialized
        // on their nestedElements.
        if( element.specialization )
        {
            if( element.specialization->annotation != annotation )
            {
                errDetails.m_compilerFeedback
                    << "\t" << "annotation : " << element.specialization->annotation << "\n"
                    << "\t" << "offset     : " << element.specialization->offset << "\t"
                    << "\t" << "size       : " << element.specialization->value.size() << "\n";
                if( element.partialSpecializationMask && element.specializedSubElements )
                {
                    // This load has multiple partial specializations. Print those.
                    for( const LoadElement& subElement : element.subElements )
                        errDetails.m_compilerFeedback
                            << "\t" << "annotation : " << subElement.specialization->annotation << "\n"
                            << "\t" << "offset     : " << subElement.specialization->offset << "\t"
                            << "\t" << "size       : " << subElement.specialization->value.size() << "\n";
                }
                annotation = element.specialization->annotation;
            }
            errDetails.m_compilerFeedback
                << "\t" << "load offset: " << element.offset << "\t"
                << "\t" << "load size  : " << element.size << "\n";
        }
        // clang-format on
    }
    return result;
}

// Specializes the given LoadInst L based on the given LoadElement vector
// and replaces the load with the specialized value.
static OptixResult specializeAggregateLoad( const std::vector<LoadElement>& elements,
                                            PointerUse                      pointerUse,
                                            llvm::LoadInst*                 L,
                                            unsigned int                    elementsSpecialized,
                                            ErrorDetails&                   errDetails )
{
    llvm::Type* loadType = L->getType();
    if( !loadType->isVectorTy() && !loadType->isStructTy() )
    {
        errDetails.m_compilerFeedback << "Error: Invalid load to specialize for aggregate: " << getSourceLocation( L );
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }
    bool partialVectorLoad = elements.size() != elementsSpecialized || ( elements.size() && elements[0].partialSpecializationMask );

    // Replace all current uses of L with a dummy value first. Replacing L's uses
    // after inserting the new values would also replace L in the new InsertElementInsts
    // if the load is only partially specialized.
    corelib::CoreIRBuilder irb{ L };
    llvm::Instruction*     dummy = irb.CreateLoad( llvm::UndefValue::get( loadType->getPointerTo() ) );
    L->replaceAllUsesWith( dummy );

    // In the partial specialization case we want to keep some values of L.
    llvm::Value* newVec = nullptr;
    if( partialVectorLoad )
        newVec = L;
    else
        newVec = llvm::UndefValue::get( loadType );

    irb.SetInsertPoint( L->getNextNode() );
    errDetails.m_compilerFeedback << "Info: Found specialization\n"
                                  << "\t" << getSourceLocation( L ) << "\n";

    OptixResult result = specializeAggregateElements( elements, loadType, L, pointerUse, elementsSpecialized, irb, newVec, errDetails );

    newVec->takeName( L );
    dummy->replaceAllUsesWith( newVec );
    dummy->eraseFromParent();
    return result;
}

// Fills the LoadElement vector for struct types. Works relwrsively for nested structs.
static void fillStructTypeLoadElements( std::vector<LoadElement>& elements,
                                        size_t                    startOffset,
                                        llvm::StructType*         structType,
                                        const llvm::DataLayout&   DL )
{
    const llvm::StructLayout* sl            = DL.getStructLayout( structType );
    for( int i = 0; i < structType->getNumElements(); ++i )
    {
        size_t lwrrentOffset = startOffset + sl->getElementOffset( i );
        llvm::Type* elementTy = structType->getElementType( i );
        size_t      eltSize   = DL.getTypeStoreSize( elementTy );
        elements.push_back( LoadElement( lwrrentOffset, eltSize, elementTy, nullptr, nullptr ) );
        if( elementTy->isStructTy() )
            fillStructTypeLoadElements( elements.back().subElements, lwrrentOffset, llvm::cast<llvm::StructType>( elementTy ), DL );
    }
}

// Searches the specializations in the compile params for matching regions in the given LoadElement vector.
static OptixResult findElementSpecializations( CompilationUnit&          module,
                                               std::vector<LoadElement>& elements,
                                               unsigned int&             elementsSpecialized,
                                               llvm::LoadInst*           L,
                                               ErrorDetails&             errDetails )
{
    OptixResultOneShot result;
    for( LoadElement& element : elements )
    {
        if( element.type->isStructTy() )
        {
            if( element.subElements.empty() )
            {
                errDetails.m_compilerFeedback << "Error: trying to specialize a nested struct without elements.\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                continue;
            }
            result += findElementSpecializations( module, element.subElements, element.specializedSubElements, L, errDetails );
            if( element.specializedSubElements != 0 )
                ++elementsSpecialized;
            continue;
        }

        bool specializationFound = false;
        bool               partialSpecialization = false;
        // Create the specialized value as an int of the right type to allow partial specialization. It will be bitcasted to the right type afterwards.
        llvm::IntegerType* intType = llvm::IntegerType::get( module.llvmModule->getContext(), /* NumBits */ element.size * 8 );
        llvm::Constant* specializedValue          = llvm::Constant::getNullValue( intType );
        llvm::Constant* partialSpecializationMask = llvm::ConstantExpr::getNeg( specializedValue );

        // Find specialization
        for( size_t index = 0; index < module.compileParams.specializedLaunchParam.size(); ++index )
        {
            const CompileBoundValueEntry& specialization = module.compileParams.specializedLaunchParam[index];
            // The load's offset needs to be between the specialization's offset
            // and that specialization's size (i.e. within the specialization's
            // applicable range).

            size_t elementBegin        = element.offset;
            size_t elementEnd          = elementBegin + element.size;
            size_t specializationBegin = specialization.offset;
            size_t specializationEnd   = specializationBegin + specialization.value.size();

            // specialization        <--...
            // element        ...--->
            if( elementEnd <= specializationBegin )
                break;  // stop looking as there are no more candidates

            // specialization ...--->
            // element               <--...
            if( specializationEnd <= elementBegin )
                continue;


            // Callwlate start index of the value to extract from the specialization's byte array...
            int valStart = elementBegin - specializationBegin;
            // ...and the end index.
            int valEnd = valStart + element.size;

            // Handle partially specialized loads.
            // These two hold the start and the end of the range within the loaded value
            // where the specialized bytes will be inserted.
            int partialMaskStart = 0;
            int partialMaskEnd   = 0;
            if( elementBegin < specializationBegin )  // specializationBegin < elementEnd
            {
                // specialization     <--...
                // element         <-------->
                // Partial specialization of the load. Start index of the value in the byte array is 0
                // since the load started before the specialization.
                valStart = 0;
                // The number of bytes that are specialized is either up to the end of the element
                // or the number of specialized bytes available (whichever one is smaller).
                size_t numValues = elementEnd - specializationBegin;
                valEnd           = std::min( specialization.value.size(), numValues );

                // Callwlate at which position in the loaded value the specialized value will be inserted.
                partialMaskStart      = specializationBegin - elementBegin;
                partialMaskEnd        = partialMaskStart + valEnd;
                partialSpecialization = true;
            }
            else if( specializationEnd < elementEnd )  // specializationBegin <= elementBegin
            {
                // specialization      ..->
                // element             <-------->
                // Partial specialization of the load with the load "continuing" after the specialization.
                // We need to extract up to the end of the byte array.
                valEnd = specialization.value.size();

                // Callwlate at which position in the loaded value the specialized value will be inserted.
                // specializationBegin <= elementBegin, so the specialized value starts at the beginning of the loaded value.
                partialMaskStart = 0;
                // The number of specialized values goes up to the size of the specialization.
                partialMaskEnd        = valEnd - valStart;
                partialSpecialization = true;
            } else {
                if( partialSpecialization )
                {
                    errDetails.m_compilerFeedback << "Error: found complete specialization of load which is already "
                                                     "partially specialized. Validation failed.\n";
                    // This is an internal error since validation should have caught the overlap.
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                    break;
                }
            }

            llvm::Constant* nextPart = makeConstantInt( specialization.value, valStart, valEnd, partialMaskStart * 8,
                                                     element.size, module );
            specializedValue = llvm::ConstantExpr::getOr( specializedValue, nextPart );

            if( partialSpecialization )
            {
                // Build the mask for the partial specialization. This is used to zero out the bytes of the loaded
                // value which will be replaced by the specialized value in createPartiallySpecializedValue.
                std::vector<char> temp( element.size, char( 0xFF ) );
                for( int i = partialMaskStart; i < partialMaskEnd; ++i )
                    temp[i] = char( 0 );
                llvm::Constant* nextMaskPart = makeConstantInt( temp, 0, element.size, 0, element.size, module );
                partialSpecializationMask = llvm::ConstantExpr::getOr( partialSpecializationMask, nextMaskPart );
            }

            if( !specializationFound )
            {
                element.specialization = &specialization;
                specializationFound    = true;
                if( !partialSpecialization )
                    break;
            }
            else
            {
                // There was already a previous partial specialization found for this element.
                // Record additional partial specializations of this load for information output.
                // Reuse the subElements vector of the LoadElement which is normally used for struct specializations.
                element.subElements.push_back( LoadElement( 0, 0, nullptr, nullptr, &specialization ) );
                ++element.specializedSubElements;
            }
        }

        if( specializationFound )
        {
            ++elementsSpecialized;
            if( element.type->isPointerTy() )
            {
                specializedValue = llvm::ConstantExpr::getIntToPtr( specializedValue, element.type );
            }
            else
            {
                specializedValue = llvm::ConstantExpr::getBitCast( specializedValue, element.type );
            }

            element.specializedValue = specializedValue;
            if( partialSpecialization )
                element.partialSpecializationMask = partialSpecializationMask;
        }
        if( (OptixResult)result )
            break;
    }
    return result;
}

static OptixResult specializeLaunchParam( CompilationUnit& module, std::set<llvm::Function*>& specializedFunctions, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    // Start at global for launch param
    llvm::GlobalVariable* var = module.llvmModule->getNamedGlobal( module.compileParams.pipelineLaunchParamsVariableName );
    if( !var )
        return result;
    if( module.compileParams.specializedLaunchParam.empty() )
        return result;

    llvm::DataLayout DL( module.llvmModule );

    if( OptixResult res = validateSpecializations( module.compileParams.specializedLaunchParam,
                                                   DL.getTypeStoreSize( var->getType()->getPointerElementType() ), errDetails ) )
        return res;


    // care about launchParam + offset.  load val; where val = v1 + offset, v1 = lp + offset, etc.

    // initial conditions are launchParam + 0.

    std::vector<PointerUse> pointerUses;
    pointerUses.reserve( 16 );

    pointerUses.push_back( {var, 0} );

    // Find each use, and find loads and adds with constants
    while( !pointerUses.empty() )
    {
        //PointerUse pointerUse = pointerUses.pop_back_val();
        PointerUse pointerUse = pointerUses.back();
        pointerUses.pop_back();
        for( llvm::Value::user_iterator UI = pointerUse.pointer->user_begin(), UE = pointerUse.pointer->user_end(); UI != UE; )
        {
            llvm::Value* U = *UI++;

            if( llvm::LoadInst* L = llvm::dyn_cast<llvm::LoadInst>( U ) )
            {
                // Check the size of the load
                llvm::Type* loadType = L->getType();
                size_t      size     = DL.getTypeStoreSize( loadType );

                // It is legal to only partially specialize vectorized loads and struct loads. So, those get split up
                // in individual elements which get specialized. This also allows to pull in multiple partial
                // values from different specializations.

                std::vector<LoadElement> elements;
                if( loadType->isVectorTy() )
                {
                    llvm::VectorType* vecType   = llvm::cast<llvm::VectorType>( loadType );
                    llvm::Type*       elementTy = vecType->getElementType();
                    size_t            eltSize   = DL.getTypeStoreSize( elementTy );
                    for( size_t i = 0; i < vecType->getNumElements(); ++i )
                        elements.push_back( LoadElement( pointerUse.offset + i * eltSize, eltSize, elementTy, nullptr, nullptr ) );
                }
                else if( loadType->isStructTy() )
                {
                    llvm::StructType* structType   = llvm::cast<llvm::StructType>( loadType );
                    fillStructTypeLoadElements( elements, pointerUse.offset, structType, DL );
                }
                else
                    elements.push_back( LoadElement( pointerUse.offset, size, loadType, nullptr, nullptr ) );

                // Check if there are specializations available for the elements of this load.
                unsigned int       elementsSpecialized = 0;
                OptixResult elementChecks = findElementSpecializations(module, elements, elementsSpecialized, L, errDetails);

                // Errors were found. Stop processing this load and move on to the next.
                if( (OptixResult)elementChecks )
                {
                    result += elementChecks;
                    continue;
                }

                // No specializations found
                if( elementsSpecialized == 0 )
                    continue;

                // Replace the load with the specialized value.
                if( loadType->isVectorTy() || loadType->isStructTy() )
                {
                    result += specializeAggregateLoad(elements, pointerUse, L, elementsSpecialized, errDetails);
                }
                else
                {
                    // Replace all current uses of L with a dummy value first. Replacing L's uses
                    // after inserting the new values would also replace L in the bit twiddling
                    // if the load is only partially specialized.
                    corelib::CoreIRBuilder irb{ L };
                    llvm::Instruction* dummy = irb.CreateLoad( llvm::UndefValue::get( loadType->getPointerTo() ) );
                    L->replaceAllUsesWith( dummy );

                    if( elements.size() != 1 || elementsSpecialized != 1 )
                    {
                        errDetails.m_compilerFeedback << "Error: Unexpected split of non-vector load to specialize in "
                                                         "multiple elements.\n";
                        result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                        continue;
                    }
                    // clang-format off
                    errDetails.m_compilerFeedback
                        << "Info: Found specialization\n"
                        << "\t" << getSourceLocation(L) << "\n"
                        << "\t" << "annotation : " << elements[0].specialization->annotation << "\n"
                        << "\t" << "offset     : " << elements[0].specialization->offset << "\t"
                        << "\t" << "size       : " << elements[0].specialization->value.size() << "\n";
                    if( elements[0].partialSpecializationMask && elements[0].specializedSubElements )
                    {
                        // This load has multiple partial specializations. Print those.
                        for( const LoadElement& subElement : elements[0].subElements )
                            errDetails.m_compilerFeedback
                            << "\t" << "annotation : "<< subElement.specialization->annotation << "\n"
                            << "\t" << "offset     : " << subElement.specialization->offset << "\t"
                            << "\t" << "size       : " << subElement.specialization->value.size() << "\n";
                    }
                    errDetails.m_compilerFeedback
                        << "\t" << "load offset: " << elements[0].offset << "\t"
                        << "\t" << "load size  : " << elements[0].size << "\n";
                    // clang-format on

                    llvm::Value* newLoad = elements[0].specializedValue;

                    if( elements[0].partialSpecializationMask )
                    {
                        // The load is only partially specialized
                        corelib::CoreIRBuilder irb{ L->getNextNode() };
                        newLoad = createPartiallySpecializedValue( L, newLoad, elements[0], L, irb );
                    }
                    newLoad->takeName( L );

                    dummy->replaceAllUsesWith( newLoad );
                    dummy->eraseFromParent();
                }
                specializedFunctions.emplace( L->getParent()->getParent() );
            }
            else if( llvm::BinaryOperator* B = llvm::dyn_cast<llvm::BinaryOperator>( U ) )
            {
                if( pointerUse.pointer != B->getOperand( 0 ) && pointerUse.pointer != B->getOperand( 1 ) )
                {
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                    continue;
                }

                llvm::Instruction* insertBefore = B;

                int pointerPosition = 0;
                int offsetPosition  = 1;
                if( pointerUse.pointer != B->getOperand( pointerPosition ) )
                    std::swap( pointerPosition, offsetPosition );

                llvm::Value* offset = B->getOperand( offsetPosition );
                if( B->getOpcode() != llvm::BinaryOperator::Add && B->getOpcode() != llvm::BinaryOperator::Sub )
                {
                    errDetails.m_compilerFeedback << "Warning: launch parameters have specialization, but found an "
                                                     "operation against launch param "
                                                     "pointer that prevents further analysis and specialization."
                                                  << getSourceLocation( B ) << "\n";
                    continue;
                }
                int64_t offsetValue;
                bool    isConstant = corelib::getConstantValue( offset, offsetValue );
                if( !isConstant )
                {
                    errDetails.m_compilerFeedback << "Warning: launch parameters have specialization, but found an add "
                                                     "operator where the value isn't a constant integer."
                                                  << getSourceLocation( B ) << "\n";
                    continue;
                }

                size_t newOffset = pointerUse.offset;
                if( B->getOpcode() == llvm::BinaryOperator::Add )
                    newOffset += offsetValue;
                else
                    newOffset -= offsetValue;
                pointerUses.push_back( {B, newOffset} );
            }
            else if( llvm::ConstantExpr* C = llvm::dyn_cast<llvm::ConstantExpr>( U ) )
            {
                if( C->isCast() )
                {
                    pointerUses.push_back( {C, pointerUse.offset} );
                }
                else if( llvm::GEPOperator* GEP = llvm::dyn_cast<llvm::GEPOperator>( C ) )
                {
                    unsigned    BitWidth = DL.getPointerTypeSizeInBits( GEP->getType() );
                    llvm::APInt immOff( BitWidth, 0 );
                    if( !GEP->aclwmulateConstantOffset( DL, immOff ) )
                    {
                        errDetails.m_compilerFeedback
                            << "Warning: launch parameters used in an GEP with non-constant offset.\n";
                        continue;
                    }
                    int offset = static_cast<int>( immOff.getSExtValue() );
                    pointerUses.push_back( {GEP, pointerUse.offset + offset} );
                }
                else if( C->getOpcode() == llvm::BinaryOperator::Add || C->getOpcode() == llvm::BinaryOperator::Sub )
                {
                    int pointerPosition = 0;
                    int offsetPosition  = 1;
                    if( pointerUse.pointer != C->getOperand( pointerPosition ) )
                        std::swap( pointerPosition, offsetPosition );

                    llvm::Value* offset = C->getOperand( offsetPosition );
                    int64_t      offsetValue;
                    bool         isConstant = corelib::getConstantValue( offset, offsetValue );
                    if( !isConstant )
                    {
                        errDetails.m_compilerFeedback << "Error: expected constant value, but didn't find one\n";
                        result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                        continue;
                    }
                    size_t newOffset = pointerUse.offset;
                    if( C->getOpcode() == llvm::BinaryOperator::Add )
                        newOffset += offsetValue;
                    else
                        newOffset -= offsetValue;
                    pointerUses.push_back( {C, newOffset} );
                }
                else
                {
                    errDetails.m_compilerFeedback << "Warning: launch parameters used in an unhandled condition\n"
                                                  << C->getOpcodeName() << "\n";
                }
            }
            else if( llvm::CastInst* C = llvm::dyn_cast<llvm::CastInst>( U ) )
            {
                pointerUses.push_back( {C, pointerUse.offset} );
            }
            else if( llvm::GetElementPtrInst* GEP = llvm::dyn_cast<llvm::GetElementPtrInst>( U ) )
            {
                unsigned    BitWidth = DL.getPointerTypeSizeInBits( GEP->getType() );
                llvm::APInt immOff( BitWidth, 0 );
                if( !GEP->aclwmulateConstantOffset( DL, immOff ) )
                {
                    errDetails.m_compilerFeedback << "Warning: launch parameters have specialization, but found a GEP "
                                                     "instruction with non-constant offset.\n";
                    continue;
                }
                int offset = static_cast<int>( immOff.getSExtValue() );
                pointerUses.push_back( { GEP, pointerUse.offset + offset } );
            }
        }
    }
    return result;
}

static OptixResult lowerPrivateCompileTimeConstant( CompilationUnit&           module,
                                                    std::set<llvm::Function*>& specializedFunctions,
                                                    ErrorDetails&              errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    F = module.llvmIntrinsics[optix_private_get_compile_time_constant];
    if( !F )
        return OPTIX_SUCCESS;
    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    if( !calls.empty() )
    {
        if( !module.compileParams.isBuiltinModule )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_FUNCTION_USE,
                                          "Private functions are not supported in user code" );
        for( llvm::CallInst* CI : calls )
        {
            llvm::Value* index = CI->getOperand( 0 );

            unsigned int constantIndex;
            bool         isConstant = corelib::getConstantValue( index, constantIndex );
            if( !isConstant )
            {
                errDetails.m_compilerFeedback
                    << "Error: Requested compile time constant index isn't a constant integer: " << getSourceLocation( CI )
                    << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
                continue;
            }

            if( constantIndex >= module.compileParams.privateCompileTimeConstants.size() )
            {
                errDetails.m_compilerFeedback << "Error: Requested private compile time constant is out of bounds: " << constantIndex
                                              << " max: " << module.compileParams.privateCompileTimeConstants.size()
                                              << " : " << getSourceLocation( CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
            }

            specializedFunctions.emplace( CI->getParent()->getParent() );
            llvm::Value* value =
                llvm::ConstantInt::get( CI->getType(), module.compileParams.privateCompileTimeConstants[constantIndex] );
            CI->replaceAllUsesWith( value );
            CI->eraseFromParent();
        }
    }
    module.eraseIntrinsic( optix_private_get_compile_time_constant );
    return result;
}

static OptixResult optimizeSpecializedFunctions( CompilationUnit&                 module,
                                                 const std::set<llvm::Function*>& specializedFunctions,
                                                 ErrorDetails&                    errDetails )
{
    // Run some optimization passes to propagate the constants and remove dead code
    llvm::legacy::FunctionPassManager FPM( module.llvmModule );
    // Clean up partial load specialization bit twiddling by running InstCombine.
    //Must be first to provide optimization opportunities to subsequent passes.
    FPM.add( llvm::createInstructionCombiningPass() );
    FPM.add( llvm::createSCCPPass() );                  // Constant prop with SCCP
    FPM.add( llvm::createAggressiveDCEPass() );         // Delete dead instructions
    FPM.add( llvm::createCFGSimplificationPass() );     // Merge & remove BBs
    FPM.doInitialization();
    for( llvm::Function* F : specializedFunctions )
    {
        FPM.run( *F );
    }
    FPM.doFinalization();
    return OPTIX_SUCCESS;
}

static OptixResult handleNoInlineFunctions( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    for( int intrinsic = 0; intrinsic < intrinsicCount; ++intrinsic )
    {
        result += inlineAllCallersOfIntrinsic( module, static_cast<IntrinsicIndex>( intrinsic ), errDetails );
    }
    result += inlineCalleesToCallers( module, errDetails );
    return result;
}

static OptixResult extractPipelineParamsSize( InitialCompilationUnit& module, const std::string& paramName, ErrorDetails& errDetails )
{
    if( paramName.empty() )
        return OPTIX_SUCCESS;

    llvm::GlobalVariable* var = module.llvmModule->getNamedGlobal( paramName );
    if( !var )
        return OPTIX_SUCCESS;

    if( var->getType()->getPointerAddressSpace() != corelib::ADDRESS_SPACE_CONST )
    {
        errDetails.m_compilerFeedback << "Error: Launch parameter \"" << paramName
                                      << "\" is not declared in constant memory."
                                         "  Use __constant__ instead of __device__ in the declaration.\n";
        return OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER;
    }

    if( !var->getType()->isPointerTy() )
    {
        // I do not think that this can happen.
        errDetails.m_compilerFeedback << "Error: Launch parameter is not a pointer type.\n";
        return OPTIX_ERROR_ILWALID_PTX;
    }

    if( !var->isDeclaration() )
    {
        // launch parameter is not declared extern.
        llvm::Constant* initializer = var->getInitializer();
        if( initializer && !( initializer->isNullValue() || initializer->isZeroValue() ) )
        {
            // check and warn if the value is initialized
            errDetails.m_compilerFeedback << "Warning: Launch parameter \"" << paramName
                                          << "\" is initialized to the value \"" << llvmToString( initializer )
                                          << "\" in the module. Initialization will be dropped.\n";
        }

        // Replace with declaration.
        llvm::GlobalVariable* gNew =
            new llvm::GlobalVariable( *module.llvmModule, var->getType()->getPointerElementType(), var->isConstant(),
                                      llvm::GlobalValue::ExternalLinkage, nullptr, "", var, var->getThreadLocalMode(),
                                      corelib::ADDRESS_SPACE_CONST, var->isExternallyInitialized() );
        gNew->takeName( var );
        gNew->setAlignment( var->getAlignment() );
        var->replaceAllUsesWith( gNew );
        var->eraseFromParent();
        var = gNew;
    }

    if( var->hasAtLeastLocalUnnamedAddr() )
        var->setUnnamedAddr( llvm::GlobalValue::UnnamedAddr::None );

    llvm::DataLayout DL( module.llvmModule );
    size_t           size = DL.getTypeAllocSize( var->getType()->getPointerElementType() );
    module.optixModule->setPipelineParamsSize( size );
    errDetails.m_compilerFeedback << "Info: Pipeline parameter \"" << paramName << "\" size is " << size << " bytes\n";

    return OPTIX_SUCCESS;
}

using SemanticTypeCheckFunction = OptixResult ( * )( llvm::CallInst*, const InternalCompileParameters&, ErrorDetails&, const std::string& );

// If the given boolean pointer is non-null, it is set to true if the function is replaced (it is not assigned otherwise,
// allowing aclwmulation of results).
static OptixResult replaceFunctionUses( CompilationUnit&          module,
                                        IntrinsicIndex            intrinsicIndex,
                                        const std::string&        toFunctionName,
                                        SemanticTypeCheckFunction checkFunction,
                                        ErrorDetails&             errDetails,
                                        bool*                     replacedUses = nullptr )
{
    llvm::Function* fromFunction = module.llvmIntrinsics[intrinsicIndex];
    if( !fromFunction )
        return OPTIX_SUCCESS;
    llvm::Function* toFunction =
        corelib::insertOrCreateFunction( module.llvmModule, toFunctionName, fromFunction->getFunctionType() );
    if( toFunction->getType() != llvm::PointerType::getUnqual( fromFunction->getFunctionType() ) )
    {
        errDetails.m_compilerFeedback << "Error: Type mismatch for functions " << intrinsicName( intrinsicIndex )
                                      << " and " << toFunctionName
                                      << ". Expected: " << llvmToString( fromFunction->getFunctionType() )
                                      << ", found: " << llvmToString( toFunction->getFunctionType() ) << "\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }
    OptixResultOneShot result;
    if( checkFunction )
    {
        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( fromFunction );
        for( llvm::CallInst* CI : calls )
        {
            result += checkFunction( CI, module.compileParams, errDetails, apiName( intrinsicIndex ) );
        }
    }

    // If non-null, set result parameter to true if the function is replaced
    // (it is not assigned otherwise, allowing aclwmulation of results).
    if( replacedUses && fromFunction->user_begin() != fromFunction->user_end() )
    {
        *replacedUses = true;
    }

    fromFunction->replaceAllUsesWith( toFunction );
    module.eraseIntrinsic( intrinsicIndex );
    return result;
}

static OptixResult doFunctionReplacements( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    result += replaceFunctionUses( module, optix_get_launch_index_x, "lw.rt.read.launch.idx.x", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_launch_index_y, "lw.rt.read.launch.idx.y", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_launch_index_z, "lw.rt.read.launch.idx.z", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_launch_dimension_x, "lw.rt.read.launch.dim.x", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_launch_dimension_y, "lw.rt.read.launch.dim.y", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_launch_dimension_z, "lw.rt.read.launch.dim.z", nullptr, errDetails );

    result += replaceFunctionUses( module, optix_read_instance_id, "lw.rt.read.instance.id", isGetInstanceIdLegal, errDetails );
    result += replaceFunctionUses( module, optix_read_instance_idx, "lw.rt.read.instance.idx", isGetInstanceIdxLegal, errDetails );
    result += replaceFunctionUses( module, optix_read_sbt_gas_idx, "lw.rt.read.geometry.idx", isGetSbtGasIndexLegal, errDetails );

    result += replaceFunctionUses( module, optix_ignore_intersection, "lw.rt.ignore.intersection",
                                   isIgnoreIntersectionLegal, errDetails );
    result += replaceFunctionUses( module, optix_terminate_ray, "lw.rt.terminate.ray", isTerminateRayLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_ray_tmin, "lw.rt.read.ray.tmin", isRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_ray_tmax, "lw.rt.read.ray.tmax", isRayAccessLegal, errDetails );

    result += replaceFunctionUses( module, optix_get_world_ray_origin_x, "lw.rt.read.world.ray.origin.x",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_world_ray_origin_y, "lw.rt.read.world.ray.origin.y",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_world_ray_origin_z, "lw.rt.read.world.ray.origin.z",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_world_ray_direction_x, "lw.rt.read.world.ray.direction.x",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_world_ray_direction_y, "lw.rt.read.world.ray.direction.y",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_world_ray_direction_z, "lw.rt.read.world.ray.direction.z",
                                   isWorldRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_origin_x, "lw.rt.read.object.ray.origin.x",
                                   isObjectRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_origin_y, "lw.rt.read.object.ray.origin.y",
                                   isObjectRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_origin_z, "lw.rt.read.object.ray.origin.z",
                                   isObjectRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_direction_x, "lw.rt.read.object.ray.direction.x",
                                   isObjectRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_direction_y, "lw.rt.read.object.ray.direction.y",
                                   isObjectRayAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_object_ray_direction_z, "lw.rt.read.object.ray.direction.z",
                                   isObjectRayAccessLegal, errDetails );

    result += replaceFunctionUses( module, optix_get_transform_list_size, "lw.rt.read.transform.list.size",
                                   isTransformListAccessLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_transform_list_handle, "lw.rt.read.transform.list.traversable",
                                   isTransformListAccessLegal, errDetails );

    result += replaceFunctionUses( module, optix_get_transform_type_from_handle,
                                   "lw.rt.read.transform.type.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_instance_id_from_handle, "lw.rt.read.instance.id.from.traversable",
                                   nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_instance_child_from_handle,
                                   "lw.rt.read.instance.child.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_instance_transform_from_handle,
                                   "lw.rt.read.instance.transform.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_instance_ilwerse_transform_from_handle,
                                   "lw.rt.read.instance.ilwerse.transform.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_static_transform_from_handle,
                                   "lw.rt.read.static.transform.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_matrix_motion_transform_from_handle,
                                   "lw.rt.read.matrix.motion.transform.from.traversable", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_srt_motion_transform_from_handle,
                                   "lw.rt.read.srt.motion.transform.from.traversable", nullptr, errDetails );

    result += replaceFunctionUses( module, optix_get_gas_ptr, "lw.rt.read.blas.ptr", nullptr, errDetails );

    result += replaceFunctionUses( module, optix_get_gas_traversable_handle, "lw.rt.read.blas.traversable",
                                   isGetGASTraversableLegal, errDetails );

    result += replaceFunctionUses( module, optix_get_instance_traversable_from_ias, "lw.rt.read.instance.traversable",
                                   nullptr, errDetails );

    result += replaceFunctionUses( module, optix_get_triangle_vertex_data, "lw.rt.read.triangle.vertex.data", nullptr, errDetails );

    result += replaceFunctionUses( module, optix_get_gas_motion_time_begin, "lw.rt.read.blas.motion.time.begin", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_gas_motion_time_end, "lw.rt.read.blas.motion.time.end", nullptr, errDetails );
    result += replaceFunctionUses( module, optix_get_gas_motion_step_count, "lw.rt.read.blas.motion.step.count", nullptr, errDetails );

    result += replaceFunctionUses( module, optix_get_exception_code, "lw.rt.read.exception.code", isGetExceptionCodeLegal, errDetails );

    result += replaceFunctionUses( module, optix_get_exception_ilwalid_traversable,
                                   "lw.rt.read.exception.invalid.traversable", isGetExceptionDetailLegal, errDetails );
    result += replaceFunctionUses( module, optix_get_exception_ilwalid_sbt_offset,
                                   "lw.rt.read.exception.invalid.sbt.offset", isGetExceptionDetailLegal, errDetails );

    bool usedTextureIntrinsics = false;
    result += replaceFunctionUses( module, optix_tex_footprint_2d_v2, "_lw_optix_tex_footprint_2d", nullptr, errDetails,
                                   &usedTextureIntrinsics );
    result += replaceFunctionUses( module, optix_tex_footprint_2d_grad_v2, "_lw_optix_tex_footprint_2d_grad", nullptr,
                                   errDetails, &usedTextureIntrinsics );
    result += replaceFunctionUses( module, optix_tex_footprint_2d_lod_v2, "_lw_optix_tex_footprint_2d_lod", nullptr,
                                   errDetails, &usedTextureIntrinsics );
    if( usedTextureIntrinsics )
        module.subModule->setUsesTextureIntrinsic();

    // lower internal intrinsics. these are used exclusively to speed up the builtin lwrve/sphere intersectors.
    result += replaceFunctionUses( module, optix_read_prim_va, "lw.rt.read.prim.va", isGetPrimitiveIndexLegal, errDetails );
    result += replaceFunctionUses( module, optix_read_key_time, "lw.rt.read.current.key.time", isGetPrimitiveIndexLegal, errDetails );

    return result;
}

static OptixResult replace( llvm::Module* llvmModule, const std::string& fromFunctionName, const std::string& toFunctionName, ErrorDetails& errDetails )
{
    llvm::Function* fromFunction = llvmModule->getFunction( fromFunctionName );
    if( !fromFunction )
        return OPTIX_SUCCESS;
    llvm::Function* toFunction = corelib::insertOrCreateFunction( llvmModule, toFunctionName, fromFunction->getFunctionType() );
    if( toFunction->getType() != llvm::PointerType::getUnqual( fromFunction->getFunctionType() ) )
    {
        errDetails.m_compilerFeedback << "Error: Type mismatch for functions " << fromFunctionName << " and "
                                      << toFunctionName << ". Expected: " << llvmToString( fromFunction->getFunctionType() )
                                      << ", found: " << llvmToString( toFunction->getFunctionType() ) << "\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }

    fromFunction->replaceAllUsesWith( toFunction );
    return OPTIX_SUCCESS;
}

static OptixResult replaceWithValueNull( CompilationUnit& module, llvm::Function* fromFunction, ErrorDetails& errDetails )
{
    llvm::Value* Null = llvm::Constant::getIntegerValue( fromFunction->getReturnType(), llvm::APInt( 32, 0, true ) );
    for( llvm::CallInst* call : corelib::getCallsToFunction( fromFunction ) )
    {
        call->replaceAllUsesWith( Null );
        Null->takeName( call );
        call->eraseFromParent();
    }
    if( fromFunction->use_empty() )
        fromFunction->eraseFromParent();
    return OPTIX_SUCCESS;
}

static OptixResult replaceWithValueOne( CompilationUnit& module, llvm::Function* fromFunction, ErrorDetails& errDetails )
{
    llvm::Value* One = llvm::Constant::getIntegerValue( fromFunction->getReturnType(), llvm::APInt( 32, 1, true ) );
    for( llvm::CallInst* call : corelib::getCallsToFunction( fromFunction ) )
    {
        call->replaceAllUsesWith( One );
        One->takeName( call );
        call->eraseFromParent();
    }

    if( fromFunction->use_empty() )
        fromFunction->eraseFromParent();

    return OPTIX_SUCCESS;
}

// Pass to handle all accesses to special PTX registers. Note that due to supporting both PTX and optixir input,
// we have to handle both optix.lwvm.read.ptx.sreg.XXX and llvm.lwvm.read.ptx.sreg.XXX variants.
static OptixResult rewriteSpecialRegisters( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    // simple replacements using pairs of (original name, new name)
    std::vector<std::pair<std::string, std::string>> functionReplacements = {
        std::make_pair( "optix.lwvm.read.ptx.sreg.tid.x", "lw.rt.read.launch.idx.x" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.tid.y", "lw.rt.read.launch.idx.y" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.tid.z", "lw.rt.read.launch.idx.z" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.tid", "lw.rt.read.launch.idx" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.ntid.x", "lw.rt.read.launch.dim.x" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.ntid.y", "lw.rt.read.launch.dim.y" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.ntid.z", "lw.rt.read.launch.dim.z" ),
        std::make_pair( "optix.lwvm.read.ptx.sreg.ntid", "lw.rt.read.launch.dim" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.tid.x", "lw.rt.read.launch.idx.x" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.tid.y", "lw.rt.read.launch.idx.y" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.tid.z", "lw.rt.read.launch.idx.z" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.tid", "lw.rt.read.launch.idx" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.ntid.x", "lw.rt.read.launch.dim.x" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.ntid.y", "lw.rt.read.launch.dim.y" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.ntid.z", "lw.rt.read.launch.dim.z" ),
        std::make_pair( "llvm.lwvm.read.ptx.sreg.ntid", "lw.rt.read.launch.dim" ),
    };
    for( const auto& p : functionReplacements )
    {
        // doing the inlining first avoids potentially issuing error msgs with rtcore intrinsic names
        result += inlineAllCallersOfSRegIntrinsic( module.llvmModule, module.compileParams, p.first, errDetails );
        result += replace( module.llvmModule, p.first, p.second, errDetails );
    }

    // replace %ctaid variants by always returning 0
    std::vector<std::string> fromFunctionNames = {
        "optix.lwvm.read.ptx.sreg.ctaid.x", "optix.lwvm.read.ptx.sreg.ctaid.y", "optix.lwvm.read.ptx.sreg.ctaid.z",
        "optix.lwvm.read.ptx.sreg.ctaid",   "llvm.lwvm.read.ptx.sreg.ctaid.x",  "llvm.lwvm.read.ptx.sreg.ctaid.y",
        "llvm.lwvm.read.ptx.sreg.ctaid.z",  "llvm.lwvm.read.ptx.sreg.ctaid",
    };
    for( const std::string& fromFunctionName : fromFunctionNames )
    {
        llvm::Function* fromFunction = module.llvmModule->getFunction( fromFunctionName );
        if( fromFunction )
            result += replaceWithValueNull( module, fromFunction, errDetails );
    }
    // replace %nctaid variants by always returning 1
    fromFunctionNames = { "optix.lwvm.read.ptx.sreg.nctaid.x", "optix.lwvm.read.ptx.sreg.nctaid.y",
                          "optix.lwvm.read.ptx.sreg.nctaid.z", "optix.lwvm.read.ptx.sreg.nctaid",
                          "llvm.lwvm.read.ptx.sreg.nctaid.x",  "llvm.lwvm.read.ptx.sreg.nctaid.y",
                          "llvm.lwvm.read.ptx.sreg.nctaid.z",  "llvm.lwvm.read.ptx.sreg.nctaid" };
    for( const std::string& fromFunctionName : fromFunctionNames )
    {
        llvm::Function* fromFunction = module.llvmModule->getFunction( fromFunctionName );
        if( fromFunction )
            result += replaceWithValueOne( module, fromFunction, errDetails );
    }

    // catch early unsupported sregs
    std::vector<std::string> unsupportedFunctionNames = {
        "optix.lwvm.read.ptx.sreg.pm0",
        "optix.lwvm.read.ptx.sreg.pm1",
        "optix.lwvm.read.ptx.sreg.pm2",
        "optix.lwvm.read.ptx.sreg.pm3",
        "optix.lwvm.read.ptx.sreg.pm4",
        "optix.lwvm.read.ptx.sreg.pm5",
        "optix.lwvm.read.ptx.sreg.pm6",
        "optix.lwvm.read.ptx.sreg.pm7",
        "optix.lwvm.read.ptx.sreg.pm0_64",
        "optix.lwvm.read.ptx.sreg.pm1_64",
        "optix.lwvm.read.ptx.sreg.pm2_64",
        "optix.lwvm.read.ptx.sreg.pm3_64",
        "optix.lwvm.read.ptx.sreg.pm4_64",
        "optix.lwvm.read.ptx.sreg.pm5_64",
        "optix.lwvm.read.ptx.sreg.pm6_64",
        "optix.lwvm.read.ptx.sreg.pm7_64",
        "optix.lwvm.read.ptx.sreg.elwreg0",
        "optix.lwvm.read.ptx.sreg.elwreg1",
        "optix.lwvm.read.ptx.sreg.elwreg2",
        "optix.lwvm.read.ptx.sreg.elwreg3",
        "optix.lwvm.read.ptx.sreg.elwreg4",
        "optix.lwvm.read.ptx.sreg.elwreg5",
        "optix.lwvm.read.ptx.sreg.elwreg6",
        "optix.lwvm.read.ptx.sreg.elwreg7",
        "optix.lwvm.read.ptx.sreg.elwreg8",
        "optix.lwvm.read.ptx.sreg.elwreg9",
        "optix.lwvm.read.ptx.sreg.elwreg10",
        "optix.lwvm.read.ptx.sreg.elwreg11",
        "optix.lwvm.read.ptx.sreg.elwreg12",
        "optix.lwvm.read.ptx.sreg.elwreg13",
        "optix.lwvm.read.ptx.sreg.elwreg14",
        "optix.lwvm.read.ptx.sreg.elwreg15",
        "optix.lwvm.read.ptx.sreg.elwreg16",
        "optix.lwvm.read.ptx.sreg.elwreg17",
        "optix.lwvm.read.ptx.sreg.elwreg18",
        "optix.lwvm.read.ptx.sreg.elwreg19",
        "optix.lwvm.read.ptx.sreg.elwreg20",
        "optix.lwvm.read.ptx.sreg.elwreg21",
        "optix.lwvm.read.ptx.sreg.elwreg22",
        "optix.lwvm.read.ptx.sreg.elwreg23",
        "optix.lwvm.read.ptx.sreg.elwreg24",
        "optix.lwvm.read.ptx.sreg.elwreg25",
        "optix.lwvm.read.ptx.sreg.elwreg26",
        "optix.lwvm.read.ptx.sreg.elwreg27",
        "optix.lwvm.read.ptx.sreg.elwreg28",
        "optix.lwvm.read.ptx.sreg.elwreg29",
        "optix.lwvm.read.ptx.sreg.elwreg30",
        "optix.lwvm.read.ptx.sreg.elwreg31",
        "optix.lwvm.read.ptx.sreg.total_smem_size",
        "optix.lwvm.read.ptx.sreg.dynamic_smem_size",
        "optix.lwvm.read.ptx.sreg.reserved_smem_offset_begin",
        "optix.lwvm.read.ptx.sreg.reserved_smem_offset_end",
        "optix.lwvm.read.ptx.sreg.reserved_smem_offset_cap",
        "optix.lwvm.read.ptx.sreg.reserved_smem_offset_<2>",
        "llvm.lwvm.read.ptx.sreg.pm0",
        "llvm.lwvm.read.ptx.sreg.pm1",
        "llvm.lwvm.read.ptx.sreg.pm2",
        "llvm.lwvm.read.ptx.sreg.pm3",
        "llvm.lwvm.read.ptx.sreg.pm4",
        "llvm.lwvm.read.ptx.sreg.pm5",
        "llvm.lwvm.read.ptx.sreg.pm6",
        "llvm.lwvm.read.ptx.sreg.pm7",
        "llvm.lwvm.read.ptx.sreg.pm0_64",
        "llvm.lwvm.read.ptx.sreg.pm1_64",
        "llvm.lwvm.read.ptx.sreg.pm2_64",
        "llvm.lwvm.read.ptx.sreg.pm3_64",
        "llvm.lwvm.read.ptx.sreg.pm4_64",
        "llvm.lwvm.read.ptx.sreg.pm5_64",
        "llvm.lwvm.read.ptx.sreg.pm6_64",
        "llvm.lwvm.read.ptx.sreg.pm7_64",
        "llvm.lwvm.read.ptx.sreg.elwreg0",
        "llvm.lwvm.read.ptx.sreg.elwreg1",
        "llvm.lwvm.read.ptx.sreg.elwreg2",
        "llvm.lwvm.read.ptx.sreg.elwreg3",
        "llvm.lwvm.read.ptx.sreg.elwreg4",
        "llvm.lwvm.read.ptx.sreg.elwreg5",
        "llvm.lwvm.read.ptx.sreg.elwreg6",
        "llvm.lwvm.read.ptx.sreg.elwreg7",
        "llvm.lwvm.read.ptx.sreg.elwreg8",
        "llvm.lwvm.read.ptx.sreg.elwreg9",
        "llvm.lwvm.read.ptx.sreg.elwreg10",
        "llvm.lwvm.read.ptx.sreg.elwreg11",
        "llvm.lwvm.read.ptx.sreg.elwreg12",
        "llvm.lwvm.read.ptx.sreg.elwreg13",
        "llvm.lwvm.read.ptx.sreg.elwreg14",
        "llvm.lwvm.read.ptx.sreg.elwreg15",
        "llvm.lwvm.read.ptx.sreg.elwreg16",
        "llvm.lwvm.read.ptx.sreg.elwreg17",
        "llvm.lwvm.read.ptx.sreg.elwreg18",
        "llvm.lwvm.read.ptx.sreg.elwreg19",
        "llvm.lwvm.read.ptx.sreg.elwreg20",
        "llvm.lwvm.read.ptx.sreg.elwreg21",
        "llvm.lwvm.read.ptx.sreg.elwreg22",
        "llvm.lwvm.read.ptx.sreg.elwreg23",
        "llvm.lwvm.read.ptx.sreg.elwreg24",
        "llvm.lwvm.read.ptx.sreg.elwreg25",
        "llvm.lwvm.read.ptx.sreg.elwreg26",
        "llvm.lwvm.read.ptx.sreg.elwreg27",
        "llvm.lwvm.read.ptx.sreg.elwreg28",
        "llvm.lwvm.read.ptx.sreg.elwreg29",
        "llvm.lwvm.read.ptx.sreg.elwreg30",
        "llvm.lwvm.read.ptx.sreg.elwreg31",
        "llvm.lwvm.read.ptx.sreg.total_smem_size",
        "llvm.lwvm.read.ptx.sreg.dynamic_smem_size",
        "llvm.lwvm.read.ptx.sreg.reserved_smem_offset_begin",
        "llvm.lwvm.read.ptx.sreg.reserved_smem_offset_end",
        "llvm.lwvm.read.ptx.sreg.reserved_smem_offset_cap",
        "llvm.lwvm.read.ptx.sreg.reserved_smem_offset_<2>",
    };
    for( const std::string& unsupportedFunctionName : unsupportedFunctionNames )
    {
        llvm::Function* unsupportedFunction = module.llvmModule->getFunction( unsupportedFunctionName );
        if (unsupportedFunction)
        {
            errDetails.m_compilerFeedback << "Error: access to this special PTX register " << unsupportedFunction->getName().str() << " is not allowed\n";
            result += OPTIX_ERROR_ILWALID_PTX;
        }
    }

    return result;
}

static llvm::Value* insertExceptionSourceLocation( CompilationUnit& module, llvm::Value* details, llvm::Instruction* instruction )
{
    llvm::Value* sourceLoc = getSourceLocatiolwalue( module, instruction );
    std::pair<llvm::Value*, llvm::Value*> splitSourceLoc = corelib::createCast_i64_to_2xi32( sourceLoc, instruction );
    corelib::CoreIRBuilder irb{instruction};
    details = irb.CreateInsertValue( details, splitSourceLoc.first, RTC_NUM_EXCEPTION_DETAILS - 2 );
    details = irb.CreateInsertValue( details, splitSourceLoc.second, RTC_NUM_EXCEPTION_DETAILS - 1 );
    return details;
}

static OptixResult lowerGetBuiltinISData( CompilationUnit&              module,
                                          const unsigned int            exceptionFlags,
                                          const unsigned int            usesPrimitiveTypeFlags,
                                          const bool                    usesMotionBlur,
                                          const OptixPrimitiveTypeFlags primitiveType,
                                          IntrinsicIndex                intrinsicIndex,
                                          const char*                   runtimeFuncName,
                                          ErrorDetails&                 errDetails )
{
    llvm::Function* runtimeFunc = module.llvmModule->getFunction( runtimeFuncName );
    if( !runtimeFunc )
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;

    // set linkage to internal so runtime functions linked into each modules don't collide when linking the pipeline.
    runtimeFunc->setLinkage( llvm::GlobalValue::InternalLinkage );

    llvm::Function* intrinsicFunc = module.llvmIntrinsics[intrinsicIndex];
    if( !intrinsicFunc )
        return OPTIX_SUCCESS;

    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i64Ty       = llvm::Type::getInt64Ty( llvmContext );
    llvm::Type*        floatTy     = llvm::Type::getFloatTy( llvmContext );

    std::vector<llvm::Type*> paramTy;
    paramTy.push_back( i64Ty );  // gas traversable parameter
    llvm::FunctionType* getBlasFuncTy      = llvm::FunctionType::get( i64Ty, paramTy, false /*isVarArg*/ );
    llvm::FunctionType* getStepCountFuncTy = llvm::FunctionType::get( i32Ty, paramTy, false /*isVarArg*/ );
    llvm::FunctionType* getTimeFuncTy      = llvm::FunctionType::get( floatTy, paramTy, false /*isVarArg*/ );

    llvm::Function* getBlasFunc = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.blas.ptr", getBlasFuncTy );
    llvm::Function* getTimeBeginFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.blas.motion.time.begin", getTimeFuncTy );
    llvm::Function* getTimeEndFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.blas.motion.time.end", getTimeFuncTy );
    llvm::Function* getStepCountFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.blas.motion.step.count", getStepCountFuncTy );

    auto calls = corelib::getCallsToFunction( intrinsicFunc );

    bool primitiveTypeIsSupported = ( primitiveType & usesPrimitiveTypeFlags ) == primitiveType;

    if( !calls.empty() )
    {
        for( llvm::CallInst* CI : calls )
        {
            llvm::Value* newValue = llvm::UndefValue::get( CI->getType() );

            if( !primitiveTypeIsSupported )
            {
                if( exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG )
                {
                    // generate an exception when debug exceptions are enabled
                    llvm::Value* details = llvm::UndefValue::get( llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS ) );
                    details              = insertExceptionSourceLocation( module, details, CI );
                    generateThrowOptixException( module, OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE, details, CI, errDetails );
                }
                else
                {
                    // optimize away all code unconditionally leading up to the unsupported call
                    corelib::detachReachableBasicBlock( CI->getParent() );
                }
            }
            else
            {
                corelib::CoreIRBuilder irb{CI};

                llvm::Value* gas         = CI->getOperand( 0 );
                llvm::Value* primIdx     = CI->getOperand( 1 );
                llvm::Value* sbtGASIndex = CI->getOperand( 2 );
                llvm::Value* time        = CI->getOperand( 3 );

                llvm::Value* bvhPtr = irb.CreateCall( getBlasFunc, gas );

                llvm::Value *beginTime, *endTime, *stepCount;
                if( usesMotionBlur )
                {
                    beginTime = irb.CreateCall( getTimeBeginFunc, gas );
                    endTime   = irb.CreateCall( getTimeEndFunc, gas );
                    stepCount = irb.CreateCall( getStepCountFunc, gas );
                }
                else
                {
                    // we replace motion parameters with compile constants to help the compiler optimize motion code away.
                    llvm::Value* zero = llvm::ConstantFP::get( floatTy, 0.f );

                    beginTime = zero;
                    endTime   = zero;
                    stepCount = irb.getInt32( 1 );
                }

                llvm::Value* args[] = {bvhPtr, primIdx, sbtGASIndex, stepCount, beginTime, endTime, time};
                llvm::Value* ret    = irb.CreateCall( runtimeFunc, args, "" );

                // colwert return value from vector type to array type
                const int numVectorElems = ret->getType()->getVectorNumElements();
                const int numArrayElems  = CI->getType()->getStructNumElements();

                for( unsigned int i = 0; i < numVectorElems; ++i )
                {
                    llvm::Value* elemVal = irb.CreateExtractElement( ret, irb.getInt32( i ) );
                    newValue             = irb.CreateInsertValue( newValue, elemVal, i );
                }
            }

            CI->replaceAllUsesWith( newValue );
            CI->eraseFromParent();
        }
    }

    return OPTIX_SUCCESS;
}

static OptixResult lowerGetSBTData( CompilationUnit& module, int headerSize, ErrorDetails& errDetails )
{
    llvm::Function* f = module.llvmIntrinsics[optix_get_sbt_data_ptr_64];
    if( !f )
        return OPTIX_SUCCESS;
    llvm::Function* rtcFunction =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.sbt.data.ptr", f->getFunctionType() );

    OptixResultOneShot result;

    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        result += isGetSbtDataPtrLegal( CI, module.compileParams, errDetails, apiName( optix_get_sbt_data_ptr_64 ) );

        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           sbtPtr = irb.CreateCall( rtcFunction );
        int                    diff   = OPTIX_SBT_RECORD_HEADER_SIZE - headerSize;
        // No need to check diff is non-negative, since we already checked this when
        // creating the DeviceContext.

        sbtPtr = irb.CreateAdd( sbtPtr, llvm::ConstantInt::get( sbtPtr->getType(), diff ) );
        sbtPtr->takeName( CI );
        CI->replaceAllUsesWith( sbtPtr );
        CI->eraseFromParent();
    }

    module.eraseIntrinsic( optix_get_sbt_data_ptr_64 );

    return result;
}

static OptixResult lowerGetPayloadLegacy( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    IntrinsicIndex toHandle[] = { optix_get_payload_0, optix_get_payload_1, optix_get_payload_2, optix_get_payload_3,
                                  optix_get_payload_4, optix_get_payload_5, optix_get_payload_6, optix_get_payload_7 };

    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function*              F     = module.llvmIntrinsics[intrinsicIdx];
        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        // Legacy payload accessors where deprecated (ABI 46) before payload types (ABI 47) where introduced.
        // There should only be exactly one payload type.
        if( module.compileParams.payloadTypes.size() != 1 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "There should be exactly one payload type when using legacy payload accesses" );
        const int numPayloadValues = (int)module.compileParams.payloadTypes.front().semantics.size();

        int payloadIndex = intrinsicIdx - optix_get_payload_0;
        if( numPayloadValues <= payloadIndex )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << ( payloadIndex + 1 ) << " payload values in optixGetPayload but only " << numPayloadValues
                << " are configured in the pipeline: " << getSourceLocation( module.llvmModule, calls.front() ) << "\n";
            result += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
        }
        module.usedPayloadValues = std::max( module.usedPayloadValues, payloadIndex + 1 );

        llvm::Type*  i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Value* idx   = llvm::ConstantInt::get( i32Ty, payloadIndex );

        llvm::Type*         argTypes[] = { i32Ty };
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
        llvm::Function* rtcGetPayloadFunc = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.payload.i32", funcTy );
        for( llvm::CallInst* CI : calls )
        {
            result += isPayloadAccessLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            corelib::CoreIRBuilder irb{ CI };
            llvm::Value*           plValue = irb.CreateCall( rtcGetPayloadFunc, idx );
            CI->replaceAllUsesWith( plValue );
            CI->eraseFromParent();
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        module.eraseIntrinsic( intrinsicIdx );
    }

    return result;
}

static OptixResult lowerGetPayload( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot results;
    llvm::Function*    F = module.llvmIntrinsics[optix_get_payload];
    if( !F )
        return OPTIX_SUCCESS;

    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    if( !calls.empty() )
    {
        llvm::Type*         i32Ty      = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Type*         argTypes[] = { i32Ty };
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
        llvm::Function* rtcGetPayloadFunc = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.payload.i32", funcTy );

        for( llvm::CallInst* CI : calls )
        {
            results += isPayloadAccessLegal( CI, module.compileParams, errDetails, apiName( optix_get_payload ) );

            llvm::Value* index = CI->getOperand( 0 );

            int32_t payloadIndex;
            bool    isConstant = corelib::getConstantValue( index, payloadIndex );
            if( !isConstant )
            {
                errDetails.m_compilerFeedback << "Error: Requested payload values in optixGetPayload but "
                                              << "payload index operator isn't a constant integer."
                                              << getSourceLocation( CI ) << "\n";
                results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                continue;
            }

            llvm::Function* inFunction = CI->getParent()->getParent();

            // get semantics for the current shader type
            const optix_exp::SemanticType stype =
                getSemanticTypeForFunction( inFunction, module.compileParams.noinlineEnabled, module.compileParams.disableNoinlineFunc );

            // get mask with supported payload types
            unsigned int supportedPayloadTypesMask = 0u;
            if( OptixResult result = getPayloadTypeAnnotation( inFunction, module, stype, supportedPayloadTypesMask, errDetails ) )
                return result;

            OptixPayloadSemantics readSemanticsFlag;
            switch( stype )
            {
            case ST_MISS:         readSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_MS_READ; break;
            case ST_CLOSESTHIT:   readSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_CH_READ; break;
            case ST_ANYHIT:       readSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_AH_READ; break;
            case ST_INTERSECTION: readSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_IS_READ; break;
            default:
                ; // isPayloadAccessLegal above should have triggered an error
            }

            // find all payload types supported in this function
            for( unsigned int t = 0; t < module.compileParams.payloadTypes.size(); ++t )
            {
                if( ( 1u << t ) & supportedPayloadTypesMask )
                {
                    const unsigned int numPayloadValues = module.compileParams.payloadTypes[t].semantics.size();
                    if( payloadIndex >= module.compileParams.payloadTypes[t].semantics.size() )
                    {
                        errDetails.m_compilerFeedback
                            << "Error: Requested payload value " << payloadIndex << " of payload type " << t << " in optixGetPayload, but only "
                            << numPayloadValues << " values are configured in the type: " << getSourceLocation( module.llvmModule, CI ) << "\n";
                        results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                    }
                    else if( ( module.compileParams.payloadTypes[t].semantics[payloadIndex] & readSemanticsFlag ) == 0 )
                    {
                        errDetails.m_compilerFeedback
                            << "Error: Requested payload value " << payloadIndex << " of payload type " << t << " in optixGetPayload, but "
                            << "value has no read semantics: " << getSourceLocation( module.llvmModule, CI ) << "\n";
                        results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                    }
                }
            }

            module.usedPayloadValues = std::max( module.usedPayloadValues, payloadIndex + 1 );

            corelib::CoreIRBuilder irb{ CI };
            llvm::Value*           plValue = irb.CreateCall( rtcGetPayloadFunc, index );
            CI->replaceAllUsesWith( plValue );
            CI->eraseFromParent();
        }
    }

    module.eraseIntrinsic( optix_get_payload );

    return results;
}

static OptixResult lowerSetPayloadLegacy( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot           result;
    std::vector<llvm::Function*> toDelete;
    IntrinsicIndex toHandle[] = { optix_set_payload_0, optix_set_payload_1, optix_set_payload_2, optix_set_payload_3,
                                  optix_set_payload_4, optix_set_payload_5, optix_set_payload_6, optix_set_payload_7 };

    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function*              F     = module.llvmIntrinsics[intrinsicIdx];
        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        // Legacy payload accessors where deprecated (ABI 46) before payload types (ABI 47) where introduced.
        // There should only be exactly one payload type.
        if( module.compileParams.payloadTypes.size() != 1 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "There should be exactly one payload type when using legacy payload accesses" );
        const int numPayloadValues = (int)module.compileParams.payloadTypes.front().semantics.size();

        int payloadIndex = intrinsicIdx - optix_set_payload_0;
        if( numPayloadValues <= payloadIndex )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << ( payloadIndex + 1 ) << " payload values in optixSetPayload but only " << numPayloadValues
                << " are configured in the pipeline: " << getSourceLocation( module.llvmModule, calls.front() ) << "\n";
            result += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
        }
        module.usedPayloadValues = std::max( module.usedPayloadValues, payloadIndex + 1 );

        llvm::Type*  i32Ty  = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Type*  voidTy = llvm::Type::getVoidTy( module.llvmModule->getContext() );
        llvm::Value* idx    = llvm::ConstantInt::get( i32Ty, payloadIndex );

        llvm::Type*         argTypes[] = { i32Ty, i32Ty };
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( voidTy, argTypes, false );
        llvm::Function*     rtcSetPayloadFunc =
            corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.write.payload.i32", funcTy );
        for( llvm::CallInst* CI : calls )
        {
            result += isPayloadAccessLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            corelib::CoreIRBuilder irb{ CI };
            llvm::Value*           plValue = CI->getArgOperand( 0 );
            llvm::Value*           args[]  = { idx, plValue };
            plValue                        = irb.CreateCall( rtcSetPayloadFunc, args );
            CI->replaceAllUsesWith( plValue );
            CI->eraseFromParent();
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
        module.eraseIntrinsic( intrinsicIdx );

    return result;
}


static OptixResult lowerSetPayload( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot results;
    llvm::Function*    F = module.llvmIntrinsics[optix_set_payload];
    if( !F )
        return OPTIX_SUCCESS;

    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    if( !calls.empty() )
    {
        llvm::Type*         i32Ty      = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Type*         voidTy     = llvm::Type::getVoidTy( module.llvmModule->getContext() );
        llvm::Type*         argTypes[] = { i32Ty, i32Ty };
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( voidTy, argTypes, false );
        llvm::Function*     rtcSetPayloadFunc =
            corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.write.payload.i32", funcTy );

        for( llvm::CallInst* CI : calls )
        {
            results += isPayloadAccessLegal( CI, module.compileParams, errDetails, apiName( optix_set_payload ) );

            llvm::Value* index = CI->getOperand( 0 );
            llvm::Value* value = CI->getOperand( 1 );

            int32_t payloadIndex;
            bool    isConstant = corelib::getConstantValue( index, payloadIndex );
            if( !isConstant )
            {
                errDetails.m_compilerFeedback << "Error: Requested payload values in optixSetPayload but "
                                              << "payload index operator isn't a constant integer."
                                              << getSourceLocation( CI ) << "\n";
                results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                continue;
            }

            llvm::Function* inFunction = CI->getParent()->getParent();

            // get semantics for the current shader type
            const optix_exp::SemanticType stype =
                getSemanticTypeForFunction( inFunction, module.compileParams.noinlineEnabled, module.compileParams.disableNoinlineFunc );

            // get mask with supported payload types
            unsigned int supportedPayloadTypesMask = 0u;
            if( OptixResult result = getPayloadTypeAnnotation( inFunction, module, stype, supportedPayloadTypesMask, errDetails ) )
                return result;
            OptixPayloadSemantics writeSemanticsFlag;
            switch( stype )
            {
            case ST_MISS:         writeSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_MS_WRITE; break;
            case ST_CLOSESTHIT:   writeSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_CH_WRITE; break;
            case ST_ANYHIT:       writeSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_AH_WRITE; break;
            case ST_INTERSECTION: writeSemanticsFlag = OPTIX_PAYLOAD_SEMANTICS_IS_WRITE; break;
            default:
                ; // isPayloadAccessLegal above should have triggered an error
            }

            // find all payload types supported in this function
            for( unsigned int t = 0; t < module.compileParams.payloadTypes.size(); ++t )
            {
                if( ( 1u << t ) & supportedPayloadTypesMask )
                {
                    const unsigned int numPayloadValues = module.compileParams.payloadTypes[t].semantics.size();
                    if( payloadIndex >= module.compileParams.payloadTypes[t].semantics.size() )
                    {
                        errDetails.m_compilerFeedback
                            << "Error: Requested payload value " << payloadIndex << " of payload type " << t << " in optixSetPayload, but only "
                            << numPayloadValues << " values are configured in the type: " << getSourceLocation( module.llvmModule, CI ) << "\n";
                        results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                    }
                    else if( ( module.compileParams.payloadTypes[t].semantics[payloadIndex] & writeSemanticsFlag ) == 0 )
                    {
                        errDetails.m_compilerFeedback
                            << "Error: Requested payload value " << payloadIndex << " of payload type " << t << " in optixSetPayload, but "
                            << "value has no write semantics: " << getSourceLocation( module.llvmModule, CI ) << "\n";
                        results += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
                    }
                }
            }

            module.usedPayloadValues = std::max( module.usedPayloadValues, payloadIndex + 1 );

            corelib::CoreIRBuilder irb{ CI };
            llvm::Value*           plValue = irb.CreateCall( rtcSetPayloadFunc, { index, value } );
            CI->replaceAllUsesWith( plValue );
            CI->eraseFromParent();
        }
    }

    module.eraseIntrinsic( optix_set_payload );

    return results;
}


static OptixResult lowerSetPayloadTypes( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot           result;
    llvm::Function*              F     = module.llvmIntrinsics[optix_set_payload_types];
    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    for( llvm::CallInst* CI : calls )
    {
        // May only be called in functions that may access the payload
        result += isPayloadAccessLegal( CI, module.compileParams, errDetails, apiName( optix_set_payload_types ) );

        llvm::Value* typeMask = CI->getArgOperand( 0 );

        llvm::BasicBlock* BB         = CI->getParent();
        llvm::Function*   inFunction = BB->getParent();
        llvm::Module*     inModule   = inFunction->getParent();

        int maskAsInt;
        if( !corelib::getConstantValue( typeMask, maskAsInt ) )
        {
            errDetails.m_compilerFeedback << "Error: Type in " << apiName( optix_set_payload_types )
                                          << " is not a constant: " << getSourceLocation( inModule, CI ) << "\n";
            result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
            continue;
        }

        if( ( ( ( 1u << module.compileParams.payloadTypes.size() ) - 1 ) & maskAsInt ) != maskAsInt )
        {
            errDetails.m_compilerFeedback << "Error: Type in " << apiName( optix_set_payload_types )
                                          << " is not configured in the pipeline: " << getSourceLocation( inModule, CI ) << "\n";
            result += OPTIX_ERROR_ILWALID_FUNCTION_USE;
        }

        // The payload type may only be set once per function
        if( inFunction->getMetadata( "optix.payload.types" ) )
        {
            errDetails.m_compilerFeedback << "Error: " << apiName( optix_set_payload_types )
                                          << " called multiple times: " << getSourceLocation( inModule, CI ) << "\n";
            result += OPTIX_ERROR_ILWALID_FUNCTION_USE;
            continue;
        }

        // The payload must be set unconditionally.
        // Walk down the function basic blocks until either a conditional branching or the call is reached.
        llvm::BasicBlock* entryBB = &inFunction->getEntryBlock();
        while( entryBB != nullptr && entryBB != BB )
        {
            const llvm::TerminatorInst* terminator = entryBB->getTerminator();
            if( terminator->getNumSuccessors() > 1 )
            {
                errDetails.m_compilerFeedback
                    << "Error: " << apiName( optix_set_payload_types )
                    << " must be called unconditionally: " << getSourceLocation( inModule, CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_USE;
                entryBB = nullptr;  // mark error
            }
            else if( terminator->getNumSuccessors() == 0 )
            {
                errDetails.m_compilerFeedback << "Internal error()\n";
                result += OPTIX_ERROR_INTERNAL_ERROR;
                entryBB = nullptr;  // mark error
            }
            else
            {
                entryBB = terminator->getSuccessor( 0 );
            }
        }

        // check for marked error
        if( !entryBB )
            continue;

        // The lwvm_move intrinsic may "wrap" the constant int in debug builds.
        // If that is the case, we need to create an actual ConstantInt with the correct value
        // because the result of lwvm_move would not be valid for metadata.
        if( !llvm::isa<llvm::ConstantInt>( typeMask ) )
            typeMask = llvm::ConstantInt::get( llvm::Type::getInt32Ty( CI->getContext() ), maskAsInt );

        // Attach the supported types as metadata to the function.
        inFunction->setMetadata( "optix.payload.types", llvm::MDNode::get( CI->getContext(), UseValueAsMd( typeMask ) ) );

        CI->eraseFromParent();
    }

    module.eraseIntrinsic( optix_set_payload_types );

    return result;
}

static OptixResult lowerReportIntersection( CompilationUnit& module, int numAttributeValues, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    IntrinsicIndex toHandle[] = {optix_report_intersection_0, optix_report_intersection_1, optix_report_intersection_2,
                                 optix_report_intersection_3, optix_report_intersection_4, optix_report_intersection_5,
                                 optix_report_intersection_6, optix_report_intersection_7, optix_report_intersection_8};
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        int numRegister = intrinsicIdx - optix_report_intersection_0;

        if( numAttributeValues < numRegister )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << numRegister << " attribute values in " << apiName( intrinsicIdx )
                << " but only " << numAttributeValues
                << " are configured in the pipeline: " << getSourceLocation( module.llvmModule, calls.front() ) << "\n";
            result += OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS;
        }

        module.usedAttributeValues = std::max( module.usedAttributeValues, numRegister );

        // Options:
        //      1) Leave as is (only register attributes)
        //      2) Attribute access > numAttributeValues configured in the pipeline
        //         are memory attributes, effectively changing the semantics of numAttributeValues
        //         to numAttributeRegisters.
        //      3) Define internal (not exposed) cut-off, trying to access attribute
        //         beyond that value automatically become memory attribute.

        // Current POR is to do 1 and make it a hard error if you exceed the maximum
        // number of attribute values which is lwrrently 8.

        llvm::Type* i32Ty                     = llvm::Type::getInt32Ty( llvmContext );
        llvm::Type* registerAttributeStructTy = llvm::ArrayType::get( i32Ty, numRegister );
        llvm::Type* memoryAttributeStructTy   = llvm::ArrayType::get( i32Ty, 0 );
        llvm::Type* floatTy                   = llvm::Type::getFloatTy( llvmContext );
        llvm::Type* i8Ty                      = llvm::Type::getInt8Ty( llvmContext );
        llvm::Type* i1Ty                      = llvm::Type::getInt1Ty( llvmContext );
        llvm::Type* argTypes[] = {floatTy, i8Ty, i32Ty, registerAttributeStructTy, memoryAttributeStructTy, i1Ty};
        llvm::FunctionType* functionType              = llvm::FunctionType::get( i1Ty, argTypes, false );
        llvm::Function*     rtcReportIntersectionFunc = corelib::insertOrCreateFunction(
            module.llvmModule, "lw.rt.report.intersection." + corelib::getTypeName( registerAttributeStructTy ), functionType );
        for( llvm::CallInst* CI : calls )
        {
            result += isReportIntersectionLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            corelib::CoreIRBuilder irb{CI};
            llvm::Value*           hitT    = CI->getArgOperand( 0 );
            llvm::Value*           hitKind = CI->getArgOperand( 1 );
            hitKind                        = irb.CreateIntCast( hitKind, i8Ty, false );

            llvm::Value* registerAttributeStruct = llvm::UndefValue::get( registerAttributeStructTy );
            for( unsigned int i = 0, e = numRegister; i < e; ++i )
            {
                llvm::Value* attrValue = CI->getArgOperand( 2 + i );
                registerAttributeStruct =
                    irb.CreateInsertValue( registerAttributeStruct, attrValue, i, "packedAttributes" );
            }

            llvm::Value* memoryAttributeStruct = llvm::UndefValue::get( memoryAttributeStructTy );
            llvm::Value* args[]                = {hitT,
                                   hitKind,
                                   /*sbtSkip*/ llvm::ConstantInt::get( i32Ty, 0 ),
                                   registerAttributeStruct,
                                   memoryAttributeStruct,
                                   /*checkHitTInterval*/ llvm::ConstantInt::get( i1Ty, 1 )};

            llvm::Value* rtcRI = irb.CreateCall( rtcReportIntersectionFunc, args );
            rtcRI              = irb.CreateIntCast( rtcRI, i32Ty, false );
            CI->replaceAllUsesWith( rtcRI );
            CI->eraseFromParent();
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        module.eraseIntrinsic( intrinsicIdx );
    }

    return result;
}

static OptixResult lowerGetPrimitiveIndex( CompilationUnit& module, const unsigned int usesPrimitiveTypeFlags, ErrorDetails& errDetails )
{
    llvm::Function* f = module.llvmIntrinsics[optix_read_primitive_idx];
    if( !f )
        return OPTIX_SUCCESS;

    llvm::Type* i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::FunctionType* rtcPrimitiveIndexType = llvm::FunctionType::get( i32Ty, false );
    llvm::Function*     rtcPrimitiveIndexFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.primitive.idx", rtcPrimitiveIndexType );

    OptixResultOneShot result;
    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        if( OptixResult res = isGetPrimitiveIndexLegal( CI, module.compileParams, errDetails, apiName( optix_read_primitive_idx ) ) )
        {
            result += res;
            continue;
        }

        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           primIdx = irb.CreateCall( rtcPrimitiveIndexFunc );

        CI->replaceAllUsesWith( primIdx );
        CI->eraseFromParent();
    }

    return result;
}


static OptixResult lowerGetHitKind( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    f = module.llvmIntrinsics[optix_get_hit_kind];
    if( !f )
        return result;
    llvm::FunctionType* rtcFunctionType =
        llvm::FunctionType::get( llvm::Type::getInt8Ty( module.llvmModule->getContext() ), false );
    llvm::Function* rtcFunction = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.hitkind", rtcFunctionType );
    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        result += isGetHitKindLegal( CI, module.compileParams, errDetails, apiName( optix_get_hit_kind ) );

        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           hitKind = irb.CreateCall( rtcFunction );
        hitKind = irb.CreateIntCast( hitKind, llvm::Type::getInt32Ty( module.llvmModule->getContext() ), false );
        CI->replaceAllUsesWith( hitKind );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_hit_kind );

    return result;
}

static OptixResult lowerGetBuiltinBackfaceFromHitKind( CompilationUnit& module, ErrorDetails& errDetails )
{
    llvm::Function* f = module.llvmIntrinsics[optix_is_hitkind_backface];
    if( !f )
        return OPTIX_SUCCESS;

    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           hitkind  = CI->getOperand( 0 );
        llvm::Value*           backface = irb.CreateAnd( hitkind, irb.getInt32( OPTIX_HIT_KIND_BACKFACE_MASK ) );

        CI->replaceAllUsesWith( backface );
        CI->eraseFromParent();
    }

    return OPTIX_SUCCESS;
}

static OptixResult lowerGetBuiltinTypeFromHitKind( CompilationUnit&   module,
                                                   const unsigned int exceptionFlags,
                                                   const unsigned int usesPrimitiveTypeFlags,
                                                   ErrorDetails&      errDetails )
{
    llvm::Function* runtimeTypeFromHitKindFunc = module.llvmModule->getFunction( STRINGIFY( RUNTIME_BUILTIN_TYPE_FROM_HIT_KIND ) );
    if( !runtimeTypeFromHitKindFunc )
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    llvm::Function* runtimeIsTypeSupportedFunc = module.llvmModule->getFunction( STRINGIFY( RUNTIME_IS_BUILTIN_TYPE_SUPPORTED ) );
    if( !runtimeIsTypeSupportedFunc )
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;

    // set linkage to internal so runtime functions linked into each modules don't collide when linking the pipeline.
    runtimeTypeFromHitKindFunc->setLinkage( llvm::GlobalValue::InternalLinkage );
    runtimeIsTypeSupportedFunc->setLinkage( llvm::GlobalValue::InternalLinkage );

    llvm::Function* f = module.llvmIntrinsics[optix_get_primitive_type_from_hit_kind];
    if( !f )
        return OPTIX_SUCCESS;

    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        const bool debugExceptionsEnabled = ( exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG );

        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           hitkind       = CI->getOperand( 0 );
        llvm::Value*           args[]        = {hitkind};
        llvm::CallInst*        call          = irb.CreateCall( runtimeTypeFromHitKindFunc, args );
        llvm::Value*           primitiveType = call;
        llvm::CallInst*        isPrimitiveTypeSupported =
            irb.CreateCall( runtimeIsTypeSupportedFunc, {primitiveType, irb.getInt32( usesPrimitiveTypeFlags )} );

        if( debugExceptionsEnabled )
        {
            llvm::Value*       condition    = irb.CreateNot( isPrimitiveTypeSupported );
            llvm::Instruction* insertBefore = CI;
            generateConditionalThrowOptixException( module, condition, OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE,
                                                    insertBefore, nullptr, errDetails );
            irb.SetInsertPoint( insertBefore );
        }

        // set lwrve type to none if unsupported to allow code elimination for unsupported lwrve types.
        primitiveType = irb.CreateSelect( isPrimitiveTypeSupported, primitiveType, irb.getInt32( 0 ) );

        CI->replaceAllUsesWith( primitiveType );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_primitive_type_from_hit_kind );

    return OPTIX_SUCCESS;
}

static OptixResult lowerUndefinedValue( CompilationUnit& module, ErrorDetails& )
{
    llvm::Function* f = module.llvmIntrinsics[optix_undef_value];
    if( !f )
        return OPTIX_SUCCESS;
    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        llvm::Value* undefValue = llvm::UndefValue::get( llvm::Type::getInt32Ty( module.llvmModule->getContext() ) );
        CI->replaceAllUsesWith( undefValue );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_undef_value );

    return OPTIX_SUCCESS;
}

static OptixResult lowerTraceToRtcTrace( CompilationUnit&                 module,
                                         const InternalCompileParameters& compileParams,
                                         IntrinsicIndex                   intrinsicIdx,
                                         llvm::CallInst*                  CI,
                                         llvm::Value*                     payload,
                                         bool                             hasPayloadTypeOperand,
                                         ErrorDetails&                    errDetails )
{
    llvm::Module*      llvmModule  = CI->getParent()->getParent()->getParent();
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i8Ty        = llvm::Type::getInt8Ty( llvmContext );
    llvm::Type*        i16Ty       = llvm::Type::getInt16Ty( llvmContext );
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i64Ty       = llvm::Type::getInt64Ty( llvmContext );
    llvm::Type*        floatTy     = llvm::Type::getFloatTy( llvmContext );

    corelib::CoreIRBuilder irb{ CI };

    // If a payload type is provided it's the first operand to the call
    llvm::Value* typeValue =
        ( hasPayloadTypeOperand ? CI->getArgOperand( 0 ) : llvm::ConstantInt::get( i32Ty, OPTIX_PAYLOAD_TYPE_DEFAULT ) );

    int  typeIdAsInt;
    bool isConstant = corelib::getConstantValue( typeValue, typeIdAsInt );
    if( !isConstant )
    {
        errDetails.m_compilerFeedback << "Error: Type in " << apiName( intrinsicIdx )
                                      << " is not a constant: " << getSourceLocation( llvmModule, CI ) << "\n";
        return OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
    }

    if( typeIdAsInt == OPTIX_PAYLOAD_TYPE_DEFAULT )
    {
        // There is no default when multiple payload types are available,
        // forcing the user to patch all trace calls when mixing payload types.
        if( compileParams.payloadTypes.size() > 1 )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << apiName( intrinsicIdx ) << " without providing payload type identifier, "
                << "but multiple payload types are configured in OptixModuleCompileOptions:"
                << getSourceLocation( module.llvmModule, CI ) << "\n";
            return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }

        // If only a single payload type was configured in OptixPipelineCompileOptions we use it as the default.
        typeIdAsInt = OPTIX_PAYLOAD_TYPE_ID_0;
    }

    unsigned int typeIndex = ~0u;
    if( OptixResult result = getPayloadTypeIndexFromID( typeIndex, typeIdAsInt, CI, module, errDetails ) )
    {
        return result;
    }

    unsigned int argOffs        = ( hasPayloadTypeOperand ? 1 : 0 );
    llvm::Value* handle         = CI->getArgOperand( argOffs + 0 );
    llvm::Value* rayOriginX     = CI->getArgOperand( argOffs + 1 );
    llvm::Value* rayOriginY     = CI->getArgOperand( argOffs + 2 );
    llvm::Value* rayOriginZ     = CI->getArgOperand( argOffs + 3 );
    llvm::Value* rayDirectionX  = CI->getArgOperand( argOffs + 4 );
    llvm::Value* rayDirectionY  = CI->getArgOperand( argOffs + 5 );
    llvm::Value* rayDirectionZ  = CI->getArgOperand( argOffs + 6 );
    llvm::Value* tmin           = CI->getArgOperand( argOffs + 7 );
    llvm::Value* tmax           = CI->getArgOperand( argOffs + 8 );
    llvm::Value* rayTime        = CI->getArgOperand( argOffs + 9 );
    llvm::Value* visibilityMask = irb.CreateTrunc( CI->getArgOperand( argOffs + 10 ), i8Ty );
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    llvm::Value* rayFlags       = irb.CreateTrunc( CI->getArgOperand( argOffs + 11 ), i16Ty );
    // Only 13 bits of ray flags are exposed in Optix and only those should make it to rtcore
    // Note, this needs to be updated whenever new flags are added
    rayFlags                    = irb.CreateAnd( rayFlags, irb.getInt16( 0x1FFF ) );
#else
    // Ray flags only have 8 bits of input, but RTCore expects an i16.
    llvm::Value* rayFlags       = irb.CreateTrunc( CI->getArgOperand( argOffs + 11 ), i8Ty );
    rayFlags                    = irb.CreateIntCast( rayFlags, i16Ty, false );
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    llvm::Value* SBToffset      = irb.CreateTrunc( CI->getArgOperand( argOffs + 12 ), i8Ty );
    llvm::Value* SBTstride      = irb.CreateTrunc( CI->getArgOperand( argOffs + 13 ), i8Ty );
    llvm::Value* missSBTIndex   = CI->getArgOperand( argOffs + 14 );

    llvm::Type* payloadTy   = payload->getType();
    const int   payloadSize = payloadTy->getStructNumElements();

    // check if the payload count doesn't exceed the size of the specified payload type
    unsigned int numPayloadValues = module.compileParams.payloadTypes[typeIndex].semantics.size();
    if( payloadSize > numPayloadValues )
    {
        errDetails.m_compilerFeedback << "Error: " << payloadSize << " payload values specified in " << apiName( intrinsicIdx )
                                      << ", but only " << numPayloadValues << " values are configured for payload type "
                                      << typeIndex << ": " << getSourceLocation( llvmModule, CI ) << "\n";
        return OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
    }

    OptixResultOneShot result;
    result += isTraceCallLegal( CI, module.compileParams, errDetails, "optixTrace" );
    module.usedPayloadValues = std::max( module.usedPayloadValues, payloadSize );

    if( compileParams.exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG )
    {
        // Generate invalid ray exception check.
        llvm::Value* inf       = llvm::ConstantFP::getInfinity( floatTy );
        llvm::Value* zero      = llvm::ConstantFP::get( floatTy, 0.f );
        llvm::Value* condition = irb.CreateFCmpUEQ( rayOriginX, inf );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayOriginY, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayOriginZ, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayDirectionX, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayDirectionY, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayDirectionZ, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpULT( tmin, zero ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( tmin, inf ) );
        condition              = irb.CreateOr( condition, irb.CreateFCmpUEQ( tmax, inf ) );

        llvm::Value* details = llvm::UndefValue::get( llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS ) );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayOriginX, i32Ty ), 0 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayOriginY, i32Ty ), 1 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayOriginZ, i32Ty ), 2 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayDirectionX, i32Ty ), 3 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayDirectionY, i32Ty ), 4 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( rayDirectionZ, i32Ty ), 5 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( tmin, i32Ty ), 6 );
        details              = irb.CreateInsertValue( details, irb.CreateBitCast( tmax, i32Ty ), 7 );

        if( compileParams.usesMotionBlur )
        {
            condition = irb.CreateOr( condition, irb.CreateFCmpUEQ( rayTime, inf ) );
            details   = irb.CreateInsertValue( details, irb.CreateBitCast( rayTime, i32Ty ), 8 );
        }
        else
        {
            // Always return 0 for rayTime if motion blur is disabled.
            llvm::Value* time = llvm::ConstantInt::get( i32Ty, 0 );
            details           = irb.CreateInsertValue( details, time, 8 );
        }

        details = insertExceptionSourceLocation( module, details, CI );

        llvm::Instruction* insertBefore = CI;
        result += generateConditionalThrowOptixException( module, condition, OPTIX_EXCEPTION_CODE_ILWALID_RAY,
                                                          insertBefore, details, errDetails );
        irb.SetInsertPoint( insertBefore );
    }

    llvm::CallInst* rtcCall = nullptr;

    if( compileParams.usesMotionBlur )
    {
        // clang-format off
        llvm::Type* argTypes[] = {i64Ty, i16Ty, i8Ty, i8Ty, i8Ty, i32Ty,
                                floatTy, floatTy, floatTy, floatTy,
                                floatTy, floatTy, floatTy, floatTy,
                                floatTy, payloadTy};
        // clang-format on
        llvm::FunctionType* funcTy       = llvm::FunctionType::get( payloadTy, argTypes, false );
        std::string         funcName     = std::string( "lw.rt.trace.mblur." ) + std::to_string( payloadSize );
        llvm::Function*     rtcTraceFunc = corelib::insertOrCreateFunction( module.llvmModule, funcName, funcTy );

        // clang-format off
        llvm::Value* args[] = {handle, rayFlags, visibilityMask, SBToffset, SBTstride, missSBTIndex,
                                rayOriginX, rayOriginY, rayOriginZ, tmin,
                                rayDirectionX, rayDirectionY, rayDirectionZ, tmax,
                                rayTime, payload};
        // clang-format on
        rtcCall = irb.CreateCall( rtcTraceFunc, args );
    }
    else
    {
        // clang-format off
        llvm::Type* argTypes[] = {i64Ty, i16Ty, i8Ty, i8Ty, i8Ty, i32Ty,
                                floatTy, floatTy, floatTy, floatTy,
                                floatTy, floatTy, floatTy, floatTy, payloadTy};
        // clang-format on
        llvm::FunctionType* funcTy       = llvm::FunctionType::get( payloadTy, argTypes, false );
        std::string         funcName     = std::string( "lw.rt.trace." ) + std::to_string( payloadSize );
        llvm::Function*     rtcTraceFunc = corelib::insertOrCreateFunction( module.llvmModule, funcName, funcTy );

        // clang-format off
        llvm::Value* args[] = {handle, rayFlags, visibilityMask, SBToffset, SBTstride, missSBTIndex,
                                rayOriginX, rayOriginY, rayOriginZ, tmin,
                                rayDirectionX, rayDirectionY, rayDirectionZ, tmax,
                                payload};
        // clang-format on
        rtcCall = irb.CreateCall( rtcTraceFunc, args );
    }

    llvm::Type* inTy  = rtcCall->getType();
    llvm::Type* outTy = CI->getType();

    if( payloadSize > 0 )
    {
        // Pad the returned payload struct with undefs to match the return type of the lowered trace call.
        llvm::Value* outPayload = nullptr;
        if( outTy->isStructTy() )
        {
            outPayload = llvm::UndefValue::get( outTy );

            for( size_t pi = 0; pi < inTy->getStructNumElements(); pi++ )
            {
                llvm::Value* payloadValue = irb.CreateExtractValue( rtcCall, pi, "unpackedPayload" );
                outPayload                = irb.CreateInsertValue( outPayload, payloadValue, pi, "unpackedPayload" );
            }
        }
        else
        {
            if( payloadSize != 1 )
                return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            outPayload = irb.CreateExtractValue( rtcCall, 0, "unpackedPayload" );
        }

        CI->replaceAllUsesWith( outPayload );
    }
    else
    {
        // In debug there may be uses of the return value even if payloadSize == 0.
        llvm::Value* outPayload = llvm::UndefValue::get( outTy );
        CI->replaceAllUsesWith( outPayload );
    }

    CI->eraseFromParent();

    // Generate payload semantics metadata and attach to the trace call.
    rtcCall->setMetadata( "lwvm.rt.payloadType",
                          llvm::MDNode::get( llvmContext, UseValueAsMd( llvm::ConstantInt::get( i32Ty, typeIndex ) ) ) );

    return result;
}

// Up to ABI 45, OptiX provided unique trace intrinsics for different payload sizes
static OptixResult lowerTraceLegacy( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i8Ty        = llvm::Type::getInt8Ty( llvmContext );
    llvm::Type*        i16Ty       = llvm::Type::getInt16Ty( llvmContext );
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i64Ty       = llvm::Type::getInt64Ty( llvmContext );
    llvm::Type*        floatTy     = llvm::Type::getFloatTy( llvmContext );

    IntrinsicIndex toHandle[] = { optix_trace_0, optix_trace_1, optix_trace_2, optix_trace_3, optix_trace_4,
                                  optix_trace_5, optix_trace_6, optix_trace_7, optix_trace_8 };

    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        // Legacy payload accessors where deprecated (ABI 46) before payload types (ABI 47) where introduced.
        // There should only be exactly one payload type.
        if( module.compileParams.payloadTypes.size() != 1 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "There should be exactly one payload type when using legacy payload accesses" );
        const int numPayloadValues = (int)module.compileParams.payloadTypes.front().semantics.size();

        int payloadSize = intrinsicIdx - optix_trace_0;
        if( numPayloadValues < payloadSize )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << payloadSize << " payload values in " << apiName( intrinsicIdx )
                << " but only " << numPayloadValues
                << " are configured in the pipeline: " << getSourceLocation( module.llvmModule, calls.front() ) << "\n";
            result += OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS;
        }
        std::vector<llvm::Type*> plElements( payloadSize, i32Ty );
        llvm::Type*              payloadTy = llvm::StructType::get( llvmContext, plElements );

        for( llvm::CallInst* CI : calls )
        {
            llvm::Function* caller = CI->getParent()->getParent();
            module.recordTraceCall( caller );

            unsigned int expectedArgCount = 15 + payloadSize;
            if( CI->getNumArgOperands() != expectedArgCount )
            {
                errDetails.m_compilerFeedback << "Error: Parameter count mismatch in " << apiName( intrinsicIdx )
                                              << ". Found " << CI->getNumArgOperands() << " arguments, but expected "
                                              << expectedArgCount << ".\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                continue;
            }

            corelib::CoreIRBuilder irb{ CI };
            llvm::Value*           packedPayload = llvm::UndefValue::get( payloadTy );
            for( int i = 0; i < payloadSize; ++i )
            {
                llvm::Value* payloadArg = CI->getArgOperand( 15 + i );
                packedPayload           = irb.CreateInsertValue( packedPayload, payloadArg, i, "packedPayload" );
            }

            result += lowerTraceToRtcTrace( module, compileParams, intrinsicIdx, CI, packedPayload,
                                            /*hasPayloadTypeOperand=*/false, errDetails );
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        module.eraseIntrinsic( intrinsicIdx );
    }

    return result;
}

static OptixResult lowerTrace( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i8Ty        = llvm::Type::getInt8Ty( llvmContext );
    llvm::Type*        i16Ty       = llvm::Type::getInt16Ty( llvmContext );
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i64Ty       = llvm::Type::getInt64Ty( llvmContext );
    llvm::Type*        floatTy     = llvm::Type::getFloatTy( llvmContext );

    IntrinsicIndex toHandle[] = { optix_trace_typed_32 };

    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( !calls.empty() )
        {
            for( llvm::CallInst* CI : calls )
            {
                llvm::Function* caller = CI->getParent()->getParent();
                module.recordTraceCall( caller );

                const unsigned int payloadSizeOperandIdx  = 16;
                const unsigned int firstPayloadOperandIdx = payloadSizeOperandIdx + 1;
                const unsigned int expectedArgCount       = firstPayloadOperandIdx + 32;
                if( CI->getNumArgOperands() != expectedArgCount )
                {
                    errDetails.m_compilerFeedback << "Error: Parameter count mismatch in " << apiName( intrinsicIdx )
                                                  << ". Found " << CI->getNumArgOperands()
                                                  << " arguments, but expected " << expectedArgCount << ".\n";
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                    continue;
                }

                int32_t payloadSize;
                bool isConstant = corelib::getConstantValue( CI->getArgOperand( payloadSizeOperandIdx ), payloadSize );
                if( !isConstant )
                {
                    errDetails.m_compilerFeedback << "Error: Requested " << apiName( intrinsicIdx ) << " but "
                                                  << "payload count operator isn't a constant integer."
                                                  << getSourceLocation( CI ) << "\n";
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                    continue;
                }

                std::vector<llvm::Type*> plElements( payloadSize, i32Ty );
                llvm::Type*              payloadTy = llvm::StructType::get( llvmContext, plElements );

                // discard all input payload values beyond the payload size specified as operand 15.
                corelib::CoreIRBuilder irb{ CI };
                llvm::Value*           packedPayload = llvm::UndefValue::get( payloadTy );
                for( int i = 0; i < payloadSize; ++i )
                {
                    llvm::Value* payloadArg = CI->getArgOperand( firstPayloadOperandIdx + i );
                    packedPayload           = irb.CreateInsertValue( packedPayload, payloadArg, i, "packedPayload" );
                }

                result += lowerTraceToRtcTrace( module, compileParams, intrinsicIdx, CI, packedPayload,
                                                /*hasPayloadTypeOperand=*/true, errDetails );
            }
        }

        module.eraseIntrinsic( intrinsicIdx );
    }

    return result;
}

static OptixResult lowerGetAttribute( CompilationUnit& module, int numAttributeValues, ErrorDetails& errDetails )
{
    OptixResultOneShot           result;
    std::vector<llvm::Function*> toDelete;
    IntrinsicIndex               toHandle[] = {optix_get_attribute_0, optix_get_attribute_1, optix_get_attribute_2,
                                 optix_get_attribute_3, optix_get_attribute_4, optix_get_attribute_5,
                                 optix_get_attribute_6, optix_get_attribute_7};
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        int numRegister = intrinsicIdx - optix_get_attribute_0;

        if( numAttributeValues < numRegister )
        {
            errDetails.m_compilerFeedback
                << "Error: Requested " << numRegister << " attribute values in " << apiName( intrinsicIdx )
                << " but only " << numAttributeValues
                << " are configured in the pipeline: " << getSourceLocation( module.llvmModule, calls.front() ) << "\n";
            result += OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS;
        }
        module.usedAttributeValues = std::max( module.usedAttributeValues, numRegister + 1 );

        llvm::Type*  i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Value* idx   = llvm::ConstantInt::get( i32Ty, numRegister );

        llvm::Type*         argTypes[] = {i32Ty};
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
        llvm::Function*     rtcGetAttributeFunc =
            corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.register.attribute.i32", funcTy );
        for( llvm::CallInst* CI : calls )
        {
            result += isAttributeAccessLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            corelib::CoreIRBuilder irb{CI};
            llvm::Value*           attrValue = irb.CreateCall( rtcGetAttributeFunc, idx );
            CI->replaceAllUsesWith( attrValue );
            CI->eraseFromParent();
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
        module.eraseIntrinsic( intrinsicIdx );

    return result;
}

static OptixResult lowerGetTriangleBarycentrics( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    f = module.llvmIntrinsics[optix_get_triangle_barycentrics];
    if( !f )
        return result;

    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( f );
    if( calls.empty() )
    {
        module.eraseIntrinsic( optix_get_triangle_barycentrics );
        return result;
    }

    llvm::Type*  i32Ty   = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::Type*  floatTy = llvm::Type::getFloatTy( module.llvmModule->getContext() );
    llvm::Value* zero    = llvm::ConstantInt::get( i32Ty, 0 );
    llvm::Value* one     = llvm::ConstantInt::get( i32Ty, 1 );

    llvm::Type*         argTypes[] = {i32Ty};
    llvm::FunctionType* funcTy     = llvm::FunctionType::get( floatTy, argTypes, false );
    llvm::Function*     rtcGetAttributeFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.register.attribute.float", funcTy );

    for( llvm::CallInst* CI : calls )
    {
        result += isGetTriangleBarycentricsLegal( CI, module.compileParams, errDetails,
                                                  apiName( optix_get_triangle_barycentrics ) );
        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           uValue = irb.CreateCall( rtcGetAttributeFunc, zero );
        llvm::Value*           vValue = irb.CreateCall( rtcGetAttributeFunc, one );
        llvm::Value*           retVal = llvm::UndefValue::get( f->getReturnType() );
        retVal                        = irb.CreateInsertValue( retVal, uValue, 0 );
        retVal                        = irb.CreateInsertValue( retVal, vValue, 1 );
        CI->replaceAllUsesWith( retVal );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_triangle_barycentrics );

    return result;
}

static OptixResult lowerGetRayVisibilityMask( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    f = module.llvmIntrinsics[optix_get_ray_visibility_mask];
    if( !f )
        return result;

    llvm::FunctionType* rtcFunctionType =
        llvm::FunctionType::get( llvm::Type::getInt8Ty( module.llvmModule->getContext() ), false );
    llvm::Function* rtcFunctionMask = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.ray.mask", rtcFunctionType );
    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        result += isRayAccessLegal( CI, module.compileParams, errDetails, apiName( optix_get_ray_visibility_mask ) );
        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           mask = irb.CreateCall( rtcFunctionMask );
        mask = irb.CreateIntCast( mask, llvm::Type::getInt32Ty( module.llvmModule->getContext() ), false );
        CI->replaceAllUsesWith( mask );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_ray_visibility_mask );
    return result;
}

static OptixResult lowerGetRayFlags( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    f = module.llvmIntrinsics[optix_get_ray_flags];
    if( !f )
        return result;

    llvm::FunctionType* rtcFunctionType =
        llvm::FunctionType::get( llvm::Type::getInt16Ty( module.llvmModule->getContext() ), false );
    llvm::Function* rtcFunctionFlags = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.ray.flags", rtcFunctionType );

    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        result += isRayAccessLegal( CI, module.compileParams, errDetails, apiName( optix_get_ray_flags ) );
        corelib::CoreIRBuilder irb{CI};
        llvm::Value*           flags = irb.CreateCall( rtcFunctionFlags );
        flags = irb.CreateIntCast( flags, llvm::Type::getInt32Ty( module.llvmModule->getContext() ), false );
        CI->replaceAllUsesWith( flags );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_ray_flags );
    return result;
}

static OptixResult lowerGetRayTime( CompilationUnit& module, bool motionEnabled, ErrorDetails& errDetails )
{
    if( motionEnabled )
        return replaceFunctionUses( module, optix_get_ray_time, "lw.rt.read.current.time", isRayAccessLegal, errDetails );

    OptixResultOneShot result;
    llvm::Function*    f = module.llvmIntrinsics[optix_get_ray_time];
    if( !f )
        return result;
    llvm::Type*  floatTy = llvm::Type::getFloatTy( module.llvmModule->getContext() );
    llvm::Value* zero    = llvm::ConstantFP::get( floatTy, 0.f );
    for( llvm::CallInst* CI : corelib::getCallsToFunction( f ) )
    {
        result += isRayAccessLegal( CI, module.compileParams, errDetails, apiName( optix_get_ray_time ) );
        CI->replaceAllUsesWith( zero );
        CI->eraseFromParent();
    }
    module.eraseIntrinsic( optix_get_ray_time );
    return result;
}

static OptixResult generateThrowOptixException( CompilationUnit&    module,
                                                OptixExceptionCodes exceptionCode,
                                                llvm::Value*        details,
                                                llvm::Instruction*  insertBefore,
                                                ErrorDetails&       errDetails )
{
    OptixResult        result      = OPTIX_SUCCESS;
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i32_23Ty    = llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS );
    llvm::Type*        voidTy      = llvm::Type::getVoidTy( llvmContext );

    llvm::Type*         argTypes[] = {i32Ty, i32_23Ty};
    llvm::FunctionType* funcTy     = llvm::FunctionType::get( voidTy, argTypes, false );
    llvm::Function* rtcThrowFunc = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.throw.exception", funcTy );

    corelib::CoreIRBuilder irb{insertBefore};
    llvm::Value*           exceptionCodeValue = irb.getInt32( exceptionCode );

    llvm::Value* exceptionDetails = details;
    if( !details )
        exceptionDetails = llvm::UndefValue::get( i32_23Ty );
    else if( details->getType() != i32_23Ty )
    {
        if( !details->getType()->isArrayTy() || details->getType()->getArrayNumElements() > RTC_NUM_EXCEPTION_DETAILS )
        {
            errDetails.m_compilerFeedback << "Error: internal compiler error. Trying to generate exception with "
                                             "invalid exception details.";
            return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }
        exceptionDetails = llvm::UndefValue::get( i32_23Ty );
        int i            = 0;
        for( int e = details->getType()->getArrayNumElements(); i < e; ++i )
        {
            llvm::Value* val = irb.CreateExtractValue( details, i );
            exceptionDetails = irb.CreateInsertValue( exceptionDetails, val, i );
        }
    }
    llvm::Value* argValues[] = {exceptionCodeValue, exceptionDetails};
    llvm::Value* call        = irb.CreateCall( rtcThrowFunc, argValues );

    return OPTIX_SUCCESS;
}

static OptixResult generateConditionalThrowOptixException( CompilationUnit&    module,
                                                           llvm::Value*        condition,
                                                           OptixExceptionCodes exceptionCode,
                                                           llvm::Instruction*& insertBefore,
                                                           llvm::Value*        details,
                                                           ErrorDetails&       errDetails )
{
    OptixResult       result      = OPTIX_SUCCESS;
    llvm::BasicBlock* parentBlock = insertBefore->getParent();
    llvm::BasicBlock* okBlock     = parentBlock->splitBasicBlock( insertBefore, "noException" );
    llvm::BasicBlock* exBlock     = parentBlock->splitBasicBlock( parentBlock->getTerminator(), "exception" );

    llvm::TerminatorInst*  ti = parentBlock->getTerminator();
    corelib::CoreIRBuilder irb{ti};
    irb.CreateCondBr( condition, exBlock, okBlock );
    ti->eraseFromParent();

    result = generateThrowOptixException( module, exceptionCode, details, exBlock->getTerminator(), errDetails );

    insertBefore = &*okBlock->getFirstInsertionPt();
    return result;
}

static OptixResult lowerThrowException( CompilationUnit& module, unsigned int exceptionFlags, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::LLVMContext& llvmContext = module.llvmModule->getContext();
    llvm::Type*        i32Ty       = llvm::Type::getInt32Ty( llvmContext );
    llvm::Type*        i32_23Ty    = llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS );
    llvm::Type*        voidTy      = llvm::Type::getVoidTy( llvmContext );

    IntrinsicIndex toHandle[] = {optix_throw_exception_0, optix_throw_exception_1, optix_throw_exception_2,
                                 optix_throw_exception_3, optix_throw_exception_4, optix_throw_exception_5,
                                 optix_throw_exception_6, optix_throw_exception_7, optix_throw_exception_8};

    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        int detailsSize = intrinsicIdx - optix_throw_exception_0;

        llvm::Type*         argTypes[] = {i32Ty, i32_23Ty};
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( voidTy, argTypes, false );
        llvm::Function* rtcThrowFunc = corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.throw.exception", funcTy );

        for( llvm::CallInst* CI : calls )
        {
            unsigned int expectedArgCount = 1 + detailsSize;
            if( CI->getNumArgOperands() != expectedArgCount )
            {
                errDetails.m_compilerFeedback << "Error: Parameter count mismatch in " << apiName( intrinsicIdx )
                                              << ". Found " << CI->getNumArgOperands() << " arguments, but expected "
                                              << expectedArgCount << ".\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                continue;
            }

            result += isThrowExceptionLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            if( module.compileParams.isBuiltinModule && ( exceptionFlags & OPTIX_EXCEPTION_FLAG_DEBUG ) == 0 )
            {
                errDetails.m_compilerFeedback
                    << "Error: " << apiName( intrinsicIdx )
                    << " used in builtin IS, but the exception flags of the pipeline compilation options "
                       "do not include OPTIX_EXCEPTION_FLAG_DEBUG: "
                    << getSourceLocation( F->getParent(), CI ) << "\n";
                result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
            }
            else if( !module.compileParams.isBuiltinModule && ( exceptionFlags & OPTIX_EXCEPTION_FLAG_USER ) == 0 )
            {
                if( module.compileParams.elideUserThrow )
                {
                    errDetails.m_compilerFeedback
                        << "Info: Removed call to " << apiName( intrinsicIdx )
                        << " because user exceptions are disabled. Add OPTIX_EXCEPTION_FLAG_USER to enable: "
                        << getSourceLocation( CI );
                    CI->eraseFromParent();
                    continue;
                }
                errDetails.m_compilerFeedback << "Error: " << apiName( intrinsicIdx )
                                              << " used, but the exception flags of the pipeline compilation options "
                                                 "do not include OPTIX_EXCEPTION_FLAG_USER: "
                                              << getSourceLocation( F->getParent(), CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_USE;
            }

            corelib::CoreIRBuilder irb{CI};
            llvm::Value*           exceptionCode = CI->getArgOperand( 0 );

            int  exceptionCodeAsInt;
            bool isConstant = corelib::getConstantValue( exceptionCode, exceptionCodeAsInt );
            if( !isConstant )
            {
                errDetails.m_compilerFeedback << "Error: Exception code in " << apiName( intrinsicIdx )
                                              << " is not a constant: " << getSourceLocation( F->getParent(), CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
            }
            else if( !module.compileParams.isBuiltinModule && exceptionCodeAsInt < 0 )
            {
                // internal modules may use the user optixThrowException to throw internal exception codes.
                errDetails.m_compilerFeedback
                    << "Error: Exception code in " << apiName( intrinsicIdx ) << " is too small. Found " << exceptionCodeAsInt
                    << ", but needs to be at least 0: " << getSourceLocation( F->getParent(), CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
            }
            else if( exceptionCodeAsInt > ( 1 << 30 ) - 1 )
            {
                errDetails.m_compilerFeedback
                    << "Error: Exception code in " << apiName( intrinsicIdx ) << " is too large. Found " << exceptionCodeAsInt
                    << ", but needs to be at most 2^30-1: " << getSourceLocation( F->getParent(), CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS;
            }

            llvm::Value* exceptionDetails = llvm::UndefValue::get( i32_23Ty );
            for( int i = 0; i < detailsSize; ++i )
            {
                llvm::Value* detailsArg = CI->getArgOperand( 1 + i );
                exceptionDetails        = irb.CreateInsertValue( exceptionDetails, detailsArg, i, "exceptionDetails" );
            }

            exceptionDetails = insertExceptionSourceLocation( module, exceptionDetails, CI );

            llvm::Value* argValues[] = {exceptionCode, exceptionDetails};
            llvm::Value* call        = irb.CreateCall( rtcThrowFunc, argValues );
            CI->replaceAllUsesWith( call );
            CI->eraseFromParent();
        }
    }

    for( IntrinsicIndex intrinsicIdx : toHandle )
        module.eraseIntrinsic( intrinsicIdx );

    return result;
}

static OptixResult lowerGetExceptionDetails( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot           result;
    std::vector<llvm::Function*> toDelete;
    IntrinsicIndex               toHandle[] = {optix_get_exception_detail_0, optix_get_exception_detail_1,
                                 optix_get_exception_detail_2, optix_get_exception_detail_3,
                                 optix_get_exception_detail_4, optix_get_exception_detail_5,
                                 optix_get_exception_detail_6, optix_get_exception_detail_7};
    for( IntrinsicIndex intrinsicIdx : toHandle )
    {
        llvm::Function* F = module.llvmIntrinsics[intrinsicIdx];

        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
        if( calls.empty() )
        {
            continue;
        }

        int numRegister = intrinsicIdx - optix_get_exception_detail_0;

        llvm::Type*  i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        llvm::Value* idx   = llvm::ConstantInt::get( i32Ty, numRegister );

        llvm::Type*         argTypes[] = {i32Ty};
        llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
        llvm::Function*     rtcGetExceptionDetailFunc =
            corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.detail", funcTy );
        for( llvm::CallInst* CI : calls )
        {
            result += isGetExceptionDetailLegal( CI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            corelib::CoreIRBuilder irb{CI};
            llvm::Value*           detailValue = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
            CI->replaceAllUsesWith( detailValue );
            CI->eraseFromParent();
        }
    }
    for( IntrinsicIndex intrinsicIdx : toHandle )
        module.eraseIntrinsic( intrinsicIdx );

    return result;
}

static OptixResult lowerGetExceptionIlwalidRay( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    F = module.llvmIntrinsics[optix_get_exception_ilwalid_ray];

    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    if( calls.empty() )
    {
        return result;
    }
    llvm::Type*         i32Ty      = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::Type*         argTypes[] = {i32Ty};
    llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
    llvm::Function*     rtcGetExceptionDetailFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.detail", funcTy );

    funcTy = llvm::FunctionType::get( i32Ty, false );
    llvm::Function* readExceptionCodeFunction =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.code", funcTy );

    for( llvm::CallInst* CI : calls )
    {
        result += isGetExceptionDetailLegal( CI, module.compileParams, errDetails, apiName( optix_get_exception_ilwalid_ray ) );
        llvm::Type*  returnTy    = CI->getType();
        llvm::Value* resultOk    = llvm::UndefValue::get( returnTy );
        llvm::Value* resultNotOk = llvm::UndefValue::get( returnTy );

        if( !returnTy->isStructTy() || returnTy->getStructNumElements() != 9 )
        {
            errDetails.m_compilerFeedback
                << "Error: Type mismatch for call to optixGetExceptionIlwalidRay. Expected { 9 x float }, found "
                << llvmToString( returnTy ) << ": " << getSourceLocation( CI ) << "\n";
            result += OPTIX_ERROR_ILWALID_PTX;
            continue;
        }

        llvm::BasicBlock* parentBlock      = CI->getParent();
        llvm::BasicBlock* fallthroughBlock = parentBlock->splitBasicBlock( CI, "fallthrough" );
        llvm::BasicBlock* notOkBlock =
            parentBlock->splitBasicBlock( parentBlock->getTerminator(), "noBadRayException" );
        llvm::TerminatorInst*  ti = parentBlock->getTerminator();
        corelib::CoreIRBuilder irb{ti};
        for( int i = 0; i < returnTy->getStructNumElements(); ++i )
        {
            if( returnTy->getStructElementType( i ) != irb.getFloatTy() )
            {
                errDetails.m_compilerFeedback
                    << "Error: Type mismatch for call to optixGetExceptionIlwalidRay. Expected { 9 x float }, found "
                    << llvmToString( returnTy ) << ": " << getSourceLocation( CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_PTX;
                continue;
            }
            llvm::Value* idx         = llvm::ConstantInt::get( i32Ty, i );
            llvm::Value* detailValue = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
            // Return type of _optix_get_exception_ilwalid_ray is a struct with 9 floats.
            resultOk = irb.CreateInsertValue( resultOk, irb.CreateBitCast( detailValue, irb.getFloatTy() ), i );
        }
        llvm::Value* exceptionCode = irb.CreateCall( readExceptionCodeFunction );
        llvm::Value* condition = irb.CreateICmpEQ( exceptionCode, irb.getInt32( OPTIX_EXCEPTION_CODE_ILWALID_RAY ) );
        irb.CreateCondBr( condition, fallthroughBlock, notOkBlock );

        ti->eraseFromParent();

        irb.SetInsertPoint( &*notOkBlock->getFirstInsertionPt() );
        llvm::Value* zero = llvm::ConstantFP::get( irb.getFloatTy(), 0 );
        for( int i = 0; i < returnTy->getStructNumElements(); ++i )
            resultNotOk = irb.CreateInsertValue( resultNotOk, zero, i );

        irb.SetInsertPoint( &*fallthroughBlock->getFirstInsertionPt() );
        llvm::PHINode* phi = irb.CreatePHI( returnTy, 2 );
        phi->addIncoming( resultNotOk, notOkBlock );
        phi->addIncoming( resultOk, parentBlock );
        CI->replaceAllUsesWith( phi );
        CI->eraseFromParent();
    }

    return result;
}

static OptixResult lowerGetExceptionCallableParameterMismatch( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Function*    F = module.llvmIntrinsics[optix_get_exception_parameter_mismatch];

    std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( F );
    if( calls.empty() )
    {
        return result;
    }
    llvm::Type*         i32Ty      = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::Type*         argTypes[] = {i32Ty};
    llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
    llvm::Function*     rtcGetExceptionDetailFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.detail", funcTy );

    funcTy = llvm::FunctionType::get( i32Ty, false );
    llvm::Function* readExceptionCodeFunction =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.code", funcTy );

    for( llvm::CallInst* CI : calls )
    {
        result += isGetExceptionDetailLegal( CI, module.compileParams, errDetails,
                                             apiName( optix_get_exception_parameter_mismatch ) );
        llvm::Type*  returnTy    = CI->getType();
        llvm::Value* resultOk    = llvm::UndefValue::get( returnTy );
        llvm::Value* resultNotOk = llvm::UndefValue::get( returnTy );

        if( !returnTy->isStructTy() || returnTy->getStructNumElements() != 4 )
        {
            errDetails.m_compilerFeedback << "Error: Type mismatch for call to optixGetExceptionParameterMismatch. "
                                             "Expected { u32, u32, u32, u64 }, found "
                                          << llvmToString( returnTy ) << ": " << getSourceLocation( CI ) << "\n";
            result += OPTIX_ERROR_ILWALID_PTX;
            continue;
        }

        llvm::BasicBlock* parentBlock      = CI->getParent();
        llvm::BasicBlock* fallthroughBlock = parentBlock->splitBasicBlock( CI, "fallthrough" );
        llvm::BasicBlock* notOkBlock =
            parentBlock->splitBasicBlock( parentBlock->getTerminator(), "noParamMismatchException" );
        llvm::TerminatorInst*  ti = parentBlock->getTerminator();
        corelib::CoreIRBuilder irb{ti};
        for( int i = 0; i < 3; ++i )
        {
            if( returnTy->getStructElementType( i ) != i32Ty )
            {
                errDetails.m_compilerFeedback << "Error: Type mismatch for call to optixGetExceptionParameterMismatch. "
                                                 "Expected { u32, u32, u32, u64 }, found "
                                              << llvmToString( returnTy ) << ": " << getSourceLocation( CI ) << "\n";
                result += OPTIX_ERROR_ILWALID_PTX;
                continue;
            }
            llvm::Value* idx         = llvm::ConstantInt::get( i32Ty, i );
            llvm::Value* detailValue = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
            resultOk                 = irb.CreateInsertValue( resultOk, detailValue, i );
        }

        llvm::Value* idx           = llvm::ConstantInt::get( i32Ty, 3 );
        llvm::Value* detailValueLo = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
        idx                        = llvm::ConstantInt::get( i32Ty, 4 );
        llvm::Value* detailValueHi = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
        llvm::Value* detailValue64 = corelib::createCast_2xi32_to_i64( detailValueLo, detailValueHi, ti );
        resultOk                   = irb.CreateInsertValue( resultOk, detailValue64, 3 );

        // Create branch to check for the correct exception code.
        llvm::Value* exceptionCode = irb.CreateCall( readExceptionCodeFunction );
        llvm::Value* condition =
            irb.CreateICmpEQ( exceptionCode, irb.getInt32( OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH ) );
        irb.CreateCondBr( condition, fallthroughBlock, notOkBlock );

        ti->eraseFromParent();

        // Create Null return value for other exception codes.
        resultNotOk = llvm::Constant::getNullValue( returnTy );

        irb.SetInsertPoint( &*fallthroughBlock->getFirstInsertionPt() );
        llvm::PHINode* phi = irb.CreatePHI( returnTy, 2 );
        phi->addIncoming( resultNotOk, notOkBlock );
        phi->addIncoming( resultOk, parentBlock );
        CI->replaceAllUsesWith( phi );
        CI->eraseFromParent();
    }

    return result;
}

static OptixResult lowerInbuiltExceptionDetails( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    result += lowerGetExceptionIlwalidRay( module, errDetails );
    result += lowerGetExceptionCallableParameterMismatch( module, errDetails );
    return result;
}

static OptixResult lowerGetExceptionLineInfo( CompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot           result;
    llvm::Function*              function = module.llvmIntrinsics[optix_get_exception_line_info];
    std::vector<llvm::CallInst*> calls    = corelib::getCallsToFunction( function );
    if( calls.empty() )
        return result;

    llvm::Type*         i32Ty      = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::Type*         argTypes[] = {i32Ty};
    llvm::FunctionType* funcTy     = llvm::FunctionType::get( i32Ty, argTypes, false );
    llvm::Function*     rtcGetExceptionDetailFunc =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.detail", funcTy );
    funcTy = llvm::FunctionType::get( i32Ty, false );
    llvm::Function* readExceptionCodeFunction =
        corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.code", funcTy );
    llvm::Function* isInfoAvailableFunction =
        module.llvmModule->getFunction( STRINGIFY( RUNTIME_IS_EXCEPTION_LINE_INFO_AVAILABLE ) );
    if( !isInfoAvailableFunction )
    {
        errDetails.m_compilerFeedback << "Error: Runtime function missing\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }
    for( llvm::CallInst* CI : calls )
    {
        result += isGetExceptionDetailLegal( CI, module.compileParams, errDetails, apiName( optix_get_exception_line_info ) );
        llvm::BasicBlock* parentBlock      = CI->getParent();
        llvm::BasicBlock* fallthroughBlock = parentBlock->splitBasicBlock( CI, "fallthrough" );
        llvm::BasicBlock* okBlock = parentBlock->splitBasicBlock( parentBlock->getTerminator(), "infoAvailable" );

        llvm::TerminatorInst*  ti = parentBlock->getTerminator();
        corelib::CoreIRBuilder irb{ti};
        llvm::Value*           infoAvailable = irb.CreateCall( readExceptionCodeFunction );
        infoAvailable                        = irb.CreateCall( isInfoAvailableFunction, infoAvailable );
        irb.CreateCondBr( infoAvailable, okBlock, fallthroughBlock );
        ti->eraseFromParent();

        irb.SetInsertPoint( &*okBlock->getFirstInsertionPt() );
        llvm::Value* idx     = irb.getInt32( RTC_NUM_EXCEPTION_DETAILS - 2 );
        llvm::Value* lo      = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
        idx                  = irb.getInt32( RTC_NUM_EXCEPTION_DETAILS - 1 );
        llvm::Value* hi      = irb.CreateCall( rtcGetExceptionDetailFunc, idx );
        llvm::Value* infoPtr = corelib::createCast_2xi32_to_i64( lo, hi, okBlock->getTerminator() );

        llvm::Value* noInfoAvailablePtr = addModuleString( module,
                                                           "No line information available for this "
                                                           "exception." );

        irb.SetInsertPoint( &*fallthroughBlock->getFirstInsertionPt() );
        llvm::PHINode* phi = irb.CreatePHI( irb.getInt64Ty(), 2 );
        phi->addIncoming( noInfoAvailablePtr, parentBlock );
        phi->addIncoming( infoPtr, okBlock );
        CI->replaceAllUsesWith( phi );
        CI->eraseFromParent();
    }
    return result;
}

// helper function to extract the value nested inside in cast instructions
static llvm::Value* getDefSkipCasts( llvm::Value* val )
{
    while( llvm::CastInst* CI = llvm::dyn_cast<llvm::CastInst>( val ) )
        val = CI->getOperand( 0 );
    return val;
}

static OptixResult collectCallableCallsites( llvm::CallInst*                  rtCallableCI,
                                             std::vector<llvm::Instruction*>& toDelete,
                                             llvm::SmallVector<llvm::CallInst*, 2>& calls,
                                             ErrorDetails& errDetails )
{
    // Find all calls based on the use of the output of _optix_call_callable and

    llvm::SmallVector<llvm::Instruction*, 4> worklist;
    llvm::SmallSet<llvm::Instruction*, 4>    visited;
    // For robustness we use process the uses in a worklist although
    // we expect only one use.
    worklist.push_back( rtCallableCI );
    OptixResultOneShot result;
    while( !worklist.empty() )
    {
        llvm::Instruction* inst = worklist.back();
        worklist.pop_back();

        if( !std::get<1>( visited.insert( inst ) ) )
            continue;

        // All uses of the return value of _optix_call_callable should be either
        //  a) the cast instruction to the function pointer or
        //  b) the call to the function pointer (to be precise, that call is done on the result of a) ).
        // We want to collect the calls to be able to replace those with the
        // correct lwvm rt intrinsic.
        for( llvm::Instruction::user_iterator I = inst->user_begin(), E = inst->user_end(); I != E; ++I )
        {
            if( llvm::isa<llvm::CastInst>( *I ) )
            {
                llvm::Instruction* use = llvm::cast<llvm::Instruction>( *I );
                toDelete.push_back( use );
                // Collect the cast for further processing. The use of it should be the actual call.
                worklist.push_back( use );
            }
            else if( llvm::CallInst* CI = llvm::dyn_cast<llvm::CallInst>( *I ) )
            {
                if( getDefSkipCasts( CI->getCalledValue() ) != rtCallableCI )
                {
                    errDetails.m_compilerFeedback
                        << "Error: Return value of _optix_call_callable not used as a function pointer: " << llvmToString( CI )
                        << "\n";
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                }
                else
                {
                    calls.push_back( CI );
                }
            }
            else
            {
                if( getDefSkipCasts( CI->getCalledValue() ) != rtCallableCI )
                {
                    errDetails.m_compilerFeedback
                        << "Error: Return value of _optix_call_callable not used as a function pointer: " << llvmToString( CI )
                        << "\n";
                    result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
                }
            }
        }
    }
    return result;
}

static llvm::FunctionType* canonicalizeCallType( llvm::CallInst* call )
{
    llvm::SmallVector<llvm::Type*, 4> argTypes;
    for( unsigned i = 0, e = call->getNumArgOperands(); i < e; ++i )
    {
        argTypes.push_back( call->getArgOperand( i )->getType() );
    }
    llvm::FunctionType* oldFuncTy   = llvm::FunctionType::get( call->getType(), argTypes, false );
    llvm::FunctionType* cleanFuncTy = optix::getCleanFunctionType( oldFuncTy );
    return cleanFuncTy;
}

static std::vector<llvm::Value*> canonicalizeCallArguments( llvm::CallInst* call, llvm::FunctionType* cleanFuncTy )
{
    std::vector<llvm::Value*> callArgs( call->getNumArgOperands() );
    for( unsigned int i = 0; i < call->getNumArgOperands(); ++i )
    {
        // "Clean up" arguments by replacing byte array args with larger types if possible
        llvm::Value* arg   = call->getArgOperand( i );
        llvm::Type*  oldTy = arg->getType();
        llvm::Type*  newTy = cleanFuncTy->getParamType( i );
        if( oldTy != newTy )
        {
            // Cast arg
            arg = optix::castThroughAlloca( arg, newTy, call );
        }
        callArgs[i] = arg;
    }
    return callArgs;
}

// Information needed during rewriting of calls to callable programs
struct CallableCallInformation
{
    int          numParamVals = 0;
    llvm::Type*  newReturnTy  = nullptr;
    llvm::Type*  paramArrayTy = nullptr;
    llvm::Value* paramArray   = nullptr;
};

static OptixResult collectCallableCallInformation( int                      numDebugParameters,
                                                   llvm::CallInst*          call,
                                                   CallableCallInformation& callInfo,
                                                   ErrorDetails&            errDetails )
{
    llvm::IntegerType*  i32Ty       = llvm::Type::getInt32Ty( call->getParent()->getParent()->getContext() );
    llvm::IntegerType*  i64Ty       = llvm::Type::getInt64Ty( call->getParent()->getParent()->getContext() );
    llvm::FunctionType* cleanFuncTy = canonicalizeCallType( call );

    std::vector<llvm::Value*> callArgs = canonicalizeCallArguments( call, cleanFuncTy );

    // pack up arguments in individual 32bit values for rtcore
    llvm::SmallVector<llvm::Value*, 8> paramVals;
    for( llvm::Value* arg : callArgs )
    {
        std::vector<llvm::Value*> flattened = corelib::flattenAggregateTo32BitValuesForRTCore( arg, call );
        paramVals.insert( paramVals.end(), flattened.begin(), flattened.end() );
    }

    corelib::CoreIRBuilder irb{call};

    llvm::Type* oldReturnTy = call->getType();
    callInfo.newReturnTy    = cleanFuncTy->getReturnType();
    if( llvm::StructType* ST = llvm::dyn_cast<llvm::StructType>( callInfo.newReturnTy ) )
    {
        if( ST->getNumElements() == 1 )
        {
            llvm::Type* cleanReturnType = optix::getCleanTypeForArg( ST->getElementType( 0 ) );
            callInfo.newReturnTy =
                llvm::StructType::get( oldReturnTy->getContext(), llvm::ArrayRef<llvm::Type*>{cleanReturnType} );
        }
    }

    OptixResultOneShot result;
    callInfo.numParamVals = paramVals.size() + numDebugParameters;

    // Pack the individual 32-bit values into an aggregate of 32-bit types
    // (int or float) as required by the RTCore spec.
    callInfo.paramArrayTy = llvm::ArrayType::get( i32Ty, callInfo.numParamVals );
    callInfo.paramArray   = llvm::UndefValue::get( callInfo.paramArrayTy );
    for( int j = 0, e = static_cast<int>( paramVals.size() ); j < e; ++j )
    {
        llvm::Value* arg = paramVals[j];
        if( arg->getType() != i32Ty )
            arg             = irb.CreateBitCast( arg, i32Ty );
        callInfo.paramArray = irb.CreateInsertValue( callInfo.paramArray, arg, j + numDebugParameters );
    }

    return result;
}

static OptixResult handleCallableReturlwalue( llvm::CallInst*                call,
                                              llvm::Value*&                  newValue,
                                              const CallableCallInformation& callInfo,
                                              ErrorDetails&                  errDetails )
{
    if( !newValue->getType()->isVoidTy() )
    {
        corelib::CoreIRBuilder irb{call};
        // Extract return value

        llvm::Type* oldReturnTy = call->getType();
        newValue                = corelib::unflattenAggregateForRTCore( callInfo.newReturnTy, newValue, 0, call );
        if( callInfo.newReturnTy != oldReturnTy )
        {
            newValue = optix::castThroughAlloca( newValue, oldReturnTy, call );
        }
        newValue->takeName( call );
    }
    return OPTIX_SUCCESS;
}

static OptixResult lowerCalls( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails )
{
    OptixResultOneShot        result;
    IntrinsicIndex            intrinsicIndices[] = {optix_call_direct_callable, optix_call_continuation_callable};
    SemanticTypeCheckFunction checkFuncs[]       = {isDirectCallableCallLegal, isContinuationCallableCallLegal};
    std::string               rtcoreNames[]      = {"lw.rt.call.direct.", "lw.rt.call.continuation."};

    llvm::Type* i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    llvm::Type* i64Ty = llvm::Type::getInt64Ty( module.llvmModule->getContext() );

    for( int i = 0; i < 2; ++i )
    {
        IntrinsicIndex  intrinsicIdx = intrinsicIndices[i];
        llvm::Function* f            = module.llvmIntrinsics[intrinsicIdx];
        if( !f )
            continue;
        const std::string&              rtcoreName = rtcoreNames[i];
        std::vector<llvm::Instruction*> toDelete;
        for( llvm::CallInst* rtCallableCI : corelib::getCallsToFunction( f ) )
        {
            llvm::Function* caller = rtCallableCI->getParent()->getParent();
            module.recordCallableCall( caller, intrinsicIdx );

            result += checkFuncs[i]( rtCallableCI, module.compileParams, errDetails, apiName( intrinsicIdx ) );

            llvm::Value* callableSbtIndex = rtCallableCI->getArgOperand( 0 );

            toDelete.push_back( rtCallableCI );

            // translate call instruction to actual calls of the callable
            llvm::SmallVector<llvm::CallInst*, 2> calls;
            result += collectCallableCallsites( rtCallableCI, toDelete, calls, errDetails );

            // Process indirect calls.
            // Extract the arguments of the call and pack them up for rtcore.
            // Extract the return type and pack that up for rtcore.
            // If the number registers needed for the arguments or the return value
            // exceeds the number of parameter registers that are available, create
            // a spill alloca and pass in the pointer to the function.
            for( llvm::CallInst* call : calls )
            {
                CallableCallInformation callInformation;
                int                     numDebugParameters =
                    compileParams.enableCallableParamCheck ? compileParams.paramCheckExceptionRegisters : 0;

                result += collectCallableCallInformation( numDebugParameters, call, callInformation, errDetails );

                // If no sbt index was passed in, this is a noinline call. In that case we do not generate
                // the exception check code, because that exception can never occur for those since the
                // compiler would already catch it.
                if( callableSbtIndex && numDebugParameters > 0 )
                {
                    // Add the SBT index, the number of passed in parameters and the pointer to the source location string for the exception info.
                    llvm::Value* argCount = llvm::ConstantInt::get( i32Ty, callInformation.numParamVals - numDebugParameters );
                    llvm::Value* sourceLoc = getSourceLocatiolwalue( module, call );
                    std::pair<llvm::Value*, llvm::Value*> splitSourceLoc = corelib::createCast_i64_to_2xi32( sourceLoc, call );

                    corelib::CoreIRBuilder irb{call};
                    if( callableSbtIndex->getType() != i32Ty )
                        callableSbtIndex = irb.CreateBitCast( callableSbtIndex, i32Ty );

                    callInformation.paramArray = irb.CreateInsertValue( callInformation.paramArray, argCount, 0 );
                    callInformation.paramArray = irb.CreateInsertValue( callInformation.paramArray, callableSbtIndex, 1 );
                    callInformation.paramArray = irb.CreateInsertValue( callInformation.paramArray, splitSourceLoc.first, 2 );
                    callInformation.paramArray = irb.CreateInsertValue( callInformation.paramArray, splitSourceLoc.second, 3 );
                }

                llvm::Type* rtcReturnTy =
                    call->getType()->isVoidTy() ?
                        call->getType() :
                        llvm::ArrayType::get( i32Ty, corelib::getNumRequiredRegisters( callInformation.newReturnTy ) );
                // first argument is the SBT index of the callable
                llvm::Type*         rtcArgTypes[] = {i32Ty, callInformation.paramArrayTy};
                llvm::FunctionType* rtcFuncTy     = llvm::FunctionType::get( rtcReturnTy, rtcArgTypes, false );

                // Create function signature hash for unique lwvm function names.
                std::string str         = llvmToString( rtcFuncTy );
                unsigned token = llvm::djbHash( str ) & 0x00FFFFFF;
                std::string rtcFuncName = rtcoreName + std::to_string( token );

                // create lwvm function
                llvm::Function* rtcFunc = corelib::insertOrCreateFunction( module.llvmModule, rtcFuncName, rtcFuncTy );  //asserts if function already exists  with different type
                rtcFunc->addFnAttr( llvm::Attribute::NoUnwind );

                corelib::CoreIRBuilder irb{call};
                // create call to lwvm intrinsic with sbtIndex + newly packed args
                llvm::Value* args[]   = {callableSbtIndex, callInformation.paramArray};
                llvm::Value* newValue = irb.CreateCall( rtcFunc, args );

                result += handleCallableReturlwalue( call, newValue, callInformation, errDetails );

                call->replaceAllUsesWith( newValue );
                toDelete.push_back( call );
            }
            // Delete old calls.  We need to iterate in reverse insertion order, since we used
            // forward data flow to find the sequence of instructions, we go backward to delete
            // the bottom uses first.
            for( auto I = toDelete.rbegin(), IE = toDelete.rend(); I != IE; ++I )
                ( *I )->eraseFromParent();
            toDelete.clear();
        }

        // erase _optix_call_callable function
        module.eraseIntrinsic( intrinsicIdx );
    }

    return result;
}

static OptixResult optimizeModule( CompilationUnit& module, const InternalCompileParameters& compileParams, ErrorDetails& errDetails )
{
    // Run a small set of optimizations.
    llvm::legacy::PassManager PM;

    // Run alwaysInliner to clean up the linked runtime. We do that regardless of the optimization
    // level. LWCC inlines function marked as __forceinline__ in debug/O0 mode, too.
    PM.add( llvm::createAlwaysInlinerLegacyPass() );

    if( compileParams.optLevel != OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 )
    {
        // Run SROA to clean up the allocas that were introduced to stage the callable parameters through and
        // to clean up code that was introduced during launch parameter specialization.
        PM.add( llvm::createGlobalDCEPass() );
        PM.add( llvm::createSROAPass() );
        PM.add( llvm::createInstructionCombiningPass() );  // clean up after SFT
        PM.add( llvm::createGVNPass() );                   // global value numbering needed to clean up after SFT
        PM.add( llvm::createInstructionCombiningPass() );  // clean up after GVN
    }
    PM.run( *module.llvmModule );

    return OPTIX_SUCCESS;
}

static OptixResult rewriteCalleeParametersToI32Array( llvm::Function* oldFunction, llvm::Function* newFunction, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    // rewrite calls to no-inline functions.
    llvm::Type* i64Ty = llvm::Type::getInt64Ty( oldFunction->getContext() );

    std::vector<llvm::CallInst*>    calls = corelib::getCallsToFunction( oldFunction );
    std::vector<llvm::Instruction*> toDelete;
    for( llvm::CallInst* call : calls )
    {
        CallableCallInformation callInformation;
        result += collectCallableCallInformation( /*numDebugParameters*/ 0, call, callInformation, errDetails );

        corelib::CoreIRBuilder irb{call};
        llvm::Value*           newValue = nullptr;
        if( callInformation.numParamVals != 0 )
        {
            llvm::Value* args[] = {callInformation.paramArray};
            newValue            = irb.CreateCall( newFunction, args );
        }
        else
        {
            newValue = irb.CreateCall( newFunction );
        }

        result += handleCallableReturlwalue( call, newValue, callInformation, errDetails );

        call->replaceAllUsesWith( newValue );
        toDelete.push_back( call );
    }
    for( auto I = toDelete.rbegin(), IE = toDelete.rend(); I != IE; ++I )
        ( *I )->eraseFromParent();

    return result;
}

// If the original parameter was an aggregate, the rewritten parameters will first extract
// the original aggregate out of the new aggregate and then extract the desired value from
// that one. Instead, we want to extract the value directly from the new aggregate and replace
// the subsequent uses.
// This function callwlates the index of the relevant value in the new aggregate and the uses
// of the extracted value which need to be replaced.
static OptixResult extractActualUses( llvm::Instruction*                   originalExtract,
                                      const int                            oldIndex,
                                      int&                                 actualIndex,
                                      llvm::SetVector<llvm::Instruction*>& actualUses,
                                      ErrorDetails&                        errDetails )
{
    int                     offset      = 0;
    llvm::ExtractValueInst* extractInst = llvm::dyn_cast<llvm::ExtractValueInst>( originalExtract );
    if( !extractInst )
    {
        llvm::ExtractElementInst* extractEle = llvm::dyn_cast<llvm::ExtractElementInst>( originalExtract );
        if( !extractEle )
        {
            // Not an extract. This could be a store, for example. In this case we need to extract
            // the complete old aggregate type from the new one and replace the old arg with it.
            return OPTIX_SUCCESS;
        }
        if( !corelib::getConstantValue( extractEle->getIndexOperand(), offset ) )
        {
            errDetails.m_compilerFeedback << "Error: found invalid extract from vector parameter " << oldIndex
                                          << ". Only constant indices are supported: " << getSourceLocation( originalExtract );
            return OPTIX_ERROR_ILWALID_PTX;
        }
    }
    else
    {
        if( extractInst->getNumIndices() != 1 )
        {
            errDetails.m_compilerFeedback
                << "Error: Unexpected number of indices in extract instruction for parameter value. "
                << "Expected 1, but found " << extractInst->getNumIndices() << ": " << getSourceLocation( originalExtract );
            return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }
        offset = extractInst->getIndices()[0];
    }

    // If the used value is extracted from an aggregate, we need to move the iterator
    // up to the new argument that represents that value.
    // The aggregate has been split up into aggrSize * numRegisters(elementType) i32 values.
    int numVals = corelib::getNumRequiredRegisters( originalExtract->getType() );
    actualIndex += offset * numVals;
    // Find the uses of the value we are actually interested in.
    for( llvm::Value::user_iterator U = originalExtract->user_begin(), UE = originalExtract->user_end(); U != UE; ++U )
    {
        llvm::Instruction* actUse = llvm::dyn_cast<llvm::Instruction>( *U );
        if( !actUse )
        {
            errDetails.m_compilerFeedback << "Error: Invalid use of argument in callable program "
                                          << originalExtract->getParent()->getParent()->getName().str() << ":" <<
                                          llvmToString( *U ) << "\n";

            return OPTIX_ERROR_ILWALID_PTX;
        }
        actualUses.insert( actUse );
    }

    return OPTIX_SUCCESS;
}

// Extracts the values needed newType from the aggregate newArg starting at indexInNewArg and
// colwerts it to newType.
static llvm::Value* colwertArgument( llvm::Type* newType, llvm::Value* newArg, const int indexInNewArg, llvm::Instruction* insertBefore )
{
    corelib::CoreIRBuilder irb{insertBefore};
    llvm::Type*            i32Ty    = irb.getInt32Ty();
    llvm::Value*           newValue = nullptr;
    if( newType != i32Ty && newType->getPrimitiveSizeInBits() == i32Ty->getPrimitiveSizeInBits() )
    {
        newValue = irb.CreateExtractValue( newArg, indexInNewArg );
        newValue = irb.CreateBitCast( newValue, newType );
    }
    else
    {
        newValue = corelib::unflattenAggregateForRTCore( newType, newArg, indexInNewArg, insertBefore );
    }
    return newValue;
}

// Replace all uses of valueToReplace in uses with extracted value from newArg.
static OptixResult replaceParameterUses( llvm::Value*                               newArg,
                                         const int                                  indexInNewArg,
                                         llvm::Value*                               valueToReplace,
                                         const llvm::SetVector<llvm::Instruction*>& uses,
                                         llvm::Value*                               dbgValue )
{
    for( llvm::Instruction* useI : uses )
    {

        if( llvm::PHINode* PN = llvm::dyn_cast<llvm::PHINode>( useI ) )
        {
            unsigned int numIncoming = PN->getNumIncomingValues();
            for( unsigned int i = 0; i < numIncoming; ++i )
            {
                llvm::Value* incoming = PN->getIncomingValue( i );
                if( incoming == valueToReplace )
                {
                    llvm::BasicBlock*  block        = PN->getIncomingBlock( i );
                    llvm::Instruction* insertBefore = &block->back();
                    llvm::Value* newVal = dbgValue ? dbgValue : colwertArgument( valueToReplace->getType(), newArg, indexInNewArg, insertBefore );
                    PN->setIncomingValue( i, newVal );
                }
            }
        }
        else
        {
            llvm::Value* newVal =  dbgValue ? dbgValue : colwertArgument( valueToReplace->getType(), newArg, indexInNewArg, useI );
            useI->replaceUsesOfWith( valueToReplace, newVal );
        }
    }
    return OPTIX_SUCCESS;
}

// Find calls to llvm.dbg.value that represent the old function argument at index oldIndex.
// Returns the llvm.dbg.value call that assigns the initial value of the argument to the DILocalVariable
// in dbgValCall and all other value changes of the DILocalVariable that represents the argument in dbgValChanges.
static OptixResult findCallableParameterDebugInfo( llvm::Function*                newFunction,
                                                   llvm::Value*                   oldArg,
                                                   const int                      oldIndex,
                                                   llvm::DbgValueInst*&           dbgValCall,
                                                   std::set<llvm::DbgValueInst*>& dbgValChanges,
                                                   ErrorDetails&                  errDetails )
{
    dbgValCall = nullptr;
    llvm::Function* dbgValIntrinsic = llvm::Intrinsic::getDeclaration( newFunction->getParent(), llvm::Intrinsic::dbg_value );
    if( dbgValIntrinsic && dbgValIntrinsic->getFunctionType()->getNumParams() != 3 )
    {
        errDetails.m_compilerFeedback << "Error: illegal llvm.dbg.value function in the module.\n";
        return OPTIX_ERROR_ILWALID_PTX;
    }
    std::vector<llvm::CallInst*> dbgValCalls = corelib::getCallsToFunction( dbgValIntrinsic, newFunction );

    // There should only be one call that takes the actual argument as its first operand.
    llvm::DILocalVariable* argDiVariable = nullptr;
    for( llvm::CallInst* ci : dbgValCalls )
    {
        llvm::DbgValueInst* dbgValInst = llvm::dyn_cast<llvm::DbgValueInst>( ci );
        if( !dbgValInst )
            continue;
        llvm::DILocalVariable* lVar = dbgValInst->getVariable();
        if( !lVar->isParameter() || lVar->getArg() != oldIndex + 1 || lVar->getScope() != newFunction->getSubprogram() )
            continue;
        // This llvm.dbg.value call sets the value of the DILocalVariable that represent
        // the old argument.
        llvm::Value* val = dbgValInst->getValue();
        if( dbgValInst->getValue() != oldArg )
        {
            // It is not recording the value of the incoming argument, so it must be a
            // subsequent value change of the argument (i.e. an assignment to the argument).
            // Record this for future processing.
            dbgValChanges.insert( dbgValInst );
            continue;
        }
        // The llvm.dbg.value call records the value of the incoming argument. There should
        // only be a single call of the llvm.dbg.value intrinsic that does this.
        if( dbgValCall )
        {
            errDetails.m_compilerFeedback << "Error: Found multiple llvm.dbg.value calls for the same "
                                             "parameter.\n";
            return OPTIX_ERROR_ILWALID_PTX;
        }
        // Record the initial llvm.dbg.value call. It will be replaced with a new call that takes
        // the extracted value of the rewritten parameter.
        dbgValCall    = dbgValInst;
        argDiVariable = lVar;
    }
    return OPTIX_SUCCESS;
}

// Rewrite the uses of oldArg to use the extracted value from the new aggregate parameter from newArg.
static OptixResult rewriteCallableParameterUses( llvm::Function*                             newFunction,
                                                 llvm::Value*                                oldArg,
                                                 const int                                   oldIndex,
                                                 llvm::Value*                                newArg,
                                                 const int                                   indexInNewArg,
                                                 const llvm::SetVector<llvm::Instruction*>&  uses,
                                                 ErrorDetails&                               errDetails )
{
    llvm::Value* dbgValue = nullptr;
    // Rewrite debug info.
    llvm::DbgValueInst*           dbgValCall = nullptr;
    std::set<llvm::DbgValueInst*> dbgValChanges;
    if( OptixResult res = findCallableParameterDebugInfo( newFunction, oldArg, oldIndex, dbgValCall, dbgValChanges, errDetails ) )
        return res;
    if( dbgValCall )
    {
        if( newFunction->getSubprogram() )
        {
            // Extract the original parameter value from the aggregate and add debug info that pretends that the extracted
            // value is actually a parameter.

            llvm::Module* module = newFunction->getParent();

            // First extract the value from the aggregate. This extracted value will be used for all other uses.
            dbgValue = colwertArgument(oldArg->getType(), newArg, indexInNewArg, dbgValCall);
            dbgValue = corelib::createDebugValueUse(dbgValue, dbgValCall);

            // The original DILocalVariable that will be replaced. We need its info in the new variable.
            llvm::DILocalVariable* oldDiVar = dbgValCall->getVariable();

            // The parameter scope is the function.
            llvm::DIScope* scope = newFunction->getSubprogram();

            llvm::DIBuilder diBuilder(*module);
            // Create a new parameter variable which replaces the old one (just changing the value of dbgValCall to dbgValue did not work).
            llvm::DILocalVariable* newDiVar =
                diBuilder.createParameterVariable(scope, oldDiVar->getName(), oldDiVar->getArg(), oldDiVar->getFile(),
                                                  oldDiVar->getLine(), oldDiVar->getType().resolve(),
                                                  /*AlwaysPreserve=*/false, oldDiVar->getFlags());
            diBuilder.insertDbgValueIntrinsic(dbgValue, newDiVar, diBuilder.createExpression(), dbgValCall->getDebugLoc(), dbgValCall);

            // Replace the old DIVariable with the new one in subsequent llvm.dbg.value intrinsics that used the old one.
            for( llvm::DbgValueInst* ci : dbgValChanges )
            {
                ci->setArgOperand(1, llvm::MetadataAsValue::get(newArg->getContext(), newDiVar));
            }
            // Remove the original llvm.dbg.value call that we just replaced.
            dbgValCall->eraseFromParent();
        }
        else
        {
            // There were cases where noinline functions did not have a DISubprogram associated but still contained
            // llvm.dbg.value calls. Since we need the DISubprogram for the parameter scope, we cannot rewrite them.
            // Remove the intrinsics instead.
            errDetails.m_compilerFeedback << "Warning: Incomplete debug info for function " << newFunction->getName().str()
                                          << ". Compiling function without debug info.\n";
            dbgValCall->eraseFromParent();
            for( llvm::DbgValueInst* ci : dbgValChanges )
                ci->eraseFromParent();
        }
    }

    if( uses.empty() )
        return OPTIX_SUCCESS;
    if( llvm::isa<llvm::SequentialType>( oldArg->getType() ) )
    {
        std::vector<llvm::Instruction*> toDelete;
        for( llvm::Instruction* useI : uses )
        {
            llvm::SetVector<llvm::Instruction*> actualUses;
            // extractActualUses advances the iterator to point to the one that matches the extract for this use.
            int actualIndex = indexInNewArg;
            if( OptixResult res = extractActualUses( useI, oldIndex, actualIndex, actualUses, errDetails ) )
                return res;
            if( actualUses.size() )
            {
                if( OptixResult res = replaceParameterUses( newArg, actualIndex, useI, actualUses, dbgValue ) )
                    return res;
                toDelete.push_back( useI );
            }
            else
            {
                // The use was no extract (e.g. a store). We need to extract the complete old param
                // from the new aggregate and replace the use.
                llvm::SetVector<llvm::Instruction*> tmp;
                tmp.insert( useI );
                if( OptixResult res = replaceParameterUses( newArg, indexInNewArg, oldArg, tmp, dbgValue ) )
                    return res;
            }
        }

        for( llvm::Instruction* useI : toDelete )
            useI->eraseFromParent();
    }
    else
    {
        if( OptixResult res = replaceParameterUses( newArg, indexInNewArg, oldArg, uses, dbgValue ) )
            return res;
    }


    return OPTIX_SUCCESS;
}

static OptixResult colwertParametersToI32Array( llvm::Function*& F, int numDebugParameters, ErrorDetails& errDetails )
{
    // replace byte array args with larger types if possible
    OptixResult res = optix::handleCallableProgramParameters( F, errDetails );
    if( res != OPTIX_SUCCESS )
        return res;

    // Replace function with a version that takes arguments packed
    // as individual i32 values as required by rtcore.

    // Determine the new function type.
    llvm::FunctionType* funcTy             = F->getFunctionType();
    const int           returlwalueNumRegs = corelib::getNumRequiredRegisters( funcTy->getReturnType() );

    int parametersNumRegs = 0;
    for( unsigned int i = 0, e = funcTy->getNumParams(); i < e; ++i )
        parametersNumRegs += corelib::getNumRequiredRegisters( funcTy->getParamType( i ) );

    llvm::FunctionType* newFuncTy = nullptr;
    llvm::Type*         returnTy  = funcTy->getReturnType();
    llvm::Type*         i32Ty     = llvm::Type::getInt32Ty( F->getContext() );
    llvm::Type* newReturnTy       = returnTy->isVoidTy() ? returnTy : llvm::ArrayType::get( i32Ty, returlwalueNumRegs );

    parametersNumRegs += numDebugParameters;

    if( parametersNumRegs == 0 )
    {
        newFuncTy = llvm::FunctionType::get( newReturnTy, false );
    }
    else
    {
        llvm::Type* newParamTy = llvm::ArrayType::get( i32Ty, parametersNumRegs );
        newFuncTy              = llvm::FunctionType::get( newReturnTy, newParamTy, false );
    }

    // Create the new function
    llvm::Function* newFunc = llvm::Function::Create( newFuncTy, F->getLinkage(), "", F->getParent() );
    newFunc->takeName( F );
    newFunc->setAttributes( newFunc->getAttributes().addAttributes( newFunc->getContext(), llvm::AttributeList::FunctionIndex,
                                                                    F->getAttributes().getFnAttributes() ) );
    newFunc->setSubprogram( F->getSubprogram() );

    // Splice the body.
    newFunc->getBasicBlockList().splice( newFunc->begin(), F->getBasicBlockList() );
    if( !F->getBasicBlockList().empty() )
    {
        errDetails.m_compilerFeedback << "Error: Processing callable program " << newFunc->getName().str()
                                      << " failed. Old function not empty.\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }
    if( !newFunc->isDeclaration() )
    {
        if( funcTy->getNumParams() > 0 )
        {
            llvm::Function::arg_iterator aggregateArg = newFunc->arg_begin();
            llvm::Function::arg_iterator origA        = F->arg_begin();

            // Replace uses of the original arguments (origA) by the values in newArgs
            int index     = 0;
            int origIndex = 0;
            // Skip the arguments that were added for the parameter check.
            index += numDebugParameters;
            for( auto origAE = F->arg_end(); origA != origAE; ++origA )
            {
                // Make sure we only generate a single extract if an instruction uses the
                // same argument multiple times by collecting all uses upfront. This also
                // prevents messing up the iteration order by replacing uses.
                llvm::SetVector<llvm::Instruction*> uses;
                for( llvm::Argument::user_iterator U = origA->user_begin(), UE = origA->user_end(); U != UE; ++U )
                {
                    if( !llvm::isa<llvm::Instruction>( *U ) )
                    {
                        errDetails.m_compilerFeedback << "Error: Invalid use of argument " << origIndex
                                                      << " in callable program " << newFunc->getName().str() << ": "
                                                      << llvmToString( *U ) << "\n";
                        return OPTIX_ERROR_ILWALID_PTX;
                    }

                    llvm::Instruction* useI = llvm::cast<llvm::Instruction>( *U );

                    if( useI->getParent()->getParent() != newFunc )
                    {
                        errDetails.m_compilerFeedback << "Error: Argument " << origIndex << " of callable program "
                                                      << newFunc->getName().str() << " is used outside the function."
                                                      << ": " << llvmToString( *U ) << "\n";
                        return OPTIX_ERROR_ILWALID_PTX;
                    }

                    uses.insert( useI );
                }
                res = rewriteCallableParameterUses( newFunc, origA, origIndex, aggregateArg, index, uses, errDetails );
                if( res != OPTIX_SUCCESS )
                    return res;

                index += corelib::getNumRequiredRegisters( origA->getType() );

                ++origIndex;
            }
        }
    }

    // For callable programs, the return type, if non-void, is now an array of
    // i32, so we have to translate the old return value.
    if( !funcTy->getReturnType()->isVoidTy() )
    {
        for( llvm::BasicBlock& BB : *newFunc )
        {
            llvm::ReturnInst* ret = llvm::dyn_cast<llvm::ReturnInst>( BB.getTerminator() );
            if( !ret )
                continue;
            llvm::Value* retVal = ret->getReturlwalue();
            if( !retVal )
            {
                errDetails.m_compilerFeedback << "Error: Missing return value in non-void function " << F->getName().str()
                                              << ": " << getSourceLocation( F->getParent(), ret ) << "\n";
                return OPTIX_ERROR_ILWALID_PTX;
            }

            std::vector<llvm::Value*> retVals = corelib::flattenAggregateTo32BitValuesForRTCore( retVal, ret );

            // Insert return values into return value array.
            llvm::Value*           retArray = llvm::UndefValue::get( newFuncTy->getReturnType() );
            corelib::CoreIRBuilder irb{ret};
            for( int retValIdx = 0, e = (int)retVals.size(); retValIdx < e; ++retValIdx )
            {
                llvm::Value* retValElement = retVals[retValIdx];
                if( retValElement->getType() != i32Ty )
                    retValElement = irb.CreateBitCast( retValElement, i32Ty );
                retArray          = irb.CreateInsertValue( retArray, retValElement, retValIdx );
            }

            irb.CreateRet( retArray );
            ret->eraseFromParent();
        }
    }

    rewriteCalleeParametersToI32Array( F, newFunc, errDetails );

    // Move any LWVM metadata from the old to the new function.
    if( llvm::NamedMDNode* lwvmMd = F->getParent()->getNamedMetadata( "lwvm.annotations" ) )
        corelib::replaceMetadataUses( lwvmMd, F, newFunc );

    // Erase the original function.
    if( !F->use_empty() )
    {
        errDetails.m_compilerFeedback
            << "Error: Callable program was rewritten, but old function is still used: " << newFunc->getName().str() << "\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }

    F->eraseFromParent();

    F = newFunc;

    return OPTIX_SUCCESS;
}

static OptixResult rewriteCallablesAndNoInline( CompilationUnit&                 module,
                                                const InternalCompileParameters& compileParams,
                                                ErrorDetails&                    errDetails )
{
    // Rewrite callables to the requirements of rtcore.
    llvm::Type*        i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
    OptixResultOneShot result;

    int  inlineCount                = 0;
    bool hasNonDefaultInliningKnobs = false;
    for( llvm::Function* F : corelib::getFunctions( module.llvmModule ) )
    {
        optix_exp::SemanticType stype = optix_exp::getSemanticTypeForFunction( F, module.compileParams.noinlineEnabled,
                                                                               module.compileParams.disableNoinlineFunc );
        // standalone functione, ie functions witout any semantic type, will be handled as if noinline would be enabled
        // this is done to colwert their parameters to i32 array type to avoid loosing alignment
        bool isStandaloneFunc = false;
        if( stype == optix_exp::ST_ILWALID )
            isStandaloneFunc = optix_exp::getSemanticTypeForFunction( F, true, "" ) == optix_exp::ST_NOINLINE;
        if( stype == optix_exp::ST_DIRECT_CALLABLE || stype == optix_exp::ST_CONTINUATION_CALLABLE
            || stype == optix_exp::ST_NOINLINE || isStandaloneFunc )
        {
            if( stype != optix_exp::ST_NOINLINE && !isStandaloneFunc )
            {
                if( F->isDeclaration() )
                    continue;
                if( !F->use_empty() )
                {
                    errDetails.m_compilerFeedback << "Error: It is illegal to call callables by name. Found calls to "
                                                  << F->getName().str() << "\n";
                    result += OPTIX_ERROR_ILWALID_PTX;
                }
            }
            else if( !isStandaloneFunc )
            {
                if( module.compileParams.removeUnusedNoinlineFunctions && F->use_empty() )
                {

                    F->eraseFromParent();
                    continue;
                }
                else
                {
                    int instructionCount       = 0;
                    int inlineInstructionLimit = module.compileParams.inlineInstructionLimit;
                    if( inlineInstructionLimit != -1 )
                    {
                        hasNonDefaultInliningKnobs = true;
                        for( const llvm::BasicBlock& bb : *F )
                        {
                            instructionCount += std::distance( bb.begin(), bb.end() );
                        }
                    }
                    size_t callCount           = std::distance( F->user_begin(), F->user_end() );
                    int    inlineCallLimitLow  = module.compileParams.inlineCallLimitLow;
                    int    inlineCallLimitHigh = module.compileParams.inlineCallLimitHigh;
                    if( inlineCallLimitHigh != -1 || inlineCallLimitLow != -1 )
                        hasNonDefaultInliningKnobs = true;

                    bool belowInlineInstructionLimit = inlineInstructionLimit == -1 || ( instructionCount < inlineInstructionLimit );
                    bool callCountBelowLimitLow    = inlineCallLimitLow != -1 && ( callCount < inlineCallLimitLow );
                    bool callCountExceedsLimitHigh = inlineCallLimitHigh != -1 && ( callCount > inlineCallLimitHigh );
                    if( belowInlineInstructionLimit && ( callCountBelowLimitLow || callCountExceedsLimitHigh ) )
                    {
                        corelib::inlineAllCalls( F );
                        F->eraseFromParent();
                        ++inlineCount;
                        continue;
                    }
                }
            }

            // If debug exceptions are enabled, four additional parameters are added for
            // the argument count, the sbt index and a pointer to the string that includes
            // the caller site.
            int numDebugParameters = 0;
            if( stype != ST_NOINLINE && compileParams.enableCallableParamCheck && !isStandaloneFunc )
                numDebugParameters = compileParams.paramCheckExceptionRegisters;

            OptixResult res = colwertParametersToI32Array( F, numDebugParameters, errDetails );

            if( numDebugParameters > 0 )
            {
                // Generate parameter count check.
                llvm::Function::arg_iterator aggregateArg = F->arg_begin();
                llvm::Instruction*           insertBefore = corelib::getFirstNonAlloca( F );

                llvm::Value* functionName = addModuleString( module, F->getName() );
                std::pair<llvm::Value*, llvm::Value*> splitFunctionName =
                    corelib::createCast_i64_to_2xi32( functionName, insertBefore );

                corelib::CoreIRBuilder irb{insertBefore};

                int          parametersNumRegs = static_cast<int>( aggregateArg->getType()->getArrayNumElements() );
                llvm::Value* argCount          = irb.getInt32( parametersNumRegs - numDebugParameters );
                llvm::Value* passedArgCount    = irb.CreateExtractValue( aggregateArg, 0 );
                llvm::Value* sbtIndex          = irb.CreateExtractValue( aggregateArg, 1 );
                llvm::Value* descriptionLow    = irb.CreateExtractValue( aggregateArg, 2 );
                llvm::Value* descriptionHi     = irb.CreateExtractValue( aggregateArg, 3 );

                llvm::Value* details = llvm::UndefValue::get( llvm::ArrayType::get( i32Ty, RTC_NUM_EXCEPTION_DETAILS ) );
                details              = irb.CreateInsertValue( details, argCount, 0 );
                details              = irb.CreateInsertValue( details, passedArgCount, 1 );
                details              = irb.CreateInsertValue( details, sbtIndex, 2 );
                details              = irb.CreateInsertValue( details, splitFunctionName.first, 3 );
                details              = irb.CreateInsertValue( details, splitFunctionName.second, 4 );

                // Insert the exception source location at the end of the array. Note that we cannot use the
                // insertExceptionSourceLocation function here because we get the source location passed from the call site.
                details = irb.CreateInsertValue( details, descriptionLow, RTC_NUM_EXCEPTION_DETAILS - 2 );
                details = irb.CreateInsertValue( details, descriptionHi, RTC_NUM_EXCEPTION_DETAILS - 1 );

                llvm::Value* condition = irb.CreateICmpNE( argCount, passedArgCount );
                result += generateConditionalThrowOptixException( module, condition, OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH,
                                                                  insertBefore, details, errDetails );
            }
        }
    }

    if( hasNonDefaultInliningKnobs )
        errDetails.m_compilerFeedback << "Compiler knobs caused inlining of " << inlineCount << " function(s).\n";

    return result;
}


static llvm::MDNode* createLwstomABIMetadataNode( llvm::LLVMContext& context,
                                                  int                numParamRegs,
                                                  int                firstParamReg,
                                                  int                firstReturn,
                                                  int                numAdditionalScratch,
                                                  int                numScratchBarrier,
                                                  int                smVersion )
{
    using namespace llvm;

    std::vector<MetadataValueTy*> values;
    llvm::Type*                   i32Ty = llvm::Type::getInt32Ty( context );

    values.push_back( MDString::get( context, "firstParam" ) );
    values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, firstParamReg ) ) );
    values.push_back( MDString::get( context, "numParams" ) );
    values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, numParamRegs ) ) );

    // Volta uses normal registers for the return addresses, Turing uses uniform (UR) regs except for AH/DC.
    // Earlier architectures don't have return address registers, so we don't add any metadata.
    if( smVersion >= 70 )
    {
        // If you want to use uniform return address registers use "firstReturnU".  We
        // don't actually want to do that here, though.
        values.push_back( MDString::get( context, "firstReturn" ) );
        values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, firstReturn ) ) );
    }

    // scratch starts at param registers and goes till total scratch is reached
    values.push_back( MDString::get( context, "scratchR" ) );
    values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, firstParamReg ) ) );
    values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, firstParamReg + numParamRegs + numAdditionalScratch ) ) );

    if( smVersion >= 70 )
    {
        values.push_back( MDString::get( context, "scratchCB" ) );
        values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, 0 ) ) );
        values.push_back( UseValueAsMd( ConstantInt::get( i32Ty, numScratchBarrier ) ) );
    }

    return MDNode::get( context, values );
}

static OptixResult rewriteToLwstomABI( llvm::Function* F,
                                       int             firstParamReg,
                                       int             firstReturn,
                                       int             numAdditionalScratch,
                                       int             numScratchBarrier,
                                       int             smVersion,
                                       ErrorDetails&   errDetails )
{
    llvm::LLVMContext&  llvmContext  = F->getContext();
    llvm::NamedMDNode*  lwvmannotate = F->getParent()->getOrInsertNamedMetadata( "lwvm.annotations" );
    llvm::MDString*     str          = llvm::MDString::get( llvmContext, "full_lwstom_abi" );
    llvm::Type*         i8PtrTy      = llvm::Type::getInt8Ty( llvmContext )->getPointerTo();
    llvm::Type*         caArgTys[]   = {i8PtrTy, llvm::Type::getMetadataTy( llvmContext )};
    llvm::FunctionType* caFnType     = llvm::FunctionType::get( i8PtrTy, caArgTys, false );
    llvm::Function*     caFunc =
        llvm::cast<llvm::Function>( F->getParent()->getOrInsertFunction( "llvm.lwvm.full.custom.abi.call", caFnType ) );


    // replace byte array args with larger types if possible
    if( OptixResult result = colwertParametersToI32Array( F, /*numDebugParameters*/ 0, errDetails ) )
        return result;

    ///////////////////////
    // Create metadata node

    llvm::FunctionType* funcTy = F->getFunctionType();
    // Number of parameters is the union of the input and output parameters
    int numReturnRegs     = funcTy->getReturnType()->isVoidTy() ? 0 : funcTy->getReturnType()->getArrayNumElements();
    int numParamRegs      = funcTy->getNumParams() == 0 ? 0 : ( *funcTy->param_begin() )->getArrayNumElements();
    int numABIParamRegs   = std::max( numParamRegs, numReturnRegs );
    llvm::MDNode* ABINode = createLwstomABIMetadataNode( llvmContext, numABIParamRegs, firstParamReg, firstReturn,
                                                         numAdditionalScratch, numScratchBarrier, smVersion );

    ///////////////////////
    // Add metadata node to callee

    MetadataValueTy* values[] = {UseValueAsMd( F ), str, ABINode};
    llvm::MDNode*    mdNode   = llvm::MDNode::get( llvmContext, values );
    lwvmannotate->addOperand( mdNode );

    ///////////////////////
    // Add metadata to all callers

    // Create the function type so we can cast the pointer.
    // NOTE: For some reason, LWVM can't handle array return types for prototypes,
    //       so we wrap the array in a struct. The offsets are not affected by this,
    //       so the resulting code should be the same either way.
    // NOTE2: bigler: I'm copying this from rtcore.

    llvm::FunctionType* fTy = funcTy;
    if( !funcTy->getReturnType()->isVoidTy() )
    {
        llvm::Type* retTy = llvm::StructType::get( llvmContext, funcTy->getReturnType(), false );
        llvm::SmallVector<llvm::Type*, 8> paramTy( funcTy->param_begin(), funcTy->param_end() );
        fTy = llvm::FunctionType::get( retTy, paramTy, false );
    }

    for( llvm::CallInst* call : corelib::getCallsToFunction( F ) )
    {
        corelib::CoreIRBuilder builder{call};
        llvm::Value*           funcVoidPtr = builder.CreateBitCast( F, i8PtrTy );
        llvm::Value*           caArgs[]    = {funcVoidPtr, UseMdAsValue( llvmContext, ABINode )};
        llvm::CallInst*        caCall      = builder.CreateCall( caFunc, caArgs );
        llvm::Value*           func        = builder.CreateBitCast( caCall, fTy->getPointerTo(), "func.ptr" );

        std::vector<llvm::Value*> args;
        if( call->getNumArgOperands() )
            args.push_back( call->getArgOperand( 0 ) );
        llvm::Value* newCI = builder.CreateCall( func, args );
        if( !fTy->getReturnType()->isVoidTy() )
            newCI = builder.CreateExtractValue( newCI, 0 );
        newCI->takeName( call );
        call->replaceAllUsesWith( newCI );
        call->eraseFromParent();
    }

    return OPTIX_SUCCESS;
}


static OptixResult rewriteFunctionsForLwstomABI( CompilationUnit& module, ErrorDetails& errDetails )
{
    if( !module.compileParams.enableLwstomABIProcessing )
        return OPTIX_SUCCESS;

    OptixResultOneShot result;

    // 1. Identify all the functions we need to prepare
    // 2. Make sure callees and callsites have the correct signature [N x i32] func([M x i32])
    // 3. Create meta data node describing the custom ABI
    // 4. Add metadata node for the callee
    // 5. Adjust call site to annotate with ABI information

    // This will be a mix of forward declarations and function definitions
    std::vector<llvm::Function*> callees;

    for( llvm::Function* F : corelib::getFunctions( module.llvmModule ) )
    {
        if( !corelib::stringBeginsWith( F->getName().str(), "_lw_optix_lwstom_abi_" ) )
            continue;
        callees.push_back( F );
    }

    if( callees.empty() )
        return result;

    for( llvm::Function* F : callees )
    {
        int firstParamReg        = 8;
        int firstReturn          = 6;
        int numAdditionalScratch = module.compileParams.numAdditionalABIScratchRegs;
        int numScratchBarrier    = 1;

        if( OptixResult res = rewriteToLwstomABI( F, firstParamReg, firstReturn, numAdditionalScratch, numScratchBarrier,
                                                  module.compileParams.maxSmVersion, errDetails ) )
        {
            result += res;
            continue;
        }
    }

    return result;
}

// Delete dead function annotations from the given annotation node.
// 
// annotationNode: the metadata node containing function annotations ("lwvm.annotations" or "lwvm.rt.annotations")
// annotationType: the type of annotation to remove, if dead (e.g. "directcallable")
static void deleteDeadAnnotations( llvm::NamedMDNode* annotationNode, const llvm::StringRef& annotationType )
{
    std::vector<llvm::MDNode*> operandsToRemove;

    for( unsigned int i = 0; i < annotationNode->getNumOperands(); ++i )
    {
        llvm::MDNode* lwrrOperand = annotationNode->getOperand( i );

        // custom ABI annotations have the string "full_lwstom_abi" as their second operand
        if( lwrrOperand->getNumOperands() < 2 )
            continue;
        llvm::MDString* secondOpAsString = llvm::dyn_cast<llvm::MDString>( lwrrOperand->getOperand( 1 ) );
        if( !secondOpAsString || !secondOpAsString->getString().equals( annotationType ) )
            continue;

        // Dead annotations have null as their function operand
        if( lwrrOperand->getOperand( 0 ) != nullptr )
            continue;

        operandsToRemove.push_back( lwrrOperand );
    }

    for( llvm::MDNode* op : operandsToRemove )
        annotationNode->removeOperand( op );
}

static OptixResult deleteDeadLwstomABIAnnotations( CompilationUnit& module, ErrorDetails& errDetails )
{
    if( !module.compileParams.enableLwstomABIProcessing )
        return OPTIX_SUCCESS;

    llvm::NamedMDNode* annotationNode = module.llvmModule->getNamedMetadata( "lwvm.annotations" );
    if( !annotationNode )
        return OPTIX_SUCCESS;

    deleteDeadAnnotations(annotationNode, "full_lwstom_abi" );

    return OPTIX_SUCCESS;
}

static OptixResult deleteDeadDirectCallableAnnotations( CompilationUnit& module, ErrorDetails& errDetails )
{
    // Direct callables can only be inlined and DCE'd if the noinline extension is enabled.
    if( !module.compileParams.noinlineEnabled )
        return OPTIX_SUCCESS;

    llvm::NamedMDNode* annotationNode = module.llvmModule->getNamedMetadata( "lwvm.rt.annotations" );
    if( !annotationNode )
        return OPTIX_SUCCESS;

    deleteDeadAnnotations(annotationNode, "directcallable" );

    return OPTIX_SUCCESS;
}

// When the function has exactly one return statement, return true and the BB containing the ret.
// Return false when no or multiple return statements were found.
static bool hasExactlyOneReturn( llvm::Function* function, llvm::BasicBlock** endBlock )
{
    *endBlock = nullptr;
    for( llvm::Function::iterator BB = function->begin(), BBE = function->end(); BB != BBE; ++BB )
    {
        if( !llvm::isa<llvm::ReturnInst>( BB->getTerminator() ) )
            continue;
        if( *endBlock )
            return false;
        *endBlock = &( *BB );
    }
    return *endBlock != nullptr;
}

// The -mergereturn pass does not seem to be exposed in the API - hence implementing it here.
static void mergeAllReturnStatements( CompilationUnit& module, llvm::Function* function, llvm::BasicBlock** endBlock )
{
    *endBlock = llvm::BasicBlock::Create( module.llvmModule->getContext(), "O7_SingleReturnBlock", function );
    corelib::CoreIRBuilder{*endBlock}.CreateRetVoid();

    for( llvm::Function::iterator BB = function->begin(), BBE = function->end(); BB != BBE; ++BB )
    {
        if( !llvm::isa<llvm::ReturnInst>( BB->getTerminator() ) )
            continue;
        llvm::TerminatorInst*  ti = BB->getTerminator();
        corelib::CoreIRBuilder irb{ti};
        irb.CreateBr( *endBlock );
        ti->eraseFromParent();
    }
}

// Add the hard stop via trap().
static void addTrapToBB( CompilationUnit& module, llvm::BasicBlock* BB, llvm::Value* exceptionCode )
{
    // TODO: this will be removed or replaced by some better handling
    //corelib::dbgPrint( exceptionCode, BB->getTerminator(), "Validation Mode Exit due to builtin exception with printed code...\n" );
    corelib::CoreIRBuilder irb{BB->getTerminator()};
    llvm::Type*            voidTy = llvm::Type::getVoidTy( module.llvmModule->getContext() );
    llvm::Function*        trap =
        corelib::insertOrCreateFunction( module.llvmModule, "llvm.trap", llvm::FunctionType::get( voidTy, {} ) );
    irb.CreateCall( trap );
}

// Make a thrown builtin exceptions as visible as possible by calling trap() at the end of each exception program.
// As this function gets called only when in validation mode, we can manipulate the exceptions programs here.
static OptixResult instrumentExceptionProgramsForValidationMode( CompilationUnit& module, ErrorDetails& errDetails )
{
    if( !module.compileParams.validationModeDebugExceptions )
        return OPTIX_SUCCESS;

    std::string gvName = "__optixValidationModeExceptionCode";
    // check that global variable's name isn't used by some user programs
    if( module.llvmModule->getNamedValue( gvName ) )
    {
        errDetails.m_compilerFeedback << "Error: Found usage of reserved name " << gvName.c_str() << "\n";
        return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
    }

    llvm::GlobalVariable* pinnedMemory = nullptr;
    for( llvm::Function* function : corelib::getFunctions( module.llvmModule ) )
    {
        SemanticType stype = getSemanticTypeForFunction( function, module.compileParams.noinlineEnabled,
                                                         module.compileParams.disableNoinlineFunc );
        if( stype != ST_EXCEPTION )
            continue;

        llvm::Type* i32Ty = llvm::Type::getInt32Ty( module.llvmModule->getContext() );
        if( !module.llvmModule->getNamedValue( gvName ) )
        {
            // add GV`s name to avoid linking issues
            module.addNamedConstantName( gvName );

            llvm::Type* ptrTy = llvm::PointerType::get( i32Ty, 0 );
            pinnedMemory = new llvm::GlobalVariable( *module.llvmModule,  // Global inserted at end of module globals list
                                                     ptrTy,               // type of the variable
                                                     false,               // is this variable constant
                                                     llvm::GlobalValue::ExternalLinkage,    // symbol linkage
                                                     nullptr,                               // Static initializer
                                                     gvName,                                // Name
                                                     nullptr,                               // InsertBefore
                                                     llvm::GlobalVariable::NotThreadLocal,  // Thread local
                                                     corelib::ADDRESS_SPACE_CONST );  // The variable's address space
        }

        // ensure one return
        llvm::BasicBlock* endBlock{};
        if( !hasExactlyOneReturn( function, &endBlock ) )
        {
            mergeAllReturnStatements( module, function, &endBlock );
        }
        if( !llvm::isa<llvm::ReturnInst>( endBlock->getTerminator() ) )
        {
            errDetails.m_compilerFeedback << "No return in exception program, so skipping termination on exception "
                                             "program invocation.\n";
            return OPTIX_ERROR_ILWALID_PTX;
        }

        // add builtin exception handling block which calls trap
        llvm::BasicBlock* origBlock = endBlock->splitBasicBlock( endBlock->getTerminator(), "OriginalCode" );
        llvm::BasicBlock* exBlock   = endBlock->splitBasicBlock( endBlock->getTerminator(), "HandleBuiltinException" );

        // add condition whether exception was a builtin one
        llvm::TerminatorInst*  ti = endBlock->getTerminator();
        corelib::CoreIRBuilder irb{ti};
        // condition is based on outcome of optixGetExceptionCode()
        llvm::FunctionType* funcTy = llvm::FunctionType::get( i32Ty, false );
        llvm::Function*     readExceptionCodeFunction =
            corelib::insertOrCreateFunction( module.llvmModule, "lw.rt.read.exception.code", funcTy );
        llvm::Value* exceptionCode = irb.CreateCall( readExceptionCodeFunction );
        llvm::Value* condition     = irb.CreateICmpSLT( exceptionCode, irb.getInt32( 0 ) );
        irb.CreateCondBr( condition, exBlock, origBlock );
        ti->eraseFromParent();

        // and finally store exception code and add trap call to exit
        corelib::CoreIRBuilder irbStore{exBlock->getTerminator()};
        llvm::Value*           Idx       = static_cast<llvm::Value*>( llvm::ConstantInt::get( i32Ty, 0 ) );
        llvm::Value*           ptrInst   = irbStore.CreateGEP( static_cast<llvm::Value*>( pinnedMemory ), Idx );
        llvm::LoadInst*        loadInst  = irbStore.CreateLoad( ptrInst );
        llvm::StoreInst*       storeInst = irbStore.CreateStore( exceptionCode, loadInst );

        addTrapToBB( module, exBlock, exceptionCode );
    }

    return OPTIX_SUCCESS;
}

static bool isCallableCall( InitialCompilationUnit& module, llvm::CallInst* callInst )
{
    if( callInst->getCalledFunction() )
        return false;
    llvm::Value* val = callInst->getCalledValue();
    if( !val )
        return false;
    llvm::IntToPtrInst* intToPtr = llvm::dyn_cast<llvm::IntToPtrInst>( val );
    if( !intToPtr )
        return false;
    llvm::Value* op = intToPtr->getOperand( 0 );
    if( !op )
        return false;
    llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>( op );
    if( !call )
        return false;
    llvm::Function* func = call->getCalledFunction();
    if( !func )
        return false;

    if( func != module.llvmIntrinsics[optix_call_direct_callable] && func != module.llvmIntrinsics[optix_call_continuation_callable] )
        return false;

    return true;
}

static bool isIllegalSyncIntrinsic(llvm::Intrinsic::ID id)
{
    // Intrinsic IDs are based on _out/*/include/llvm/IR/Intrinsics.gen for LWVM34
    // and _out/*/include/llvm/IR/IntrinsicEnums.inc for LWVM7.
    // There seem to be no intrinsic ids for fence.
    switch( id )
    {
        // MMA
        case llvm::Intrinsic::lwvm_mma:
        case llvm::Intrinsic::lwvm_mma_ld:
        case llvm::Intrinsic::lwvm_mma_st:

        // BAR
        case llvm::Intrinsic::lwvm_bar_sync:
        case llvm::Intrinsic::lwvm_bar_sync_all:
        case llvm::Intrinsic::lwvm_bar_sync_all_cnt:
        case llvm::Intrinsic::lwvm_bar_warp_sync:

        // BARRIER
        case llvm::Intrinsic::lwvm_barrier_n:
        case llvm::Intrinsic::lwvm_barrier:
        case llvm::Intrinsic::lwvm_barrier_red:
        case llvm::Intrinsic::lwvm_barrier_sync:
        case llvm::Intrinsic::lwvm_barrier_sync_cnt:
        case llvm::Intrinsic::lwvm_barrier0:
        case llvm::Intrinsic::lwvm_barrier0_and:
        case llvm::Intrinsic::lwvm_barrier0_or:
        case llvm::Intrinsic::lwvm_barrier0_popc:

        // MEMBAR
        case llvm::Intrinsic::lwvm_membar:
        case llvm::Intrinsic::lwvm_membar_cta:
        case llvm::Intrinsic::lwvm_membar_gl:
        case llvm::Intrinsic::lwvm_membar_sys:

        // WMMA
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_a_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_a_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_a_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_a_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_b_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_b_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_b_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_b_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_load_c_f32_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f16_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f16_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f32_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_col_f32_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f16_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f16_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f32_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_col_row_f32_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f16_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f16_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f32_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_col_f32_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f16_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f16_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f32_f16_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_mma_row_row_f32_f32_satfinite:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m16n16k16_store_d_f32_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_a_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_a_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_a_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_a_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_b_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_b_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_b_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_b_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_load_c_f32_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_col_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_col_row_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_col_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_mma_row_row_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m32n8k16_store_d_f32_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_a_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_a_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_a_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_a_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_b_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_b_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_b_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_b_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_load_c_f32_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_col_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_col_row_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_col_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f16_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f16_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f16_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f16_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f32_f16:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f32_f16_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f32_f32:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_mma_row_row_f32_f32_satfinite: case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f16_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f32_col:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f16_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f32_col_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f16_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f32_row:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f16_row_stride:
        case llvm::Intrinsic::lwvm_wmma_m8n32k16_store_d_f32_row_stride:
            return true;
        default:
            return false;
    }
}

static OptixResult validateIllegalSyncFunctions( llvm::Module* llvmModule, ErrorDetails& errDetails )
{
    OptixResultOneShot result;

    llvm::Regex illegalPatterns[] = { llvm::Regex( "^wmma\\..*" ),
                                      llvm::Regex( "^mma\\..*" ),
                                      llvm::Regex( "^bar\\..*" ),
                                      llvm::Regex( "^barrier\\..*" ),
                                      llvm::Regex( "^membar(\\..*)?" ),
                                      llvm::Regex( "^fence(\\..*)?" ),
                                      llvm::Regex( "^ld\\..*(relaxed|acquire)\\..*" ),
                                      llvm::Regex( "^st\\..*(relaxed|release)\\..*" ) };

    for( llvm::Function& function : *llvmModule )
    {
        bool isIllegal = false;
        if( function.isIntrinsic() )
        {
            isIllegal = isIllegalSyncIntrinsic( static_cast<llvm::Intrinsic::ID>( function.getIntrinsicID() ) );
        }
        else
        {
            if( !function.getName().startswith( "optix.ptx." ) )
                continue;
            llvm::StringRef subName = function.getName().substr( 10 );
            for( llvm::Regex& pattern : illegalPatterns )
            {
                if( pattern.match( subName ) )
                {
                    isIllegal = true;
                    break;
                }
            }
        }
        if( isIllegal )
        {
            std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction( &function );
            if( calls.empty() )
                continue;
            result += OPTIX_ERROR_ILWALID_PTX;
            errDetails.m_compilerFeedback << "Error: Function " << function.getName().str() << " is not supported. Called from: ";
            auto it = calls.begin();
            errDetails.m_compilerFeedback << ( *it )->getParent()->getParent()->getName().str() << " ("
                                          << getSourceLocation( llvmModule, *it ) << ")";
            ++it;
            for( auto end = calls.end(); it != end; ++it )
            {
                errDetails.m_compilerFeedback << ", " << ( *it )->getParent()->getParent()->getName().str() << " ("
                                                << getSourceLocation( llvmModule, *it ) << ")";
            }
            errDetails.m_compilerFeedback << "\n";
        }
    }

    return result;
}


static OptixResult validateSyncFunctionsActiveMaskUsage( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    llvm::Function* amFunc = module.llvmModule->getFunction( "optix.ptx.activemask.b32" );

    // Optix 7 intrinsics that are illegal in between the call to activemask and the sync function
    IntrinsicIndex illegalIntrinsics[] = {optix_trace_0,
                                          optix_trace_1,
                                          optix_trace_2,
                                          optix_trace_3,
                                          optix_trace_4,
                                          optix_trace_5,
                                          optix_trace_6,
                                          optix_trace_7,
                                          optix_trace_8,
                                          optix_trace_32,
                                          optix_trace_typed_32,
                                          optix_call_continuation_callable,
                                          optix_call_direct_callable,
                                          optix_get_triangle_vertex_data};

    // function names that are legal in between __activemask and the *.sync.* function
    llvm::Regex whiteList( "(_optix_.+|optix\\.ptx\\..+|optix\\.lwvm\\..+|llvm\\..+|lw\\.rt\\..+|vprintf)" );
    // functions that are illegal but would be matched by the whitelist
    llvm::Regex blackList( "llvm\\.(trap|lwca\\.syncthreads)" );

    OptixResultOneShot result;

    // warp level sync functions that take a membermask and the argument index of that mask.
    std::array<std::pair<llvm::Function*, int>, 10> functions = {
        {{module.llvmModule->getFunction( "llvm.lwvm.vote.sync" ), 0},
         {module.llvmModule->getFunction( "llvm.lwvm.match.all.sync.i32" ), 0},
         {module.llvmModule->getFunction( "llvm.lwvm.match.all.sync.i64" ), 0},
         {module.llvmModule->getFunction( "llvm.lwvm.match.any.sync.i32" ), 0},
         {module.llvmModule->getFunction( "llvm.lwvm.match.any.sync.i64" ), 0},
         {module.llvmModule->getFunction( "llvm.lwvm.shfl.sync.i32" ), 0},
         {module.llvmModule->getFunction( "optix.ptx.shfl.sync.up.b32" ), 3},
         {module.llvmModule->getFunction( "optix.ptx.shfl.sync.down.b32" ), 3},
         {module.llvmModule->getFunction( "optix.ptx.shfl.sync.bfly.b32" ), 3},
         {module.llvmModule->getFunction( "optix.ptx.shfl.sync.idx.b32" ), 3}}};
    for( const std::pair<llvm::Function*, int>& function : functions )
    {
        llvm::Function* func = function.first;
        std::vector<llvm::CallInst*> calls = corelib::getCallsToFunction(func);
        for( llvm::CallInst* call : calls )
        {
            // make sure that the membermask is the result of a call to __activemask
            llvm::Value* val = call->getArgOperand( function.second );
            if( !llvm::isa<llvm::CallInst>( val ) )
            {
                result += OPTIX_ERROR_ILWALID_PTX;
                errDetails.m_compilerFeedback
                    << "Error: Mask argument to " << func->getName().str()
                    << " is not the result of a call to __activemask: " << getSourceLocation( module.llvmModule, call ) << "\n";
                continue;
            }

            llvm::CallInst* maskCall = llvm::cast<llvm::CallInst>( val );
            if( maskCall->getCalledFunction() != amFunc )
            {
                result += OPTIX_ERROR_ILWALID_PTX;
                errDetails.m_compilerFeedback << "Error: Mask argument to " << func->getName().str()
                                              << " is not the result of a call to __activemask "
                                              << getSourceLocation( module.llvmModule, call ) << "\n";
                continue;
            }

            // Note: By requiring that the membermask is a direct result of a CallInst to amFun
            // (especially no PHI-node) it is ensured that the activemask dominates the sync call.
            // This means that the CFG traversal below will reach the mask-call or its basic block on all paths.

            // Make sure that no evil things are called in between the call to __activemask and the *.sync.* function.
            // Iterate up the CFG starting at the sync-function call until we hit the mask call.
            llvm::SetVector<llvm::BasicBlock*> visited;
            std::vector<llvm::BasicBlock*>     stack{call->getParent()};
            while( !stack.empty() )
            {
                llvm::BasicBlock* bb = stack.back();
                stack.pop_back();
                if( !visited.insert( bb ) )
                    continue;
                llvm::BasicBlock::reverse_iterator it = bb->rbegin();
                if( bb == call->getParent() )
                    it = llvm::BasicBlock::reverse_iterator( call );

                // Flag whether we want to iterate up the CFG more. We can stop once
                // we either hit the activemask call or when we reach an illegal call.
                bool goOn = true;
                for( llvm::BasicBlock::reverse_iterator end = bb->rend(); it != end; ++it )
                {
                    llvm::Instruction* instruction = &( *it );
                    if( instruction == maskCall )
                    {
                        goOn = false;
                        break;
                    }
                    if( llvm::CallInst* c = llvm::dyn_cast<llvm::CallInst>( instruction ) )
                    {
                        llvm::Function* calledFunc = c->getCalledFunction();
                        if( !calledFunc )
                        {
                            if( isCallableCall( module, c ) )
                            {
                                // Only report error for callable program call, indirect call in general
                                // is illegal anyways and already produces a different error message.
                                errDetails.m_compilerFeedback << "Error: Callable program call is "
                                                                 "illegal between the call to __activemask and "
                                                              << func->getName().str() << ": "
                                                              << getSourceLocation( module.llvmModule, c ) << "\n";
                            }
                            result += OPTIX_ERROR_ILWALID_PTX;
                            goOn = false;
                            break;
                        }

                        // Check for optix 7 intrinsics
                        // Disallow continuations and direct calls in between activemask and function call.
                        for( IntrinsicIndex illegalIntrinsicIdx : illegalIntrinsics )
                        {
                            llvm::Function* illegalIntrinsic = module.llvmIntrinsics[illegalIntrinsicIdx];
                            if( illegalIntrinsic == calledFunc )
                            {
                                errDetails.m_compilerFeedback << "Error: Call to " << apiName( illegalIntrinsicIdx )
                                                              << " is illegal between the call to __activemask and "
                                                              << func->getName().str() << ": "
                                                              << getSourceLocation( module.llvmModule, call ) << "\n";
                                result += OPTIX_ERROR_ILWALID_PTX;
                                goOn = false;
                            }
                        }
                        if( !goOn )
                            break;

                        // validate the function using the whitelist and the blacklist.
                        llvm::StringRef calledFuncName = calledFunc->getName();
                        goOn = !blackList.match( calledFuncName ) && whiteList.match( calledFuncName );
                        if( !goOn )
                        {
                            errDetails.m_compilerFeedback << "Error: Call to " << calledFuncName.str()
                                                          << " is illegal between the call to __activemask and "
                                                          << func->getName().str() << ": "
                                                          << getSourceLocation( module.llvmModule, call ) << "\n";
                            result += OPTIX_ERROR_ILWALID_PTX;
                            break;
                        }
                    }
                }
                if( goOn )
                {
                    for( llvm::pred_iterator P = llvm::pred_begin( bb ), PE = llvm::pred_end( bb ); P != PE; ++P )
                        stack.push_back( *P );
                }
            }
        }
    }
    return result;
}

static OptixResult validateSharedMemoryAccess( llvm::Module* llvmModule, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    for( auto it = llvmModule->global_begin(), end = llvmModule->global_end(); it != end; ++it )
    {
        if( it->getType()->getPointerAddressSpace() == corelib::ADDRESS_SPACE_SHARED )
        {
            errDetails.m_compilerFeedback << "Error: Global variable " << it->getName().str()
                                          << " is using shared memory which is illegal.\n";
            result += OPTIX_ERROR_ILWALID_PTX;
        }
    }
    return result;
}

static const llvm::MDNode* getNamedLwvmMetdataNode( llvm::Module* llvmModule, const std::string& name )
{
    llvm::NamedMDNode* lwvmMd = llvmModule->getNamedMetadata( "lwvm.annotations" );
    if( !lwvmMd )
        return nullptr;
    for( unsigned int i = 0, e = lwvmMd->getNumOperands(); i != e; ++i )
    {
        const llvm::MDNode* elem = lwvmMd->getOperand( i );
        if( !elem || elem->getNumOperands() < 2 )
            continue;
        const llvm::MDString* stypeMD = llvm::dyn_cast<llvm::MDString>( elem->getOperand( 1 ) );
        if( !stypeMD )
            continue;
        const llvm::StringRef mdName = stypeMD->getString();
        if( mdName == name )
            return elem;
    }
    return nullptr;
}

static OptixResult validateTargetArchitecture( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    constexpr int64_t minSmVersion = 30;
    int               maxSmVersion = module.compileParams.maxSmVersion;

    const llvm::MDNode* mdNode = getNamedLwvmMetdataNode( module.llvmModule, "targetArch" );

    OptixResultOneShot result;
    if( mdNode )
    {
        llvm::Value* v = UseMdAsValue( module.llvmModule->getContext(), mdNode->getOperand( 0 ) );
        int64_t      val;
        if( !corelib::getConstantValue( v, val ) )
        {
            errDetails.m_compilerFeedback << "Error: Module contains invalid metadata for target architecture. Value "
                                             "of \"targetArch\" is not constant int\n";
            result += OPTIX_ERROR_INTERNAL_COMPILER_ERROR;
        }
        else
        {
            if( val < minSmVersion )
            {
                errDetails.m_compilerFeedback << "Error: Invalid target architecture. Minimum required: sm_"
                                              << minSmVersion << ", found: sm_" << val << "\n";
                result += OPTIX_ERROR_ILWALID_PTX;
            }

            if( val > maxSmVersion )
            {
                errDetails.m_compilerFeedback
                    << "Error: Invalid target architecture. Maximum feasible for current context: sm_" << maxSmVersion
                    << ", found: sm_" << val << "\n";
                result += OPTIX_ERROR_ILWALID_PTX;
            }
        }
    }

    return result;
}

bool isPtxDebugEnabled( llvm::Module* llvmModule )
{
    if( !llvmModule )
        return false;
    const llvm::MDNode* mdNode = getNamedLwvmMetdataNode( llvmModule, "ptxDebug" );
    if( mdNode )
    {
        llvm::Value* v = UseMdAsValue( llvmModule->getContext(), mdNode->getOperand( 0 ) );
        int val = 0;
        if( !corelib::getConstantValue( v, val ) )
            return false;
        return val != 0;
    }
    return false;
}

static std::string generateUserString( llvm::Module* llvmModule, const llvm::User* user )
{
    std::string userString;
    if( llvm::isa<llvm::GlobalVariable>( user ) )
    {
        const llvm::GlobalVariable* gv = llvm::cast<llvm::GlobalVariable>( user );
        userString                     = "global variable " + gv->getName().str();
    }
    else if( llvm::isa<llvm::Instruction>( user ) )
    {
        const llvm::Instruction* inst = llvm::cast<llvm::Instruction>( user );
        userString                    = "function " + inst->getParent()->getParent()->getName().str() + " ("
                     + getSourceLocation( llvmModule, inst ) + ")";
    }
    else
    {
        for( auto it = user->user_begin(), e = user->user_end(); it != e; ++it )
        {
            if( !userString.empty() )
                userString += ", ";
            userString += generateUserString( llvmModule, *it );
        }
    }
    return userString;
}

// When PTX debug is enabled, the address of functions might be taken to supply additional
// debug information. In that case we look at all instruction in the module and look for indirect
// function calls.
static OptixResult validateIndirectFunctionCallsForDebug( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    for( llvm::Function& func : *module.llvmModule )
    {
        for( llvm::BasicBlock& block : func )
        {
            for( llvm::Instruction& instruction : block )
            {
                if( llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>( &instruction ) )
                {
                    if( call->getCalledFunction() == nullptr && !isCallableCall( module, call ) )
                    {
                        errDetails.m_compilerFeedback << "Error: Found an indirect function call in " << func.getName().str()
                                                      << ": " << getSourceLocation( &instruction ) << "\n";
                        result += OPTIX_ERROR_ILWALID_PTX;
                    }
                }
            }
        }
    }
    return result;
}

static OptixResult validateVirtualFunctionTables( llvm::Module* module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    llvm::Regex        vtableRegex( "_ZTV(N)?[0-9]+(.+)$" );
    for( auto it = module->global_begin(), e = module->global_end(); it != e; ++it )
    {
        llvm::GlobalVariable& gv     = *it;
        llvm::Type*           gvType = gv.getType();
        if( gvType->isPointerTy() && gvType->getPointerElementType()->isArrayTy() )
        {
            llvm::SmallVector<llvm::StringRef, 3> matches;
            if( vtableRegex.match( gv.getName(), &matches ) )
            {
                // If the mangled name of the array starts with _ZTV, it is a virtual function table.
                errDetails.m_compilerFeedback << "Virtual function calls are not supported in class \"" << matches[2].str()
                                              << "\" (vtable symbol name: \"" << gv.getName().str() << "\").\n";
                result += OPTIX_ERROR_ILWALID_PTX;
            }
        }
    }
    return result;
}

static OptixResult validateIndirectFunctionCalls( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    if( isPtxDebugEnabled( module.llvmModule ) )
        return validateIndirectFunctionCallsForDebug( module, errDetails );

    OptixResultOneShot result;
    result += validateVirtualFunctionTables(module.llvmModule, errDetails);

    for( llvm::Function& func : *module.llvmModule )
    {
        if( func.hasAddressTaken() )
        {
            std::string userString;
            for( llvm::Value::user_iterator UI = func.user_begin(), UE = func.user_end(); UI != UE; ++UI )
            {
                if( !userString.empty() )
                    userString += ", ";
                userString += generateUserString(module.llvmModule, *UI);
            }
            errDetails.m_compilerFeedback << "Error: Taking the address of functions is illegal because indirect "
                                             "function calls are illegal. Address of function "
                                          << func.getName().str() << " is taken and used in " << userString << "\n";
            result += OPTIX_ERROR_ILWALID_PTX;
        }
    }
    return result;
}

enum State
{
    UNPROCESSED,
    IN_PROCESS,
    PROCESSED
};

static OptixResult validateCallGraphCycles( llvm::CallGraphNode*   node,
                                            const llvm::CallGraph& callGraph,
                                            std::map<llvm::CallGraphNode*, State>& state,
                                            llvm::SmallVector<llvm::Function*, 4>& stack,
                                            ErrorDetails& errDetails )
{
    State& nodeState = state[node];
    if( nodeState == PROCESSED )
        return OPTIX_SUCCESS;

    if( nodeState == IN_PROCESS )
    {
        auto it = std::find( stack.rbegin(), stack.rend(), node->getFunction() );
        if( it == stack.rend() )
        {
            errDetails.m_compilerFeedback << "Internal error in validateCallGraphCycles()\n";
            return OPTIX_ERROR_INTERNAL_ERROR;
        }

        errDetails.m_compilerFeedback << "Error: Found call graph relwrsion ilwolving \""
                                      << std::string( ( *it )->getName() ) << "\"";
        while( it != stack.rbegin() )
            errDetails.m_compilerFeedback << ", \"" << std::string( ( *--it )->getName() ) << "\"";
        errDetails.m_compilerFeedback << "\n";
        return OPTIX_ERROR_ILWALID_PTX;
    }

    OptixResultOneShot result;

    nodeState = IN_PROCESS;
    stack.push_back( node->getFunction() );

    llvm::CallGraphNode* callsExternalNode = callGraph.getCallsExternalNode();
    for( auto it = node->begin(), it_end = node->end(); it != it_end; ++it )
    {
        result += validateCallGraphCycles( it->second, callGraph, state, stack, errDetails );
    }

    stack.pop_back();
    nodeState = PROCESSED;

    return result;
}

static OptixResult validateCallGraphCycles( llvm::Module* llvmModule, ErrorDetails& errDetails )
{
    llvm::CallGraph callGraph( *llvmModule );

    std::map<llvm::CallGraphNode*, State> state;
    for( auto it = callGraph.begin(), it_end = callGraph.end(); it != it_end; ++it )
        state[(it->second).get()] = UNPROCESSED;

    // Keep track of the call stack for better error messages.
    llvm::SmallVector<llvm::Function*, 4> stack;

    llvm::CallGraphNode* externalCallingNode = callGraph.getExternalCallingNode();
    OptixResultOneShot   result;
    for( auto it = externalCallingNode->begin(), it_end = externalCallingNode->end(); it != it_end; ++it )
    {
        result += validateCallGraphCycles( it->second, callGraph, state, stack, errDetails );
    }

    return result;
}

static OptixResult validateInputPtx( InitialCompilationUnit& module, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    result += validateTargetArchitecture( module, errDetails );
    result += validateSharedMemoryAccess( module.llvmModule, errDetails );
    result += validateIllegalSyncFunctions( module.llvmModule, errDetails );
    if( !module.compileParams.disableActiveMaskCheck )
        result += validateSyncFunctionsActiveMaskUsage( module, errDetails );
    if( !module.compileParams.allowIndirectFunctionCalls )
        result += validateIndirectFunctionCalls( module, errDetails );
    result += validateCallGraphCycles( module.llvmModule, errDetails );
    return result;
}

static inline bool endsWith( std::string const& str, std::string const& suffix )
{
    if( suffix.size() > str.size() )
        return false;
    return std::equal( suffix.rbegin(), suffix.rend(), str.rbegin() );
}

// Do not return optix headers as source, use the (approximate) location
// in user code instead.
static bool filenameIsBlacklisted( const std::string& filename )
{
    // For some reason the LLVM input we get from LWCC actually does not only include
    // the filename in diLocation.getFilename(); but a full path.
    // So we need to do a "ends with" check here.
    return endsWith( filename, "optix_7_device.h") || endsWith(filename, "optix_7_device_impl.h" );
}

static bool directoryIsBlacklisted( const std::string& directory )
{
    return directory == "OPTIX/generated/";
}

// Add a string to the module and return a i64 that contains a pointer to it.
// The CompilationUnit caches strings that have already been added to the module.
static llvm::Constant* addModuleString( CompilationUnit& module, const std::string& str )
{
    llvm::Constant*& result = module.m_stringsCache[str];
    if( result != nullptr )
    {
        return result;
    }

    llvm::Constant* constant   = llvm::ConstantDataArray::getString( module.llvmModule->getContext(), str, true );
    llvm::Type*     constantTy = constant->getType();
    llvm::Constant* gv = new llvm::GlobalVariable( *module.llvmModule, constantTy, true, llvm::GlobalValue::InternalLinkage,
                                                   constant, "location", nullptr, llvm::GlobalVariable::NotThreadLocal,
                                                   corelib::ADDRESS_SPACE_GLOBAL, false );
    result = llvm::ConstantExpr::getAddrSpaceCast( gv, constantTy->getPointerTo() );
    result = llvm::ConstantExpr::getPtrToInt( result, llvm::Type::getInt64Ty( module.llvmModule->getContext() ) );
    return result;
}

// Returns the source code location of the given instruction as an LLVM value that can be referenced
// in error messages (for runtime exceptions).
static llvm::Constant* getSourceLocatiolwalue( CompilationUnit& module, llvm::Instruction* instruction )
{
    return addModuleString( module, getSourceLocation( instruction ) );
}

static std::string getExactSourceLocationAsString( llvm::Module* module, const llvm::Instruction* instruction )
{
    const llvm::DebugLoc& debugLoc = instruction->getDebugLoc();

    llvm::DILocation* diLocation = debugLoc.get();
    if( !diLocation )
        return std::string();
    const std::string filename   = diLocation->getFilename();

    if( filename.empty() )
        return std::string();

    if( filenameIsBlacklisted( filename ) )
        return std::string();
    const std::string directory = diLocation->getDirectory();

    if( directoryIsBlacklisted( directory ) )
        return std::string();
    const unsigned int line   = diLocation->getLine();
    const unsigned int column = diLocation->getColumn();

    // Note: always use '/' as separator. LLVM returns paths using that separator and
    //       it looks odd to only have the last one being a '\' on windows
    return directory + '/' + filename + ":" + std::to_string( line ) + ":" + std::to_string( column );
}

// check whether any of the inlinedAt locations is valid or blacklisted
static std::string getInlinedSourceLocationAsString( llvm::Module* module, const llvm::Instruction* instruction )
{
    const llvm::DebugLoc& debugLoc = instruction->getDebugLoc();
    llvm::DILocation* lwrrentLoc = debugLoc;
    // we don't want to check debugLoc again, only starting with its inlinedAt info
    if( !lwrrentLoc || !lwrrentLoc->getInlinedAt() )
        return std::string();

    while( llvm::DILocation* inlinedAt = lwrrentLoc->getInlinedAt() )
    {
        if( !filenameIsBlacklisted( inlinedAt->getFilename() ) && !directoryIsBlacklisted( inlinedAt->getDirectory() ) )
        {
            const std::string filename = inlinedAt->getFilename();
            const std::string directory = inlinedAt->getDirectory();
            const unsigned int line = inlinedAt->getLine();
            const unsigned int column = inlinedAt->getColumn();

            // Note: always use '/' as separator. LLVM returns paths using that separator and
            //       it looks odd to only have the last one being a '\' on windows
            return directory + '/' + filename + ":" + std::to_string( line ) + ":" + std::to_string( column );
        }
        lwrrentLoc = inlinedAt;
    }
    return std::string();
}

static std::string getApproximateSourceLocationAsString( llvm::Module* module, const llvm::Instruction* instruction )
{
    std::string result = getExactSourceLocationAsString( module, instruction );
    if( !result.empty() )
        return result;

    // as the exact source line is most probably blacklisted, check for inlinedAt locations
    result = getInlinedSourceLocationAsString( module, instruction );
    if( !result.empty() )
        return result;

    // Obtain iterator to instruction after \p instruction. Initializing the iterator directly from
    // the instruction does not work. If \p instruction is the last instruction of the basic block,
    // the final increment operation after the loop causes a crash.
    llvm::BasicBlock::const_iterator next     = instruction->getParent()->begin();
    llvm::BasicBlock::const_iterator next_end = instruction->getParent()->end();
    while( &( *next ) != instruction )
        ++next;
    ++next;

    // Obtain iterator to instruction before \p instruction. Initializing the iterator directly from
    // the instruction does not work. If \p instruction is the first instruction of the basic block,
    // the final increment operation after the loop causes a crash.
    llvm::BasicBlock::const_reverse_iterator prev     = instruction->getParent()->rbegin();
    llvm::BasicBlock::const_reverse_iterator prev_end = instruction->getParent()->rend();
    while( &( *prev ) != instruction )
        ++prev;
    ++prev;

    // Walk in both directions until we find a source location or there are no more instructions in
    // the basic block.
    while( prev != prev_end || next != next_end )
    {
        if( prev != prev_end )
        {
            result = getExactSourceLocationAsString( module, &( *prev ) );
            if( !result.empty() )
                return result + " (approximately)";
            ++prev;
        }

        if( next != next_end )
        {
            result = getExactSourceLocationAsString( module, &( *next ) );
            if( !result.empty() )
                return result + " (approximately)";
            ++next;
        }
    }

    return std::string();
}


static std::string getSourceLocation( llvm::Module* module, const llvm::Instruction* instruction )
{
    std::string str = getApproximateSourceLocationAsString( module, instruction );
    if( str.empty() )
        str =
            "No source location available. The input PTX may not contain debug information (lwcc option: -lineinfo), "
            "OptixModuleCompileOptions::debugLevel set to OPTIX_COMPILE_DEBUG_LEVEL_NONE, or no useful information "
            "is present for the current block.";

    return str;
}

static std::string getSourceLocation( llvm::Instruction* instruction )
{
    return getSourceLocation( instruction->getParent()->getParent()->getParent(), instruction );
}


}  // end namespace optix_exp
