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

#include <Compile/AttributeUtil.h>
#include <Compile/FindAttributeSegments.h>
#include <Compile/UnnamedToGlobalPass.h>
#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/RTX/RTXCompile.h>
#include <ExelwtionStrategy/RTX/RTXExceptionInstrumenter.h>
#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>
#include <ExelwtionStrategy/RTX/RTXPlan.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>  // annotationForSemanticType
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/LineInfo.h>
#include <FrontEnd/PTX/LinkPTXFrontEndIntrinsics.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

#include <exp/context/ErrorHandling.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/compiler/ModuleCache.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/math/Bits.h>
#include <prodlib/system/Knobs.h>

#include <rtcore/interface/types.h>

#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>

// Temporary include until we've fully moved over to lwvm70
#include <lwvm/Support/APIUpgradeUtilities.h>

#include <cctype>
#include <sstream>

using namespace corelib;
using namespace prodlib;
using namespace optix;
using namespace llvm;

namespace {
// clang-format off
Knob<std::string> k_saveLLVM( RT_DSTRING( "rtx.saveLLVM" ), "", RT_DSTRING( "Save LLVM stages during compilation" ) );
Knob<bool>        k_useContinuationCallables( RT_DSTRING( "rtx.useContinuationCallables" ), false, RT_DSTRING( "Use Continuation Callables for all callable programs" ) );
Knob<std::string> k_limitActiveLaunchIndices(  RT_DSTRING( "launch.limitActiveIndices" ),          "",  RT_DSTRING( "When specified limit which launch indices are active. Syntax: [minX, maxX], [minY, maxY]" ) );
Knob<bool>        k_specializeDemandLoadConfig( RT_DSTRING( "rtx.specializeDemandLoadConfig" ), true, RT_DSTRING( "Specialize based on demand load configuration" ) );
PublicKnob<bool>  k_enablePTXFallback( RT_PUBLIC_DSTRING( "compile.enableBackendFallback" ), false, RT_PUBLIC_DSTRING( "Enable fallback to old compiler backend." ) );
// clang-format on


using Types  = Type* [];
using Values = Value* [];

}

#define ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED 0

// -----------------------------------------------------------------------------
// Support functions.
static void defineVideoCalls( Module* module );

// -----------------------------------------------------------------------------
RTXCompile::RTXCompile( const RTXCompile::Options&  options,
                        const AttributeDecoderList& attributeDecoders,
                        const ProgramManager*       programManager,
                        int                         launchCounterForDebugging )
    : m_attributeDecoders( attributeDecoders )
    , m_heavyWeightCallSites( options.heavyweightCallSiteNames )
    , m_stype( options.stype )
    , m_inheritedStype( options.inheritedStype )
    , m_params( options.compileParams )
    , m_programManager( programManager )
    , m_launchCounterForDebugging( launchCounterForDebugging )
    , m_numConlwrrentLaunchDevices( options.numConlwrrentLaunchDevices )
    , m_pagingMode( options.pagingMode )
    , m_useD2IR( !options.rtcoreCompileOptions.useLWPTX )
{
}

RTXCompile::~RTXCompile()
{
}

void RTXCompile::replacePagingMode( Module* module )
{
    LLVMContext& context = module->getContext();
    Type*        i32Ty   = Type::getInt32Ty( context );

    Function* funcToReplace = getFunctionOrAssert( module, "_ZN4cort24Global_getDemandLoadModeEPNS_14CanonicalStateE" );
    Value*    valueToInsert = ConstantInt::get( i32Ty, static_cast<unsigned int>( m_pagingMode ) );
    for( CallInst* call : getCallsToFunction( funcToReplace ) )
    {
        call->replaceAllUsesWith( valueToInsert );
        call->eraseFromParent();
    }
}

std::string RTXCompile::runOnFunction( Function* function, bool& fellBackToLWPTX )
{
    int dumpId = 0;

    Module*            module            = function->getParent();
    const std::string& functionName      = function->getName();
    bool               moduleHasLineInfo = hasLineInfo( module );

    // Add semantic type to dumped file names for consistency with rtcore and to disambiguate different
    // uses of the null program.
    std::string semanticType = optix::semanticTypeToString( m_stype );

    if( m_stype != m_inheritedStype )
    {
        RT_ASSERT( m_stype == ST_BOUND_CALLABLE_PROGRAM
                   || ( m_stype == ST_BINDLESS_CALLABLE_PROGRAM && m_inheritedStype == ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE ) );
        // Add the inherited semantic type to disambiguate different uses of the
        // same bound callable program
        semanticType = semanticType + "__" + semanticTypeToString( m_inheritedStype );
    }

    algorithm::transform( semanticType, semanticType.begin(), ::tolower );
    std::string dumpFunctionName = "_" + semanticType + "__" + functionName;

    dump( module, dumpFunctionName, dumpId++, "init" );

    // Process trace calls
    lowerTrace( module );
    dump( module, dumpFunctionName, dumpId++, "lower_trace" );

    // Process callable program calls.
    lowerCallableProgramCalls( module );
    dump( module, dumpFunctionName, dumpId++, "lower_callable_program_calls" );

    if( m_stype == ST_INTERNAL_AABB_ITERATOR )
        lowerAABBIteratorProgram( function );
    dump( module, dumpFunctionName, dumpId++, "lower_aabb_iterator_program" );

    // Lower attributes
    if( m_stype == ST_CLOSEST_HIT || m_stype == ST_ANY_HIT )
    {
        lowerAttributesForCHandAH( function );
        dump( module, dumpFunctionName, dumpId++, "lower_attributes_for_CH_and_AH" );

        lowerGetAttributeData( function );
        dump( module, dumpFunctionName, dumpId++, "lower_get_attribute_data" );
    }

    if( m_stype == ST_INTERSECTION )
    {
        lowerIsPotentialIntersection( function );
        lowerReportFullIntersection( function );
    }
    dump( module, dumpFunctionName, dumpId++, "lower_report_full_intersection" );


    if( m_stype == ST_EXCEPTION || m_stype == ST_INTERNAL_AABB_EXCEPTION )
        lowerExceptionDetails( module );
    dump( module, dumpFunctionName, dumpId++, "lower_exception_details" );

    lowerGetLwrrentRay( module );
    dump( module, dumpFunctionName, dumpId++, "lower_get_lwrrent_ray" );

    // Run an inline pass to inline the helper runtime functions calling rtx intrinsics
    legacy::PassManager PM;
    PM.add( createAlwaysInlinerLegacyPass() );
    PM.run( *module );
    dump( module, dumpFunctionName, dumpId++, "always_inliner" );

    // Rewrite bound callable programs to use bound callable program
    // state. Needs to happen before replacePlaceholderAccessors because
    // here we want to replace all attribute intrinsics.
    // Needs to happen after the always inliner pass for correct payload handling.
    function = lowerBoundCallableProgram( function );
    dump( module, dumpFunctionName, dumpId++, "lower_bound_callable_program" );

    // Load payloads
    lowerPayloadGetAndSet( module );
    dump( module, dumpFunctionName, dumpId++, "lower_payload_get_and_set" );

    printfToVprintf( module );

    // Move any variables that haven't been given an address space to the global
    // space, for D2IR compatibility.
    moveVariablesFromUnnamedToGlobalAddressSpace( module );

    // Apply index out of bounds check if the knob is set. Avoid adding this check to AABB programs so the launches
    // still function properly.
    if( !k_limitActiveLaunchIndices.isDefault() && m_stype == ST_RAYGEN )
        addLimitIndicesCheck( module, function );

    // If the current function is a raygen program, add a check to make sure it is running inside of the launch.
    if( m_stype == ST_RAYGEN || m_stype == ST_INTERNAL_AABB_ITERATOR )
        addLaunchBoundsCheck( module, function );
    dump( module, dumpFunctionName, dumpId++, "launch_bounds_check_added" );

    // Lower all the remaining buffer and variable lookups that have not been specialized.
    // FIXME: we are lwrrently reusing the same function that MKCompile uses.
    // The implementation of replacePlaceholderAccessors will likely change in the future
    // (e.g, the payload and the attributes are going to be handled differently by RTX)
    // nevertheless we should strive to avoid code duplication between MK and RTX.
    // The attribute map is not lwrrently used, attribute accesses have already been lowered at this point.
    AttributeOffsetMap attributeOffsets;
    SemanticType       stypeForLookup = m_inheritedStype;
    if( m_inheritedStype == ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE )
        stypeForLookup = ST_BINDLESS_CALLABLE_PROGRAM;
    replacePlaceholderAccessors( module, attributeOffsets, m_programManager, false, false, &m_stype, &stypeForLookup );
    dump( module, dumpFunctionName, dumpId++, "replace_placeholder_accessors" );

    rewriteTableAccessesToUseConstMemory( module, m_params.constMemAllocFlags );
    dump( module, dumpFunctionName, dumpId++, "rewrite_table_acesses_to_use_const_memory" );

    // Needs to happen after replacePlaceholderAccessors() because here we want to inline all calles
    // of _ZN4cort25getGeometryInstanceHandleEv().
    if( m_stype == ST_BOUNDING_BOX )
        function = lowerBoundingBoxProgram( function );
    dump( module, dumpFunctionName, dumpId++, "lower_bounding_box_program" );

    // Rewrite accesses to the sbt pointer to use the one passed to the bound callable
    // program. Needs to happen after replacePlaceholderAccessors.
    if( m_stype == ST_BOUND_CALLABLE_PROGRAM )
        rewriteBoundCallableProgramParentSbtPointer( function );
    dump( module, dumpFunctionName, dumpId++, "lower_bound_callable_program_sbt_pointer" );

    if( m_numConlwrrentLaunchDevices != DONT_SPECIALIZE_NUM_DEVICES )
        replaceGlobalDeviceCount( module, m_numConlwrrentLaunchDevices );

    if( k_specializeDemandLoadConfig.get() )
        replacePagingMode( module );

    // Annotate the entry function. Needs to happen after lowerBoundingBoxProgram() which replaces
    // the function in case of bounding boxes.
    optix::SemanticType stypeForAnnotation = m_stype;
    if( stypeForAnnotation == ST_BOUNDING_BOX )
        stypeForAnnotation = ST_BINDLESS_CALLABLE_PROGRAM;
    else if( stypeForAnnotation == ST_INTERNAL_AABB_ITERATOR )
        stypeForAnnotation = ST_RAYGEN;
    else if( stypeForAnnotation == ST_INTERNAL_AABB_EXCEPTION )
        stypeForAnnotation = ST_EXCEPTION;
    else if( RTXPlan::isDirectCalledBoundCallable( m_stype, m_inheritedStype ) )
        stypeForAnnotation = ST_BINDLESS_CALLABLE_PROGRAM;
    else if( stypeForAnnotation == ST_BINDLESS_CALLABLE_PROGRAM && m_stype != m_inheritedStype )
        stypeForAnnotation = ST_BOUND_CALLABLE_PROGRAM;  // make sure that bindless CPs with trace calls are compiled as CC

    const char* semanticTypeForAnnotation = optix::annotationForSemanticType( stypeForAnnotation );

    std::string newFunctionName;
    if( m_stype != m_inheritedStype )
    {
        if( m_stype == ST_BOUND_CALLABLE_PROGRAM )
        {
            // Add the inherited semantic type to avoid multiple definitions of bound callable
            // programs that are used from different callers.
            newFunctionName = semanticTypeToString( m_inheritedStype ) + "__" + function->getName().str();
            corelib::renameFunction( function, newFunctionName, /*changeDiLinkageNameOnly=*/false );
        }
        else
        {
            // No need to add anything to name of bindless callable programs
            // they will get the differentiation directcallable/continuationcallable
            // before the call to corelib::addLwvmRtAnnotationMetadata below. Only
            // those two types are possible.
            RT_ASSERT( m_stype == ST_BINDLESS_CALLABLE_PROGRAM );
        }
    }

    addReturnsForExceptionThrow( module, function );
    dump( module, dumpFunctionName, dumpId++, "add_returns_for_exception_throw" );

    newFunctionName = std::string( semanticTypeForAnnotation ) + "__" + function->getName().str();
    corelib::renameFunction( function, newFunctionName, /*changeDiLinkageNameOnly=*/false );
    corelib::addLwvmRtAnnotationMetadata( function, semanticTypeForAnnotation );

    // Set internal linkage where possible, explicitly remove noinline attribute at this point,
    // unless forceinline is disabled via a Context attribute.
    for( Function& F : *module )
    {
        if( F.isDeclaration() || &F == function )
            continue;
        if( !F.getName().startswith( "dbgPrint_" ) )
            F.setLinkage( GlobalVariable::InternalLinkage );

        F.addFnAttr( Attribute::NoUnwind );

        if( m_params.forceInlineUserFunctions )
        {
            F.removeFnAttr( Attribute::NoInline );
            F.addFnAttr( Attribute::AlwaysInline );
        }
        else if( F.hasFnAttribute( Attribute::NoInline ) )
        {
            // Whitelist cort functions for inlining since some of
            // them are explitly marked noinline to ensure they
            // remain available for later processing
            const std::string funcName( F.getName() );
            if( stringBeginsWith( funcName, "_ZN4cort" ) )
            {
                F.removeFnAttr( Attribute::NoInline );
                F.addFnAttr( Attribute::AlwaysInline );
            }
        }
        else
        {
            F.addFnAttr( Attribute::AlwaysInline );
        }
    }

    // We link PTX intrinsics after we've set internal linkage on all possible
    // functions, so we can eliminate dead runtime functions and avoid linking
    // intrinsics that are never used.
    linkPTXFrontEndIntrinsics( module, m_useD2IR, k_enablePTXFallback.get(), fellBackToLWPTX );

    dump( module, dumpFunctionName, dumpId++, "before_optimize" );

    // Run optimization stack.
    optimizeModule( module );

    dump( module, dumpFunctionName, dumpId++, "optimized" );

    // Strip the CanonicalState and additional arguments from the function signature. Needs to happen
    // after optimizeModule().
    changeFunctionSignature( function );
    function = nullptr;

    dump( module, dumpFunctionName, dumpId++, "changed_function_signature" );

    if( moduleHasLineInfo )
    {
        addMissingLineInfo( module, "optix_internal" );
        dump( module, dumpFunctionName, dumpId++, "missing_lineInfo_added" );
    }

    return newFunctionName;
}

void RTXCompile::addLimitIndicesCheck( Module* module, llvm::Function* function )
{
    BasicBlock::iterator   insertBefore = getSafeInsertionPoint( function );
    corelib::CoreIRBuilder irb{&*insertBefore};

    Value* statePtr = function->arg_begin();

    Function*       verticalCheckFunc = getFunctionOrAssert( module, "RTX_indicesOutsideOfLimitedRange" );
    Value*          shouldSkip        = irb.CreateCall( verticalCheckFunc, statePtr );
    TerminatorInst* thenTerm          = SplitBlockAndInsertIfThen( cast<Instruction>( shouldSkip ), true );

    irb.SetInsertPoint( thenTerm );
    irb.CreateRetVoid();
    thenTerm->eraseFromParent();
}

void RTXCompile::addLaunchBoundsCheck( Module* module, llvm::Function* function ) const
{
    const BasicBlock::iterator insertBefore = getSafeInsertionPoint( function );
    corelib::CoreIRBuilder     irb{&*insertBefore};

    Value*    statePtr         = function->arg_begin();
    Function* boundsCheckFunc  = getFunctionOrAssert( module, "RTX_indexIsOutsideOfLaunch" );
    statePtr                   = irb.CreateBitCast( statePtr, *boundsCheckFunc->getFunctionType()->param_begin() );
    Value*          shouldSkip = irb.CreateCall( boundsCheckFunc, statePtr );
    TerminatorInst* thenTerm   = SplitBlockAndInsertIfThen( cast<Instruction>( shouldSkip ), true );

    irb.SetInsertPoint( thenTerm );
    irb.CreateRetVoid();
    thenTerm->eraseFromParent();
}

void RTXCompile::linkPTXFrontEndIntrinsics( Module* module, bool useD2IR, bool enableLWPTXFallback, bool& fellBackToLWPTX )
{
    optix_exp::ErrorDetails errDetails;
    OptixResult result =
            optix_exp::linkPTXFrontEndIntrinsics( module, useD2IR, enableLWPTXFallback, fellBackToLWPTX, errDetails );

    // Log any compiler feedback we received while linking the intrinsics.
    if( !errDetails.m_compilerFeedback.str().empty() )
    {
        lwarn << errDetails.m_compilerFeedback.str();
    }

    if( result != OPTIX_SUCCESS )
        throw CompileError( RT_EXCEPTION_INFO, errDetails.m_description );
}

void RTXCompile::optimizeModule( Module* module ) const
{
    // Verify before the pass manager - otherwise we get an error that can't be caught.
    RT_ASSERT( !verifyModule( *module, &llvm::errs() ) );

    // Run a limited set of optimization passes on each callgraph state function
    legacy::PassManager PM;

    PM.add( createPromoteMemoryToRegisterPass() );
    PM.add( createInstSimplifyLegacyPass() );
    PM.add( createAlwaysInlinerLegacyPass() );
    PM.add( createCFGSimplificationPass() );
    PM.add( createLoopSimplifyPass() );
    // This is useful to remove the temporary alloca needed to create the struct
    // to pass to report.intersection.
    PM.add( createSROAPass() );
    PM.add( createGlobalDCEPass() );
    PM.add( createInstructionCombiningPass() );
    PM.run( *module );
}

void RTXCompile::changeFunctionSignature( Function* entry ) const
{
    LLVMContext& llvmContext = entry->getParent()->getContext();
    Type*        voidTy      = Type::getVoidTy( llvmContext );
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );

    // Remove unused arguments -- partilwlarly CanonicalState pointers
    {
        Module*     module = entry->getParent();
        legacy::PassManager PM;
        PM.add( createDeadArgEliminationPass() );
        PM.run( *module );
    }

    // Ensure that there are no remaining uses of the state pointer
    Value* statePtr = entry->arg_begin();
    if( !statePtr->use_empty() )
    {
        // SGP: remove debugging
        llvm::errs() << "Uses:\n";
        for( auto iter = statePtr->user_begin(); iter != statePtr->user_end(); ++iter )
            llvm::errs() << **iter << '\n';
        RT_ASSERT_FAIL_MSG( "CanonicalState still used in function: " + entry->getName().str() );
    }

    const bool isCP = m_stype == ST_BOUND_CALLABLE_PROGRAM || m_stype == ST_BINDLESS_CALLABLE_PROGRAM;
    const bool isBB = m_stype == ST_BOUNDING_BOX;

    FunctionType* funcTy = entry->getFunctionType();

    const int  returlwalueNumRegs       = corelib::getNumRequiredRegisters( funcTy->getReturnType() );
    const int  newReturlwalueNumRegs    = std::min( returlwalueNumRegs, m_params.numCallableParamRegisters );
    const bool needsReturlwalueSpilling = returlwalueNumRegs > m_params.numCallableParamRegisters;

    int parametersNumRegs = 0;
    for( unsigned int i = 1, e = funcTy->getNumParams(); i < e; ++i )
        parametersNumRegs += corelib::getNumRequiredRegisters( funcTy->getParamType( i ) );
    const int  adjustedParametersNumRegs = parametersNumRegs + ( needsReturlwalueSpilling ? 2 : 0 );
    const int  newParametersNumRegs      = std::min( adjustedParametersNumRegs, m_params.numCallableParamRegisters );
    const bool needsParameterSpilling    = adjustedParametersNumRegs > m_params.numCallableParamRegisters;

    // Create the new type, stripping the CanonicalState.
    // Unless this is a callable or bounding box program, also strip all other arguments.
    // For callable or bounding box programs, split up all arguments into 32-bit values and
    // create a single, aggregate argument from them.
    FunctionType* newFuncTy = nullptr;
    if( isCP || isBB )
    {
        Type* returnTy    = funcTy->getReturnType();
        Type* newReturnTy = returnTy->isVoidTy() ? returnTy : ArrayType::get( i32Ty, newReturlwalueNumRegs );

        if( newParametersNumRegs == 0 )
        {
            newFuncTy = FunctionType::get( newReturnTy, false );
        }
        else
        {
            Type* newParamTy = ArrayType::get( i32Ty, newParametersNumRegs );
            newFuncTy        = FunctionType::get( newReturnTy, newParamTy, false );
        }
    }
    else
    {
        // Do some sanity checks.
        for( Function::arg_iterator A = entry->arg_begin(), AE = entry->arg_end(); A != AE; ++A )
        {
            RT_ASSERT_MSG( A->use_empty(),
                           "argument of function " + entry->getName().str() + " still has a use: " + A->getName().str() );
        }
        for( const BasicBlock& BB : *entry )
        {
            if( const ReturnInst* ret = dyn_cast<ReturnInst>( BB.getTerminator() ) )
            {
                RT_ASSERT_MSG( !ret->getReturlwalue(),
                               "function " + entry->getName().str() + " must not return a value" );
            }
        }
        newFuncTy = FunctionType::get( voidTy, false );
    }

    // Create the new function
    Function* newFunc = Function::Create( newFuncTy, entry->getLinkage(), "", entry->getParent() );
    newFunc->takeName( entry );

    // Copy function attributes but no others
    // TODO: For callable (and bounding box?) programs, also copy the parameter attributes.
    newFunc->setSubprogram( entry->getSubprogram() );
    newFunc->setAttributes( newFunc->getAttributes().addAttributes( newFunc->getContext(), AttributeList::FunctionIndex,
                                                                    entry->getAttributes().getFnAttributes() ) );
    if( needsParameterSpilling || needsReturlwalueSpilling )
        newFunc->removeFnAttr( Attribute::ReadNone );
    if( needsReturlwalueSpilling )
        newFunc->removeFnAttr( Attribute::ReadOnly );

    // Splice the body.
    newFunc->getBasicBlockList().splice( newFunc->begin(), entry->getBasicBlockList() );
    RT_ASSERT( entry->getBasicBlockList().empty() );

    // For callable and bounding box programs, rewire the arguments since now the CanonicalState
    // is gone. Also, we need to create values of the appropriate type from the
    // aggregate of 32-bit values again.
    if( ( isCP || isBB ) && funcTy->getNumParams() > 0 )
    {
        Function::arg_iterator aggregateArg = newFunc->arg_begin();
        Function::arg_iterator origA        = entry->arg_begin();
        ++origA;  // Skip CanonicalState.

        // If the function has spilled parameters we use a local buffer on the stack to copy parameters
        // in registers and spilled parameters into one aggregate.
        Value* newArgs = aggregateArg;
        if( needsParameterSpilling )
        {
            BasicBlock::iterator   entryInsert  = getSafeInsertionPoint( newFunc );
            Instruction*           insertBefore = &*entryInsert;
            corelib::CoreIRBuilder irb{insertBefore};

            // Create aggregate value for all parameters.
            Type*  aggregateTy    = ArrayType::get( i32Ty, parametersNumRegs );
            Value* aggregateValue = UndefValue::get( aggregateTy );

            // Copy all parameters in registers into aggregate value.
            for( int paramIdx = 0; paramIdx < m_params.numCallableParamRegisters - 2; ++paramIdx )
            {
                Value* param   = irb.CreateExtractValue( aggregateArg, paramIdx, "param" );
                aggregateValue = irb.CreateInsertValue( aggregateValue, param, paramIdx, "aggregateValue" );
            }

            // Compute address of spilled parameters.
            Value* spillBaseAddrLo =
                irb.CreateExtractValue( aggregateArg, newParametersNumRegs - 2, "spillBaseAddrLo" );
            Value* spillBaseAddrHi =
                irb.CreateExtractValue( aggregateArg, newParametersNumRegs - 1, "spillBaseAddrHi" );
            Value* spillBaseAddr = createCast_2xi32_to_i64( spillBaseAddrLo, spillBaseAddrHi, insertBefore );
            spillBaseAddr        = irb.CreateIntToPtr( spillBaseAddr, i32Ty->getPointerTo(), "spillBaseAddr" );

            // Append spilled parameters to aggregate value.
            for( int paramIdx = m_params.numCallableParamRegisters - 2; paramIdx < parametersNumRegs; ++paramIdx )
            {
                Value* spillIndices = irb.getInt32( paramIdx - ( m_params.numCallableParamRegisters - 2 ) );
                Value* spillAddr    = irb.CreateGEP( spillBaseAddr, spillIndices, "spillAddr" );
                Value* spilledParam = irb.CreateLoad( spillAddr, "spilledParam" );
                aggregateValue      = irb.CreateInsertValue( aggregateValue, spilledParam, paramIdx, "aggregateValue" );
            }

            newArgs = aggregateValue;
        }

        // Replace uses of the original arguments (origA) by the values in newArgs
        int index = 0;
        for( auto origAE = entry->arg_end(); origA != origAE; ++origA )
        {
            // Make sure we only generate a single extract if an instruction uses the
            // same argument multiple times by collecting all uses upfront. This also
            // prevents messing up the iteration order by replacing uses.
            SetVector<Instruction*> uses;
            for( Argument::user_iterator U = origA->user_begin(), UE = origA->user_end(); U != UE; ++U )
            {
                RT_ASSERT( isa<Instruction>( *U ) );
                Instruction* useI = cast<Instruction>( *U );
                RT_ASSERT( useI->getParent()->getParent() == newFunc );
                uses.insert( useI );
            }

            for( Instruction* useI : uses )
            {
                if( PHINode* PN = dyn_cast<PHINode>( useI ) )
                {
                    unsigned int numIncoming = PN->getNumIncomingValues();
                    for( unsigned int i = 0; i < numIncoming; ++i )
                    {
                        Value* incoming = PN->getIncomingValue( i );
                        if( incoming == origA )
                        {
                            BasicBlock*  block        = PN->getIncomingBlock( i );
                            Instruction* insertBefore = &block->back();
                            // Unpack and combine as many of the 32-bit arguments as necessary to
                            // match the original type again.
                            Value* newArg = corelib::unflattenAggregateForRTCore( origA->getType(), newArgs, index, insertBefore );
                            useI->replaceUsesOfWith( origA, newArg );
                        }
                    }
                }
                else
                {
                    // Unpack and combine as many of the 32-bit arguments as necessary to
                    // match the original type again.
                    Value* newArg = corelib::unflattenAggregateForRTCore( origA->getType(), newArgs, index, useI );
                    useI->replaceUsesOfWith( origA, newArg );
                }
            }

            index += corelib::getNumRequiredRegisters( origA->getType() );
        }
    }

    // For callable programs, the return type, if non-void, is now an array of
    // i32, so we have to translate the old return value.
    if( isCP && !funcTy->getReturnType()->isVoidTy() )
    {
        for( BasicBlock& BB : *newFunc )
        {
            ReturnInst* ret = dyn_cast<ReturnInst>( BB.getTerminator() );
            if( !ret )
                continue;
            Value* retVal = ret->getReturlwalue();
            RT_ASSERT( retVal );
            std::vector<Value*> retVals = corelib::flattenAggregateTo32BitValuesForRTCore( retVal, ret );

            // Insert up to m_params.numCallableParamRegisters-2 flattened return values into return value array.
            Value*    retArray = UndefValue::get( newFuncTy->getReturnType() );
            const int e = needsReturlwalueSpilling ? m_params.numCallableParamRegisters - 2 : (int)retVals.size();
            corelib::CoreIRBuilder irb{ret};
            for( int retValIdx = 0; retValIdx < e; ++retValIdx )
            {
                Value* retValElement = retVals[retValIdx];
                if( retValElement->getType() != i32Ty )
                    retValElement = irb.CreateBitCast( retValElement, i32Ty );
                retArray          = irb.CreateInsertValue( retArray, retValElement, retValIdx );
            }

            if( needsReturlwalueSpilling )
            {
                // Append address of spilled return values to return value array.
                Function::arg_iterator aggregateArg = newFunc->arg_begin();
                Value*                 spillBaseAddrLo =
                    irb.CreateExtractValue( aggregateArg, newParametersNumRegs - 2, "spillBaseAddrLo" );
                Value* spillBaseAddrHi =
                    irb.CreateExtractValue( aggregateArg, newParametersNumRegs - 1, "spillBaseAddrHi" );
                retArray = irb.CreateInsertValue( retArray, spillBaseAddrLo, m_params.numCallableParamRegisters - 2,
                                                  "spillBaseAddrLo" );
                retArray = irb.CreateInsertValue( retArray, spillBaseAddrHi, m_params.numCallableParamRegisters - 1,
                                                  "spillBaseAddrHi" );

                // Compute address of spilled return values.
                Value* spillBaseAddr = createCast_2xi32_to_i64( spillBaseAddrLo, spillBaseAddrHi, ret );
                spillBaseAddr        = irb.CreateIntToPtr( spillBaseAddr, i32Ty->getPointerTo(), "spillBaseAddr" );

                // Store surplus return values in local memory.
                for( int retValIdx = m_params.numCallableParamRegisters - 2; retValIdx < (int)retVals.size(); ++retValIdx )
                {
                    Value* indices = ConstantInt::get( i32Ty, retValIdx - ( m_params.numCallableParamRegisters - 2 ) );
                    Value* spillAddr     = irb.CreateGEP( spillBaseAddr, indices, "spillAddr" );
                    Value* retValElement = retVals[retValIdx];
                    if( retValElement->getType() != i32Ty )
                        retValElement = irb.CreateBitCast( retValElement, i32Ty );
                    irb.CreateStore( retValElement, spillAddr );
                }
            }

            irb.CreateRet( retArray );
            ret->eraseFromParent();
        }
    }

    // Bounding box programs do not need to handle return value spilling.
    RT_ASSERT( !isBB || funcTy->getReturnType()->isVoidTy() );

    // Move any LWVM metadata from the old to the new function.
    if( NamedMDNode* lwvmMd = entry->getParent()->getNamedMetadata( "lwvm.annotations" ) )
        corelib::replaceMetadataUses( lwvmMd, entry, newFunc );

    // Move any LWVM-RT metadata from the old to the new function.
    if( NamedMDNode* lwvmRtMd = entry->getParent()->getNamedMetadata( "lwvm.rt.annotations" ) )
        corelib::replaceMetadataUses( lwvmRtMd, entry, newFunc );

    // Erase the original function.
    RT_ASSERT( entry->use_empty() );
    entry->eraseFromParent();
}

// -----------------------------------------------------------------------------
llvm::Function* RTXCompile::lowerBoundingBoxProgram( llvm::Function* function ) const
{
    RT_ASSERT( m_stype == ST_BOUNDING_BOX );

    Module*      module      = function->getParent();
    LLVMContext& llvmContext = module->getContext();
    Type*        voidTy      = Type::getVoidTy( llvmContext );
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );
    Type*        i64Ty       = Type::getInt64Ty( llvmContext );
    Type*        floatTy     = Type::getFloatTy( llvmContext );

    FunctionType* funcTy = function->getFunctionType();

    // Extract type of first argument (canonical state).
    RT_ASSERT( funcTy->getReturnType() == voidTy );
    Function::arg_iterator A                = function->arg_begin();
    Type*                  canonicalStateTy = A->getType();
    RT_ASSERT( canonicalStateTy->getTypeID() == Type::PointerTyID );
    ++A;
    RT_ASSERT( A == function->arg_end() );

    // Create type of new function.
    Type*         newParamTy[] = {canonicalStateTy, i32Ty, i32Ty, i32Ty, i64Ty};
    FunctionType* newFuncTy    = FunctionType::get( voidTy, newParamTy, false );

    // Create the new function.
    Function* newFunc = Function::Create( newFuncTy, function->getLinkage(), "", function->getParent() );
    newFunc->takeName( function );

    // Copy function attributes but no others
    newFunc->setAttributes( newFunc->getAttributes().addAttributes( newFunc->getContext(), AttributeList::FunctionIndex,
                                                                    function->getAttributes().getFnAttributes() ) );

    // Splice the body.
    newFunc->getBasicBlockList().splice( newFunc->begin(), function->getBasicBlockList() );
    RT_ASSERT( function->getBasicBlockList().empty() );

    // Replace uses of the canonical state.
    Function::arg_iterator  CS    = function->arg_begin();
    Function::arg_iterator  newCS = newFunc->arg_begin();
    SetVector<Instruction*> uses;
    for( Argument::user_iterator U = CS->user_begin(), UE = CS->user_end(); U != UE; ++U )
    {
        RT_ASSERT( isa<Instruction>( *U ) );
        Instruction* useI = cast<Instruction>( *U );
        RT_ASSERT( useI->getParent()->getParent() == newFunc );
        uses.insert( useI );
    }
    for( Instruction* useI : uses )
        useI->replaceUsesOfWith( CS, newCS );

    // Move any LWVM metadata from the old to the new function (there's no LWVM-RT metadata yet).
    if( NamedMDNode* lwvmMd = function->getParent()->getNamedMetadata( "lwvm.annotations" ) )
        corelib::replaceMetadataUses( lwvmMd, function, newFunc );

    // Erase the original function.
    RT_ASSERT( function->use_empty() );
    function->eraseFromParent();
    function = nullptr;

    Function::arg_iterator newArgs = newFunc->arg_begin();
    newArgs++;  // canonical state
    Value* gi             = newArgs++;
    Value* primitiveIndex = newArgs++;
    Value* motionIndex    = newArgs++;
    Value* aabb           = newArgs++;

    // Inline all callers of _ZN4cort25getGeometryInstanceHandleEv().
    // TODO The second argument does not seem to have any effect (why?). Not really important here, just to avoid
    // misuse of the callers later.
    Function* callee = module->getFunction( "_ZN4cort25getGeometryInstanceHandleEv" );
    if( !inlineAllCallersOfFunction( callee, /*eraseUnusedFunctions*/ true ) )
        throw CompileError( RT_EXCEPTION_INFO, "Unable to inline callers of: " + callee->getName().str() );

    // Replace uses of calls to _ZN4cort25getGeometryInstanceHandleEv() by gi argument and remove calls.
    std::vector<CallInst*> CIs = getCallsToFunction( callee, newFunc );
    for( CallInst* CI : CIs )
    {
        CI->replaceAllUsesWith( gi );
        CI->eraseFromParent();
    }

    // Replace uses of calls to optixi_getPrimitiveArgToComputeAABB() by primitiveIndex argument and remove calls.
    callee = module->getFunction( "optixi_getPrimitiveArgToComputeAABB" );
    CIs    = getCallsToFunction( callee, newFunc );
    for( CallInst* CI : CIs )
    {
        CI->replaceAllUsesWith( primitiveIndex );
        CI->eraseFromParent();
    }

    // Replace uses of calls to optixi_getMotionIndexArgToComputeAABB() by motionIndex argument and remove calls.
    callee = module->getFunction( "optixi_getMotionIndexArgToComputeAABB" );
    CIs    = getCallsToFunction( callee, newFunc );
    for( CallInst* CI : CIs )
    {
        CI->replaceAllUsesWith( motionIndex );
        CI->eraseFromParent();
    }

    // Insert i64 to float* cast for aabb argument.
    BasicBlock::iterator   functionInsert = getSafeInsertionPoint( newFunc );
    corelib::CoreIRBuilder irb{&*functionInsert};
    Value*                 aabbPtr = irb.CreateIntToPtr( aabb, floatTy->getPointerTo(), "aabbPtr" );

    // Replace uses of calls to optixi_getAABBArgToComputeAABB() by casted aabb argument and remove calls.
    callee = module->getFunction( "optixi_getAABBArgToComputeAABB" );
    CIs    = getCallsToFunction( callee, newFunc );
    for( CallInst* CI : CIs )
    {
        CI->replaceAllUsesWith( aabbPtr );
        CI->eraseFromParent();
    }

    return newFunc;
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerAABBIteratorProgram( llvm::Function* function ) const
{
    RT_ASSERT( m_stype == ST_INTERNAL_AABB_ITERATOR );

    Module*      module      = function->getParent();
    LLVMContext& llvmContext = module->getContext();
    Type*        i64Ty       = Type::getInt64Ty( llvmContext );

    replaceFunctionWithFunction( module, "optixi_computeGeometryInstanceAABB", "RTX_computeGeometryInstanceAABB" );
    replaceFunctionWithFunction( module, "optixi_computeGroupChildAABB", "RTX_computeGroupChildAABB" );
    if( module->getFunction( "optixi_gatherMotionAABBs" ) != nullptr )
        replaceFunctionWithFunction( module, "optixi_gatherMotionAABBs", "RTX_gatherMotionAABBs" );

    // While working on the AABB iterator program, change at the same time the runtime function RTX_computeAABB() to
    // redirect the call to RTX_computeAABB_BoundingBoxProgramStub() to a bindless callable program call of the
    // real bounding box program. The program ID / SBT index is already passed as 2nd argument to the stub.

    Function*              caller = module->getFunction( "RTX_computeGeometryInstanceAABB" );
    Function*              callee = module->getFunction( "RTX_computeGeometryInstanceAABB_BoundingBoxProgramStub" );
    std::vector<CallInst*> CIs    = getCallsToFunction( callee, caller );
    RT_ASSERT( !CIs.empty() );

    for( CallInst* CI : CIs )
    {
        const int aabbParamIndex = 5;
        // Change the type of the aabb argument from float* to i64.
        Value* aabb_ptr = CI->getArgOperand( aabbParamIndex );
        Value* aabb_i64 = corelib::CoreIRBuilder{CI}.CreatePtrToInt( aabb_ptr, i64Ty, "aabb_i64" );
        CI->setArgOperand( aabbParamIndex, aabb_i64 );
    }

    // Create the appropriate LWVM RT intrinsic that RTCore will understand.
    // %outputType @lw.rt.call.direct.T( i32 %callableSbtRecordIndex, %inputType %inputs ) nounwind
    const std::string rtcoreName =
        k_useContinuationCallables.get() ? "lw.rt.call.continuation.aabb" : "lw.rt.call.direct.aabb";

    lowerCalls( module, *callee, rtcoreName, false );
}

// -----------------------------------------------------------------------------
void RTXCompile::packPayload( bool              localPayload,
                              unsigned int      payloadSize,
                              Value*            payloadArg,
                              Function*         lwvmReadPayload,
                              unsigned int&     numPayloadRegisters,
                              unsigned int&     numPayloadPlusSizeRegisters,
                              llvm::Value*&     payloadValue,
                              llvm::Type*&      payloadTy,
                              llvm::BasicBlock* insertBefore ) const
{
    RT_ASSERT( !localPayload || payloadArg );
    RT_ASSERT( localPayload || lwvmReadPayload );

    corelib::CoreIRBuilder irBuilder( insertBefore );

    if( localPayload )
    {
        if( m_params.payloadInRegisters )
        {
            // If enabled, the payload size is in the next register after a payload of maximum size (over all ilwolved programs).
            numPayloadRegisters = prodlib::align( payloadSize, 4 ) / 4;
            RT_ASSERT_MSG( payloadSize % numPayloadRegisters == 0, "Fix odd-sized payload" );
            numPayloadPlusSizeRegisters = m_params.propagatePayloadSize ? ( m_params.maxPayloadSize / 4 + 1 ) : numPayloadRegisters;
            RT_ASSERT_MSG( m_params.maxPayloadSize % 4 == 0, "Fix odd-sized maximum payload" );
            payloadTy    = VectorType::get( irBuilder.getInt32Ty(), numPayloadPlusSizeRegisters );
            payloadValue = UndefValue::get( payloadTy );

            Value* payload32 = irBuilder.CreatePointerCast( payloadArg, irBuilder.getInt32Ty()->getPointerTo() );
            for( unsigned int i = 0; i < numPayloadRegisters; ++i )
            {
                Value* gep   = irBuilder.CreateConstInBoundsGEP1_32( irBuilder.getInt32Ty(), payload32, i );
                Value* load  = irBuilder.CreateLoad( gep );
                payloadValue = irBuilder.CreateInsertElement( payloadValue, load, irBuilder.getInt32( i ), "payload" );
            }

            if( m_params.propagatePayloadSize )
            {
                unsigned int i    = m_params.maxPayloadSize / 4;
                Value*       size = irBuilder.getInt32( payloadSize );
                payloadValue = irBuilder.CreateInsertElement( payloadValue, size, irBuilder.getInt32( i ), "payload" );
            }
        }
        else
        {
            // If enabled, the payload size is in the next register after the payload pointer, i.e., in payload register 2.
            numPayloadRegisters         = 2;
            numPayloadPlusSizeRegisters = numPayloadRegisters + ( m_params.propagatePayloadSize ? 1 : 0 );
            payloadTy                   = VectorType::get( irBuilder.getInt32Ty(), numPayloadPlusSizeRegisters );
            payloadValue                = UndefValue::get( payloadTy );

            if( !m_params.propagatePayloadSize )
            {
                payloadValue = irBuilder.CreatePtrToInt( payloadArg, irBuilder.getInt64Ty() );
                payloadValue = irBuilder.CreateBitCast( payloadValue, payloadTy );
            }
            else
            {
                Value* payloadArg_i64 = irBuilder.CreatePtrToInt( payloadArg, irBuilder.getInt64Ty() );
                // Do not use the corelib utility function to split the i64 here. In this
                // context it creates a crash deep inside LLVM.
                Value* payloadArg_2i32 = irBuilder.CreateBitCast( payloadArg_i64, VectorType::get( irBuilder.getInt32Ty(), 2 ) );
                Value* payloadPtrLo =
                    irBuilder.CreateExtractElement( payloadArg_2i32, irBuilder.getInt32( 0 ), "payloadPtrLo" );
                Value* payloadPtrHi =
                    irBuilder.CreateExtractElement( payloadArg_2i32, irBuilder.getInt32( 1 ), "payloadPtrHi" );
                Value* size = irBuilder.getInt32( payloadSize );
                payloadValue =
                    irBuilder.CreateInsertElement( payloadValue, payloadPtrLo, irBuilder.getInt32( 0 ), "payload" );
                payloadValue =
                    irBuilder.CreateInsertElement( payloadValue, payloadPtrHi, irBuilder.getInt32( 1 ), "payload" );
                payloadValue = irBuilder.CreateInsertElement( payloadValue, size, irBuilder.getInt32( 2 ), "payload" );
            }
        }
    }
    else
    {
        // We have to forward the current payload to lw.rt.trace.
        if( m_params.payloadInRegisters )
        {
            // If the payload has been promoted to registers, we read the payload through the lw.rt API and we create a local copy.
            // If enabled, the payload size is in the next register after a payload of maximum size (over all ilwolved programs).
            numPayloadRegisters = prodlib::align( payloadSize, 4 ) / 4;
            RT_ASSERT_MSG( payloadSize % numPayloadRegisters == 0, "Fix odd-sized payload" );
            numPayloadPlusSizeRegisters = m_params.propagatePayloadSize ? ( m_params.maxPayloadSize / 4 + 1 ) : numPayloadRegisters;
            RT_ASSERT_MSG( m_params.maxPayloadSize % 4 == 0, "Fix odd-sized maximum payload" );
            payloadTy    = VectorType::get( irBuilder.getInt32Ty(), numPayloadPlusSizeRegisters );
            payloadValue = UndefValue::get( payloadTy );

            for( unsigned int i = 0; i < numPayloadRegisters; ++i )
            {
                Value* payloadRegister = irBuilder.CreateCall( lwvmReadPayload, irBuilder.getInt32( i ) );
                payloadValue =
                    irBuilder.CreateInsertElement( payloadValue, payloadRegister, irBuilder.getInt32( i ), "payload" );
            }

            if( m_params.propagatePayloadSize )
            {
                unsigned int i = m_params.maxPayloadSize / 4;
                Value*       payloadRegister =
                    irBuilder.CreateCall( lwvmReadPayload, irBuilder.getInt32( i ), "payloadSize" );
                payloadValue =
                    irBuilder.CreateInsertElement( payloadValue, payloadRegister, irBuilder.getInt32( i ), "payload" );
            }
        }
        else
        {
            // If the payload has not been promoted to registers, we just forward the payload pointer to lw.rt.trace.

            // Insert proxy function that marks that the value we want here is actually the content of the first two
            // registers and not what is loaded from the memory where those two values point to. We cannot insert the lwvmrt intrinsic directly
            // here because that will not work for the bound callable program state.
            FunctionType* readPayloadType = FunctionType::get( irBuilder.getInt32Ty(), irBuilder.getInt32Ty(), false );
            Function*     readPayloadProxy =
                insertOrCreateFunction( lwvmReadPayload->getParent(), "proxy.lw.rt.read.payload.i32", readPayloadType );

            // If enabled, the payload size is in the next register after the payload pointer, i.e., in payload register 2.
            numPayloadRegisters         = 2;
            numPayloadPlusSizeRegisters = numPayloadRegisters + ( m_params.propagatePayloadSize ? 1 : 0 );

            payloadTy    = VectorType::get( irBuilder.getInt32Ty(), numPayloadPlusSizeRegisters );
            payloadValue = UndefValue::get( payloadTy );

            Value* payloadPtrLo = irBuilder.CreateCall( readPayloadProxy, irBuilder.getInt32( 0 ), "payloadPtrLo" );
            Value* payloadPtrHi = irBuilder.CreateCall( readPayloadProxy, irBuilder.getInt32( 1 ), "payloadPtrHi" );

            payloadValue = irBuilder.CreateInsertElement( payloadValue, payloadPtrLo, irBuilder.getInt32( 0 ) );
            payloadValue = irBuilder.CreateInsertElement( payloadValue, payloadPtrHi, irBuilder.getInt32( 1 ) );

            if( m_params.propagatePayloadSize )
            {
                Value* call  = irBuilder.CreateCall( readPayloadProxy, irBuilder.getInt32( 2 ), "payloadSize" );
                payloadValue = irBuilder.CreateInsertElement( payloadValue, call, irBuilder.getInt32( 2 ) );
            }
        }
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::unpackPayload( bool              localPayload,
                                unsigned int      numPayloadRegisters,
                                Value*            result,
                                Value*            payloadArg,
                                Function*         lwvmWritePayload,
                                llvm::BasicBlock* insertBefore ) const
{
    RT_ASSERT( !localPayload || payloadArg );
    RT_ASSERT( localPayload || lwvmWritePayload );

    corelib::CoreIRBuilder irBuilder( insertBefore );

    // Nothing to do if the payload has not been promoted to registers.
    if( !m_params.payloadInRegisters )
        return;

    if( localPayload )
    {
        // Write back the resulting payload into the memory of the payload variable.
        Value* payload32 = irBuilder.CreatePointerCast( payloadArg, irBuilder.getInt32Ty()->getPointerTo() );
        for( unsigned int i = 0; i < numPayloadRegisters; ++i )
        {
            Value* payloadElt = irBuilder.CreateExtractElement( result, irBuilder.getInt32( i ), "payload" );
            Value* gep        = irBuilder.CreateConstInBoundsGEP1_32( irBuilder.getInt32Ty(), payload32, i );
            irBuilder.CreateStore( payloadElt, gep );
        }
    }
    else
    {
        // Write back the resulting payload into the payload registers.
        for( unsigned int i = 0; i < numPayloadRegisters; ++i )
        {
            Value* payloadElt = irBuilder.CreateExtractElement( result, irBuilder.getInt32( i ), "payload" );
            irBuilder.CreateCall( lwvmWritePayload, { irBuilder.getInt32( i ), payloadElt } );
        }
    }
}

// ensure that ray flags that get passed from OptiX to rtcore match
#define STATIC_EQUAL( a, b ) static_assert( (unsigned)a == (unsigned)b, "" );
STATIC_EQUAL( RT_RAY_FLAG_NONE, RTC_RAY_FLAG_NONE );
STATIC_EQUAL( RT_RAY_FLAG_DISABLE_ANYHIT, RTC_RAY_FLAG_FORCE_OPAQUE );
STATIC_EQUAL( RT_RAY_FLAG_FORCE_ANYHIT, RTC_RAY_FLAG_FORCE_NO_OPAQUE );
STATIC_EQUAL( RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT, RTC_RAY_FLAG_TERMINATE_ON_FIRST_HIT );
STATIC_EQUAL( RT_RAY_FLAG_DISABLE_CLOSESTHIT, RTC_RAY_FLAG_SKIP_CLOSEST_HIT_SHADER );
STATIC_EQUAL( RT_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES, RTC_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES );
STATIC_EQUAL( RT_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES, RTC_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES );
STATIC_EQUAL( RT_RAY_FLAG_LWLL_DISABLED_ANYHIT, RTC_RAY_FLAG_LWLL_OPAQUE );
STATIC_EQUAL( RT_RAY_FLAG_LWLL_ENABLED_ANYHIT, RTC_RAY_FLAG_LWLL_NON_OPAQUE );

// -----------------------------------------------------------------------------
void RTXCompile::lowerTrace( Module* module ) const
{
    DataLayout   DL( module );
    LLVMContext& llvmContext = module->getContext();

    Type* floatTy           = Type::getFloatTy( llvmContext );
    Type* i32Ty             = Type::getInt32Ty( llvmContext );
    Type* i64Ty             = Type::getInt64Ty( llvmContext );
    Type* rayFlagsTy        = TraceGlobalPayloadBuilder::getRayFlagsType( llvmContext );
    Type* rayMaskTy         = TraceGlobalPayloadBuilder::getRayMaskType( llvmContext );
    Type* sbtRecordOffsetTy = Type::getIntNTy( llvmContext, SBT_RECORD_OFFSET_NBITS );
    Type* sbtRecordStrideTy = Type::getIntNTy( llvmContext, SBT_RECORD_STRIDE_NBITS );

    Function* getRtcTraversableHandle = getFunctionOrAssert( module, "RTX_getRtcTraversableHandle" );
    Function* getRayTypeCount = getFunctionOrAssert( module, "_ZN4cort22Global_getRayTypeCountEPNS_14CanonicalStateE" );

    FunctionType* readPayloadType  = FunctionType::get( i32Ty, i32Ty, false );
    FunctionType* writePayloadType = FunctionType::get( Type::getVoidTy( llvmContext ), Types{i32Ty, i32Ty}, false );
    Function*     lwvmReadPayload  = insertOrCreateFunction( module, "lw.rt.read.payload.i32", readPayloadType );
    Function*     lwvmWritePayload = insertOrCreateFunction( module, "lw.rt.write.payload.i32", writePayloadType );

    corelib::CoreIRBuilder irBuilder( llvmContext );

    for( const auto& F : corelib::getFunctions( module ) )
    {
        if( !F->isDeclaration() || !F->getName().startswith( "optixi_" ) )
            continue;

        bool globalPayload = TraceGlobalPayloadCall::isIntrinsic( F );
        bool localPayload  = parseTraceName( F->getName() );

        if( !globalPayload && !localPayload )
            continue;

        RT_ASSERT( globalPayload ^ localPayload );

        // The value type is the same as the last argument
        FunctionType* fntype = F->getFunctionType();
        RT_ASSERT( fntype->getNumParams() == 16 );

        if( m_inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM )
        {
            Function* trap = nullptr;
            for( CallInst* call : getCallsToFunction( F ) )
            {
                if( !trap )
                    trap = dyn_cast<Function>(
                        module->getOrInsertFunction( "llvm.trap", FunctionType::get( Type::getVoidTy( llvmContext ), {} ) ) );
                irBuilder.SetInsertPoint( call );
                irBuilder.CreateCall( trap );
                call->eraseFromParent();
            }
            continue;
        }

        // Pull out arguments
        Function::arg_iterator args         = F->arg_begin();
        Value*                 statePtr     = args++;
        Value*                 topOffsetArg = args++;
        Value*                 ox           = args++;
        Value*                 oy           = args++;
        Value*                 oz           = args++;
        Value*                 dx           = args++;
        Value*                 dy           = args++;
        Value*                 dz           = args++;
        Value*                 rayTypeArg   = args++;
        Value*                 tmin         = args++;
        Value*                 tmax         = args++;
        Value*                 time         = args++;
        Value*                 hastime      = args++;
        Value*                 rayMask      = args++;
        Value*                 rayFlags     = args++;
        Value*                 payloadArg   = localPayload ? args++ : nullptr;

        RT_ASSERT( rayMask->getType() == rayMaskTy );
        RT_ASSERT( rayFlags->getType() == rayFlagsTy );

        BasicBlock* B = BasicBlock::Create( llvmContext, "trace_trampoline", F );
        irBuilder.SetInsertPoint( B );

        // NB: The following computation assumes that the BVH does not live in the texture heap.
        // TODO:
        // We need to retrieve the pointer to the bvh buffer.
        // At this point we only have the offset of the top Geometry Group (GG_offset)
        // We can retrieve the base address of the bvh bffer in the following way:
        // BufferId = *( getAcceleration( GG_offset ) + 0x4)
        // 0x4 is the offset of the variable 'bvh' in the Acceleration record.
        // Once we have the BufferId we can call Buffer_getElementAddress1dFromId_linear to get the address.
        // Here we make the assumption that Acceleration always has the 'bvh' variable attached and that it lives at offset 0x4.
        Value* rayTypesCount = irBuilder.CreateCall( getRayTypeCount, statePtr, "ray.type.count" );
        Value* travHandle =
            irBuilder.CreateCall( getRtcTraversableHandle, Values{statePtr, topOffsetArg}, "bvh.address" );

        // Pack payload
        unsigned int payloadSize = 0;
        if( localPayload )
        {
            RT_ASSERT( payloadArg->getType()->isPointerTy() );
            payloadSize = DL.getTypeStoreSize( payloadArg->getType()->getPointerElementType() );
        }
        else
        {
            payloadSize = TraceGlobalPayloadCall::getPayloadSize( F );
        }
        unsigned int numPayloadRegisters         = 0;
        unsigned int numPayloadPlusSizeRegisters = 0;
        Value*       payloadValue                = nullptr;
        Type*        payloadTy                   = nullptr;
        packPayload( localPayload, payloadSize, payloadArg, lwvmReadPayload, numPayloadRegisters,
                     numPayloadPlusSizeRegisters, payloadValue, payloadTy, B );

        // Issue the call.
        // TODO: We should check somewhere that the ray type count does not exceed 4 bits.
        RT_ASSERT( sbtRecordOffsetTy->getPrimitiveSizeInBits() < 32 );
        RT_ASSERT( sbtRecordStrideTy->getPrimitiveSizeInBits() < 32 );
        Value* sbtRecordOffset = irBuilder.CreateTrunc( rayTypeArg, sbtRecordOffsetTy );
        Value* sbtRecordStride = irBuilder.CreateTrunc( rayTypesCount, sbtRecordStrideTy );
        // DXRT allows miss shader index and sbtRecordOffset to be decoupled. This is not yet used in RTX.
        Value* missIndex = irBuilder.CreateZExt( sbtRecordOffset, i32Ty );

        // call trace or motion trace, depending on hasTime
        BasicBlock* mblurTraceBB = BasicBlock::Create( llvmContext, "trace.mblur", F );
        BasicBlock* traceBB      = BasicBlock::Create( llvmContext, "trace", F );
        BasicBlock* mergeBB      = BasicBlock::Create( llvmContext, "trace.merge", F );

        Value* zeroVal = ConstantInt::getNullValue( i32Ty );
        Value* hasTime = irBuilder.CreateICmpNE( hastime, zeroVal, "hasTime" );
        irBuilder.CreateCondBr( hasTime, mblurTraceBB, traceBB );

        // Create motion blur trace call in motion trace basic block
        irBuilder.SetInsertPoint( mblurTraceBB );
        std::string mfuncName = "lw.rt.trace.mblur." + std::to_string( numPayloadPlusSizeRegisters ) + "reg";
        Type*       margsTy[] = {i64Ty,   rayFlagsTy, rayMaskTy, sbtRecordOffsetTy, sbtRecordStrideTy, i32Ty,
                           floatTy, floatTy,    floatTy,   floatTy,           floatTy,           floatTy,
                           floatTy, floatTy,    floatTy,   payloadTy};
        FunctionType* mfuncTy          = FunctionType::get( payloadTy, margsTy, false );
        Constant*     mblurTraceFunc   = cast<Constant>( module->getOrInsertFunction( mfuncName, mfuncTy ) );
        Value*        mblurTraceArgs[] = {
            travHandle, rayFlags, rayMask, sbtRecordOffset, sbtRecordStride, missIndex, ox, oy, oz, tmin, dx, dy,
            dz,         tmax,     time,    payloadValue};
        Value* mblurTraceCall = irBuilder.CreateCall( mblurTraceFunc, mblurTraceArgs, "payload" );
        irBuilder.CreateBr( mergeBB );

        // Create blur trace call in trace basic block
        irBuilder.SetInsertPoint( traceBB );
        std::string funcName = "lw.rt.trace." + std::to_string( numPayloadPlusSizeRegisters ) + "reg";
        Type*       argsTy[] = {i64Ty,   rayFlagsTy, rayMaskTy, sbtRecordOffsetTy, sbtRecordStrideTy,
                          i32Ty,   floatTy,    floatTy,   floatTy,           floatTy,
                          floatTy, floatTy,    floatTy,   floatTy,           payloadTy};
        FunctionType* funcTy      = FunctionType::get( payloadTy, argsTy, false );
        Constant*     traceFunc   = cast<Constant>( module->getOrInsertFunction( funcName, funcTy ) );
        Value*        traceArgs[] = {
            travHandle, rayFlags, rayMask, sbtRecordOffset, sbtRecordStride, missIndex, ox, oy, oz, tmin, dx,
            dy,         dz,       tmax,    payloadValue};
        Value* traceCall = irBuilder.CreateCall( traceFunc, traceArgs, "payload" );
        irBuilder.CreateBr( mergeBB );

        // Merge the result of both calls in merge basic block
        irBuilder.SetInsertPoint( mergeBB );
        PHINode* result = irBuilder.CreatePHI( mblurTraceCall->getType(), 2 );
        result->addIncoming( mblurTraceCall, mblurTraceBB );
        result->addIncoming( traceCall, traceBB );

        unpackPayload( localPayload, numPayloadRegisters, result, payloadArg, lwvmWritePayload, mergeBB );

        irBuilder.CreateRetVoid();
    }
}

// -----------------------------------------------------------------------------
llvm::Function* RTXCompile::lowerBoundCallableProgram( llvm::Function* function ) const
{

    Module* module = function->getParent();

    // Replace the original function with a function that receives the caller's
    // SBT index and potentially the bound callable program state as additional parameters.
    // The values of those parameters are filled in lowerCalls.

    LLVMContext& llvmContext = module->getContext();
    Type*        i64Ty       = Type::getInt64Ty( llvmContext );

    if( m_stype != ST_BOUND_CALLABLE_PROGRAM )
    {
        // Clean up bound callable helpers
        //
        // In order to be able to rewrite them, the two get*TransformMatrix functions have
        // been marked as noinline. Make sure that they are inlined in all non-bound callable
        // functions.
        Type*     i32Ty             = Type::getInt32Ty( llvmContext );
        Function* worldToObjectFunc = module->getFunction( "getWorldToObjectTransformMatrix" );
        Function* objectToWorldFunc = module->getFunction( "getObjectToWorldTransformMatrix" );
        inlineAllCalls( worldToObjectFunc );
        inlineAllCalls( objectToWorldFunc );
        worldToObjectFunc->eraseFromParent();
        objectToWorldFunc->eraseFromParent();

        // Replace payload access proxy with actual lw.rt intrinsics.
        Function* readPayloadProxy = module->getFunction( "proxy.lw.rt.read.payload.i32" );
        if( readPayloadProxy )
        {
            FunctionType* readPayloadType = FunctionType::get( i32Ty, i32Ty, false );
            Function*     lwvmReadPayload = insertOrCreateFunction( module, "lw.rt.read.payload.i32", readPayloadType );
            readPayloadProxy->replaceAllUsesWith( lwvmReadPayload );
            readPayloadProxy->eraseFromParent();
        }
        return function;
    }

    FunctionType* funcTy = function->getFunctionType();

    // Copy argument types
    std::vector<Type*> newParamType;
    for( Function::arg_iterator A = function->arg_begin(); A != function->arg_end(); ++A )
    {
        newParamType.push_back( A->getType() );
    }

    // add new argument for SBT data pointer of caller
    newParamType.push_back( i64Ty );

    if( m_inheritedStype != ST_RAYGEN && m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM )
    {
        // Add new argument for bound callable program state.
        // Raygen and bindless do not have the bound callable state, since it does
        // not need to access the matrices or the attributes.
        newParamType.push_back( i64Ty );
    }

    // Create type of new function.
    FunctionType* newFuncTy = FunctionType::get( funcTy->getReturnType(), newParamType, false );
    // Create the new function.
    Function* newFunc = Function::Create( newFuncTy, function->getLinkage(), "", function->getParent() );
    newFunc->takeName( function );


    // Copy function attributes but no others
    newFunc->setAttributes( newFunc->getAttributes().addAttributes( newFunc->getContext(), AttributeList::FunctionIndex,
                                                                    function->getAttributes().getFnAttributes() ) );

    // Splice the body.
    newFunc->getBasicBlockList().splice( newFunc->begin(), function->getBasicBlockList() );
    RT_ASSERT( function->getBasicBlockList().empty() );

    // Replace uses of the arguments.
    Function::arg_iterator oldArg = function->arg_begin();
    Function::arg_iterator newArg = newFunc->arg_begin();
    for( ; oldArg != function->arg_end(); ++oldArg, ++newArg )
    {
        RT_ASSERT( newArg != newFunc->arg_end() );
        oldArg->replaceAllUsesWith( newArg );
    }

    // Erase the original function.
    RT_ASSERT( function->use_empty() );
    function->eraseFromParent();

    // Sbtptr replacement is done later, because it needs to run
    // after replacePlaceholderAccessors.

    if( m_inheritedStype != ST_RAYGEN && m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM )
    {
        loadValuesFromBcpStateAndReplaceCalls( newFunc );
    }
    else
    {
        // getWorldToObjectTransformMatrix and getObjectToWorldTransformMatrix
        // are no longer marked as always inline in order to be able to rewrite them.
        // For bound callable programs that are not called from RG or bindless this
        // is rewritten in loadValuesFromBcpStateAndReplaceCalls, for non callables
        // in runOnFunction. We need to inline them for the called from RG or bindless case here.
        Function* worldToObjectFunc = module->getFunction( "getWorldToObjectTransformMatrix" );
        Function* objectToWorldFunc = module->getFunction( "getObjectToWorldTransformMatrix" );
        inlineAllCalls( worldToObjectFunc );
        inlineAllCalls( objectToWorldFunc );
        worldToObjectFunc->eraseFromParent();
        objectToWorldFunc->eraseFromParent();
    }

    return newFunc;
}

// -----------------------------------------------------------------------------
void RTXCompile::rewriteBoundCallableProgramParentSbtPointer( llvm::Function* function ) const
{
    Module* module = function->getParent();
    RT_ASSERT( ( ( m_inheritedStype != ST_RAYGEN || m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM ) && function->arg_size() >= 2 )
               || function->arg_size() > 0 );

    auto   arg    = function->arg_end();
    --arg;

    Value* sbtPtr = nullptr;
    if( m_inheritedStype == ST_RAYGEN || m_inheritedStype == ST_BINDLESS_CALLABLE_PROGRAM )
    {
        // Raygen and bindless do not have the bound callable program state,
        // so the sbtPtr is the last argument.
        sbtPtr = &( *arg );
    }
    else
    {
        // Skip the bound callable program state to get to the sbtPtr.
        --arg;
        sbtPtr = &(*arg);
    }

    Function* callee = module->getFunction( "_Z35getGeometryInstanceSBTRecordPointerv" );
    RT_ASSERT( callee );
    if( !inlineAllCallersOfFunction( callee, /*eraseUnusedFunctions*/ true ) )
        throw CompileError( RT_EXCEPTION_INFO, "Unable to inline callers of: " + callee->getName().str() );

    // Replace uses of calls to _Z35getGeometryInstanceSBTRecordPointerv() by sbtPtr argument and remove calls.
    for( CallInst* CI : getCallsToFunction( callee, function ) )
    {
        corelib::CoreIRBuilder irb{CI};
        CI->replaceAllUsesWith( irb.CreateIntToPtr( sbtPtr, callee->getReturnType(), "parentSbt" ) );
        CI->eraseFromParent();
    }
    // In case there are any calls to bound callable programs from within this bcp
    // those calls will have the placeholder function "lw.optix.sbt.param.proxy"
    // as the last parameter value. Replace those function calls with the newly added
    // parameter.
    Function* lwvmReadSbtDataPtr = module->getFunction( "lw.optix.sbt.param.proxy" );
    for( CallInst* CI : getCallsToFunction( lwvmReadSbtDataPtr, function ) )
    {
        corelib::CoreIRBuilder irb{CI};
        CI->replaceAllUsesWith( irb.CreateIntToPtr( sbtPtr, lwvmReadSbtDataPtr->getReturnType(), "inheritedSbt" ) );
        CI->eraseFromParent();
    }
}


// -----------------------------------------------------------------------------
void RTXCompile::lowerCallableProgramCalls( Module* module ) const
{
    for( Function& F : *module )
    {
        if( !F.isDeclaration()
            || ( !F.getName().startswith( "optixi_callBindless" ) && !F.getName().startswith( "optixi_callBound" ) ) )
            continue;

        bool      isBound;
        StringRef uniqueName;
        unsigned  sig;
        bool      result = parseCallableProgramName( F.getName(), isBound, uniqueName, sig );
        RT_ASSERT_MSG( result, LLVMErrorInfo( &F ) + " Invalid callable program placeholder: " + F.getName().str() );
        std::string functionName = F.getName().str();

        bool alwaysUseContinuationCallables = k_useContinuationCallables.get();
        bool isHeavyweight                  = false;

        if( !isBound )
        {
            auto it = m_heavyWeightCallSites.find( uniqueName );
            if( it != m_heavyWeightCallSites.end() )
                isHeavyweight = true;
        }

        // Create the appropriate LWVM RT intrinsic that RTCore will understand.
        // %outputType @lw.rt.call.direct.T( i32 %callableSbtRecordIndex, %inputType %inputs ) nounwind
        // %outputType @lw.rt.call.continuation.T( i32 %continuationSbtRecordIndex, %inputType %inputs ) nounwind
        std::string rtcoreName;

        // Note that RTXPlan::isDirectCalledBoundCallable also works for heavyweight bindless
        // callables because their inherited semantic type is ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE.
        if( ( isHeavyweight || alwaysUseContinuationCallables || isBound )
            && !RTXPlan::isDirectCalledBoundCallable( ST_BOUND_CALLABLE_PROGRAM, m_inheritedStype ) )
        {
            rtcoreName = "lw.rt.call.continuation." + std::to_string( sig );
        }
        else
            rtcoreName = "lw.rt.call.direct." + std::to_string( sig );

        lowerCalls( module, F, rtcoreName, isBound );
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerCalls( Module* module, Function& F, const std::string& rtcoreName, bool isBound ) const
{
    LLVMContext& llvmContext = module->getContext();
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );
    Type*        i64Ty       = Type::getInt64Ty( llvmContext );

    // Replace all uses of the function F.
    Function* callCallableFunc = nullptr;
    for( CallInst* CI : getCallsToFunction( &F ) )
    {
        Instruction*           insertBefore = CI;
        corelib::CoreIRBuilder irb{CI};

        // Split the arguments into individual 32-bit values. Skip the
        // CanonicalState (arg 0) and the programId (arg 1).
        SmallVector<Value*, 8> paramVals;
        for( unsigned int i = 2, e = CI->getNumArgOperands(); i < e; ++i )
        {
            Value* arg = CI->getArgOperand( i );

            std::vector<Value*> flattened = corelib::flattenAggregateTo32BitValuesForRTCore( arg, insertBefore );
            paramVals.insert( paramVals.end(), flattened.begin(), flattened.end() );
        }
        if( isBound )
        {
            // Bound callable programs need additional arguments to access
            // the caller's SBT entry for dynamic variable lookup and potentially
            // additional values to access payload, attributes and transformations.
            FunctionType* readSbtDataPtrTy = FunctionType::get( i64Ty, false );
            Function*     readSbtFunc      = nullptr;
            if( m_stype != ST_BOUND_CALLABLE_PROGRAM )
            {
                // add caller's sbt data pointer as last parameter
                readSbtFunc = insertOrCreateFunction( module, "lw.rt.read.sbt.data.ptr", readSbtDataPtrTy );
            }
            else
            {
                // The caller's sbt data pointer was passed in as last parameter to the bound
                // callable program. Create a placeholder that will be replaced in lowerBoundCallableProgram.
                readSbtFunc = insertOrCreateFunction( module, "lw.optix.sbt.param.proxy", readSbtDataPtrTy );
            }

            {
                Value*              sbtDataPtr = irb.CreateCall( readSbtFunc );
                std::vector<Value*> flattened = corelib::flattenAggregateTo32BitValuesForRTCore( sbtDataPtr, insertBefore );
                paramVals.insert( paramVals.end(), flattened.begin(), flattened.end() );
            }

            if( m_inheritedStype != ST_RAYGEN && m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM )
            {
                // Add the bound callable program state which allows access
                // to the transformation matrices, the payload and the attributes.
                Value* bcpState = fillBoundCallableProgramState( F, insertBefore );

                std::vector<Value*> flattened =
                    corelib::flattenAggregateTo32BitValuesForRTCore( irb.CreatePtrToInt( bcpState, i64Ty ), insertBefore );
                paramVals.insert( paramVals.end(), flattened.begin(), flattened.end() );
            }
        }

        const int  returlwalueNumRegs       = corelib::getNumRequiredRegisters( F.getReturnType() );
        const int  newReturlwalueNumRegs    = std::min( returlwalueNumRegs, m_params.numCallableParamRegisters );
        const bool needsReturlwalueSpilling = returlwalueNumRegs > m_params.numCallableParamRegisters;

        const int parametersNumRegs         = paramVals.size();
        const int adjustedParametersNumRegs = parametersNumRegs + ( needsReturlwalueSpilling ? 2 : 0 );
        // const int newParametersNumRegs = std::min( adjustedParametersNumRegs, m_params.numCallableParamRegisters );
        const bool needsParameterSpilling = adjustedParametersNumRegs > m_params.numCallableParamRegisters;

        Value* spillBaseAddr = nullptr;
        if( needsParameterSpilling || needsReturlwalueSpilling )
        {
            // Create local memory for surplus parameters or return values. The two last parameters are
            // used for the pointer to the local memory.
            int bufferSize = std::max( parametersNumRegs, returlwalueNumRegs ) - ( m_params.numCallableParamRegisters - 2 );
            Type*       allocaTy = ArrayType::get( i32Ty, bufferSize );
            AllocaInst* alloca = corelib::CoreIRBuilder{&*CI->getParent()->getParent()->getEntryBlock().begin()}.CreateAlloca(
                allocaTy, nullptr, "spillAlloca" );
            alloca->setAlignment( 4 );

            // Store surplus parameters in local memory.
            for( int paramIdx = m_params.numCallableParamRegisters - 2; paramIdx < (int)paramVals.size(); ++paramIdx )
            {
                Value* indices[] = {ConstantInt::get( i32Ty, 0 ),
                                    ConstantInt::get( i32Ty, paramIdx - ( m_params.numCallableParamRegisters - 2 ) )};
                Value* spillAddr = irb.CreateGEP( alloca, indices, "spillAddr" );
                Value* arg       = paramVals[paramIdx];
                if( arg->getType() != i32Ty )
                    arg = irb.CreateBitCast( arg, i32Ty );
                irb.CreateStore( arg, spillAddr );
            }

            // Remove surplus parameters from the parameter vector.
            if( (int)paramVals.size() > m_params.numCallableParamRegisters - 2 )
                paramVals.resize( m_params.numCallableParamRegisters - 2 );

            // Add alloca pointer as the last two parameters.
            Value* indices[]         = {ConstantInt::get( i32Ty, 0 ), ConstantInt::get( i32Ty, 0 )};
            spillBaseAddr            = irb.CreateGEP( alloca, indices, "spillBaseAddr" );
            Value* spillBaseAddr_i64 = irb.CreatePtrToInt( spillBaseAddr, i64Ty, "spillBaseAddr.i64" );
            std::pair<Value*, Value*> flattened = createCast_i64_to_2xi32( spillBaseAddr_i64, CI );
            paramVals.push_back( flattened.first );
            paramVals.push_back( flattened.second );
        }

        RT_ASSERT_MSG( (int)paramVals.size() <= m_params.numCallableParamRegisters,
                       "logic error in parameter spilling" );

        // Pack the individual 32-bit values into an aggregate of 32-bit types
        // (int or float) as required by the RTCore spec.
        Type*  paramArrayTy = ArrayType::get( i32Ty, paramVals.size() );
        Value* paramArray   = UndefValue::get( paramArrayTy );
        for( int i = 0, e = paramVals.size(); i < e; ++i )
        {
            Value* arg = paramVals[i];
            if( arg->getType() != i32Ty )
                arg    = irb.CreateBitCast( arg, i32Ty );
            paramArray = irb.CreateInsertValue( paramArray, arg, i );
        }

        // Create the function lazily after we know the parameter aggregate type.
        if( !callCallableFunc )
        {
            Type* newReturnTy = F.getReturnType()->isVoidTy() ? F.getReturnType() : ArrayType::get( i32Ty, newReturlwalueNumRegs );
            Type*         argTypes[] = {i32Ty, paramArrayTy};
            FunctionType* newFuncTy  = FunctionType::get( newReturnTy, argTypes, false );
            callCallableFunc         = dyn_cast<Function>( module->getOrInsertFunction( rtcoreName, newFuncTy ) );
            RT_ASSERT_MSG( callCallableFunc, "function already exists with different type: " + rtcoreName );
            callCallableFunc->addFnAttr( Attribute::NoUnwind );
            if( needsParameterSpilling || needsReturlwalueSpilling )
                callCallableFunc->removeFnAttr( Attribute::ReadNone );
            if( needsReturlwalueSpilling )
                callCallableFunc->removeFnAttr( Attribute::ReadOnly );
        }

        // The SBT record index for a callable program is stored in the program
        // header. The second argument of the function F is the program
        // ID, which needs to be used to access the program header.
        // The header contains the base SBT index for all records of the callable program.
        // Bindless callables only have one record at exactly that base index.
        // Bound callables have multiple records that are offset using
        // the inherited semantic type.
        Value* programId      = CI->getArgOperand( 1 );
        Value* canonicalState = CI->getArgOperand( 0 );

        Function* programIdToSbtIndexFunc =
            module->getFunction( "_ZN4cort31CallableProgram_getSBTBaseIndexEPNS_14CanonicalStateEj" );
        RT_ASSERT( programIdToSbtIndexFunc );
        SmallVector<Value*, 3> sbtFuncArgs;
        sbtFuncArgs.push_back( canonicalState );
        sbtFuncArgs.push_back( programId );
        Value* sbtRecordIndex = irb.CreateCall( programIdToSbtIndexFunc, sbtFuncArgs );
        if( isBound )
        {
            sbtRecordIndex =
                irb.CreateAdd( sbtRecordIndex, ConstantInt::get( Type::getInt32Ty( llvmContext ), m_inheritedStype ) );
        }

        Value* args[]   = {sbtRecordIndex, paramArray};
        Value* newValue = irb.CreateCall( callCallableFunc, args );

        if( !newValue->getType()->isVoidTy() )
        {
            if( needsReturlwalueSpilling )
            {
                // Create aggregate value for all return values.
                Type*  aggregateTy    = ArrayType::get( i32Ty, returlwalueNumRegs );
                Value* aggregateValue = UndefValue::get( aggregateTy );

                // Copy all return values in registers into aggregate value.
                for( int retValIdx = 0; retValIdx < m_params.numCallableParamRegisters - 2; ++retValIdx )
                {
                    Value* returlwalue = irb.CreateExtractValue( newValue, retValIdx, "returlwalue" );
                    aggregateValue = irb.CreateInsertValue( aggregateValue, returlwalue, retValIdx, "aggregateValue" );
                }

                // Compute address of spilled return values.
                //
                // The address has already been computed for parameter spilling above. We could reuse it here and pass two
                // more return values in registers. But this might require spilling of the address itself, and it causes the
                // unit tests with m_params.numCallableParamRegisters-1 and m_params.numCallableParamRegisters parameters to fail for unknown reasons.
                spillBaseAddr =
                    corelib::unflattenAggregateForRTCore( i64Ty, newValue, m_params.numCallableParamRegisters - 2, insertBefore );
                spillBaseAddr = irb.CreateIntToPtr( spillBaseAddr, i32Ty->getPointerTo(), "spillBaseAddr" );

                // Append spilled return values to aggregate value.
                for( int retValIdx = m_params.numCallableParamRegisters - 2; retValIdx < returlwalueNumRegs; ++retValIdx )
                {
                    Value* spillIndices = ConstantInt::get( i32Ty, retValIdx - ( m_params.numCallableParamRegisters - 2 ) );
                    Value* spillAddr          = irb.CreateGEP( spillBaseAddr, spillIndices, "spillAddr" );
                    Value* spilledReturlwalue = irb.CreateLoad( spillAddr, "spilledParam" );
                    aggregateValue =
                        irb.CreateInsertValue( aggregateValue, spilledReturlwalue, retValIdx, "aggregateValue" );
                }

                newValue = aggregateValue;
            }

            newValue = corelib::unflattenAggregateForRTCore( F.getReturnType(), newValue, 0, insertBefore );
            newValue->takeName( CI );
        }

        CI->replaceAllUsesWith( newValue );
        CI->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
Type* RTXCompile::getBoundCallableProgramStateType( Module* module, std::map<unsigned short, std::pair<int, std::string>>& bcpStateAttributeMappings ) const
{
    // The bound callable program state contains the information that needs
    // to be available for bound callable programs in order to perform
    // scoped variable lookup, call rtTransform* functions, and to read attributes.
    // It's a LLVM struct that contains:
    //
    //   i8*    |      Matrix4x4     |      Matrix4x4       |    m x Array<n, i32> (in AH and CH)
    // --------------------------------------------------------------------------------------------
    // Payload  | World to Object    |   Object to world    |   All m assigned attribute values
    // pointer  | transform matrix   |   transform matrix   |   stored as arrays of 32 Bit values
    // --------------------------------------------------------------------------------------------
    //
    // Note that the payload pointer is not added if the payload has been promoted
    // to registers. In that case no bound callable program accesses the payload
    // since that would disable that promotion.

    RT_ASSERT( m_inheritedStype != ST_RAYGEN && m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM );

    LLVMContext& llvmContext = module->getContext();

    std::vector<Type*> elements;

#if ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED  // TODO bigler rip out
    int attrIndex = 0;
#endif
    if( !m_params.payloadInRegisters )
    {
        Type* payloadPtrTy = Type::getInt8PtrTy( llvmContext );
        elements.push_back( payloadPtrTy );
#if ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED  // TODO bigler rip out
        ++attrIndex;
#endif
    }

    Function* matFunc = module->getFunction( "getWorldToObjectTransformMatrix" );
    RT_ASSERT( matFunc );
    Type* matrixType = matFunc->getReturnType();
    RT_ASSERT( matrixType );
    elements.push_back( matrixType );
    elements.push_back( matrixType );
#if ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED
    attrIndex += 2;
#endif

    if( m_inheritedStype == ST_CLOSEST_HIT || m_inheritedStype == ST_ANY_HIT )
    {
#if ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED
        Type* i32Ty = Type::getInt32Ty( llvmContext );
        RT_ASSERT_FAIL_MSG( "SGP: handle atts" );
#if 0
        // Attribute handling
        for( const auto& assignment : m_attributeAssignments )
        {
            unsigned short token = assignment.first;
            std::string    varrefUniqueName;
            int            elementCount = assignment.second.size;

            const ProgramManager::VariableReferenceIDListType& list = m_programManager->getReferencesForVariable( token );
            RT_ASSERT( list.size() );
            std::string attrName = m_programManager->getVariableReferenceById( list.front() )->getInputName();
#if defined( DEBUG ) || defined( DEVELOP )
            // sanity type check
            for( VariableReferenceID id : list )
            {
                const VariableReference* varref = m_programManager->getVariableReferenceById( id );
                size_t                   s      = varref->getType().computeSize();
                int                      c      = ( s + 3 ) / 4;
                RT_ASSERT( c == elementCount );
            }
#endif  // DEBUG || DEVELOP

            Type* attrType = ArrayType::get( i32Ty, elementCount );

            bcpStateAttributeMappings.insert( {token, std::make_pair( attrIndex, attrName )} );
            elements.push_back( attrType );
            ++attrIndex;
        }
#endif
#endif  // ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED
    }
    Type* bcpStateTy = StructType::get( llvmContext, elements );
    return bcpStateTy;
}

// -----------------------------------------------------------------------------
Value* RTXCompile::fillBoundCallableProgramState( llvm::Function& callable, Instruction* insertBefore ) const
{
    RT_ASSERT( m_inheritedStype != ST_RAYGEN && m_inheritedStype != ST_BINDLESS_CALLABLE_PROGRAM );

    Module*                module = callable.getParent();
    corelib::CoreIRBuilder irb{insertBefore};

    std::map<unsigned short, std::pair<int, std::string>> bcpStateAttributeMappings;
    Type* bcpStateTy = getBoundCallableProgramStateType( module, bcpStateAttributeMappings );

    Value* bcpState = nullptr;

    if( m_stype != ST_BOUND_CALLABLE_PROGRAM )
    {

        AllocaInst* alloca = corelib::CoreIRBuilder{&*insertBefore->getParent()->getParent()->getEntryBlock().begin()}.CreateAlloca(
            bcpStateTy, nullptr, "bcpStateAlloca" );

        // Iterator index for the struct elements.
        int structIdx = 0;

        if( !m_params.payloadInRegisters )
        {
            // We only need to take care of the payload, if it hasn't been put
            // into registers. If it is in registers it means that no bound callable
            // program tries to access it.

            Value*    payloadPtr            = irb.CreateStructGEP( alloca, structIdx++, "payloadPtr" );
            Function* getPayloadAddressFunc = module->getFunction( "RTX_getPayloadPointer" );
            RT_ASSERT( getPayloadAddressFunc );
            Value* addr = irb.CreateCall( getPayloadAddressFunc );
            RT_ASSERT( addr->getType() == payloadPtr->getType()->getPointerElementType() );
            irb.CreateStore( addr, payloadPtr );
        }

        // The transformation matrices
        Value*    mat1     = irb.CreateStructGEP( alloca, structIdx++, "world2ObjectMat" );
        Function* mat1Func = module->getFunction( "getWorldToObjectTransformMatrix" );
        RT_ASSERT( mat1Func );
        Value* m = irb.CreateCall( mat1Func );
        irb.CreateStore( m, mat1 );

        Value*    mat2     = irb.CreateStructGEP( alloca, structIdx++, "object2WorldMat" );
        Function* mat2Func = module->getFunction( "getObjectToWorldTransformMatrix" );
        RT_ASSERT( mat2Func );
        Value* m2 = irb.CreateCall( mat2Func );
        irb.CreateStore( m2, mat2 );

#if ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED
        if( m_inheritedStype == ST_CLOSEST_HIT || m_inheritedStype == ST_ANY_HIT )
        {
            RT_ASSERT_FAIL_MSG( "HANDLE ATTS" );
#if 0
            // The attributes
            int totalNumAttrRegs    = 0;
            int totalNumAttrMemRegs = 0;
            for( auto iter : m_attributeAssignments )
            {
                if( iter.second.memory )
                    totalNumAttrMemRegs = std::max( totalNumAttrMemRegs, iter.second.offset + iter.second.size );
                else
                    totalNumAttrRegs += iter.second.size;
            }
            for( const auto& attr : bcpStateAttributeMappings )
            {
                // Fill in current attribute values
                Value*       ptr = irb.CreateStructGEP( alloca, attr.second.first, attr.second.second + "Ptr" );
                Instruction* insertionPoint = irb.CreateUnreachable();
                lowerAttributesForBoundCallableProgram( module, attr.first, ptr, insertionPoint, totalNumAttrRegs, totalNumAttrMemRegs );
                insertionPoint->removeFromParent();
            }
#endif
        }
#endif  // ATTRIBUTES_IN_BOUND_CALLABLE_PROGRAMS_ALLOWED

        bcpState = alloca;
    }
    else
    {
        // The bound callable program state was passed in as a pointer. Pass that to the callee.
        Function* bcpParamProxy = insertOrCreateFunction( module, "lw.optix.bcpState.proxy",
                                                          FunctionType::get( bcpStateTy->getPointerTo(), false ) );
        bcpState = irb.CreateCall( bcpParamProxy );
    }
    return bcpState;
}

// -----------------------------------------------------------------------------
void optix::RTXCompile::lowerAttributesForBoundCallableProgram( Module*            module,
                                                                unsigned short     token,
                                                                llvm::Value*       alloca,
                                                                llvm::Instruction* insertBefore,
                                                                int                totalNumAttrRegs,
                                                                int                totalNumAttrMemRegs ) const
{
    RT_ASSERT_FAIL_MSG( "HANDLE ATTS" );

#if 0
    LLVMContext&  llvmContext = module->getContext();
    Type*         i32Ty       = Type::getInt32Ty( llvmContext );
    FunctionType* readType    = FunctionType::get( i32Ty, i32Ty, false );
    Constant*     readRegFn   = module->getOrInsertFunction( "lw.rt.read.register.attribute.i32", readType );
    Constant*     readMemFn   = module->getOrInsertFunction( "lw.rt.read.memory.attribute.i32", readType );

    auto iter = m_attributeAssignments.find( token );

    // We want to load the complete attribute value here not
    // only elements of it like in lowerAttributesForCHandAH
    const int  attributeBaseOffset = iter->second.offset;
    const int  attributeNumRegs    = iter->second.size;
    const bool attributeIsInMemory = iter->second.memory;

    corelib::CoreIRBuilder irb{insertBefore};

    // Create RTCore intrinsics to load the attribute value and store them to the
    // bound callable program state. The last register that we can safely load is the one at
    // attributeNumRegs-1. In cases where the target type is too large, stop
    // after that load.
    const int lastRegToLoad = attributeBaseOffset + attributeNumRegs - 1;

    // Use different intrinsics depending on the location of the attribute.
    Constant* readFn = attributeIsInMemory ? readMemFn : readRegFn;

    for( int reg = attributeBaseOffset; reg <= lastRegToLoad; ++reg )
    {
        RT_ASSERT_MSG( reg < ( attributeIsInMemory ? totalNumAttrMemRegs : totalNumAttrRegs ),
                       "invalid attribute register" );
        Value* elt = irb.CreateCall( readFn, irb.getInt32( reg ) );

        // The indexing starts at 0 rather than at the total attrib offset.
        const int allocaIdx = reg - attributeBaseOffset;
        RT_ASSERT( allocaIdx < attributeNumRegs );
        Value* indices[] = {irb.getInt32( 0 ), irb.getInt32( allocaIdx )};
        Value* ptr       = irb.CreateGEP( alloca, indices );
        ptr              = irb.CreateBitCast( ptr, i32Ty->getPointerTo() );
        irb.CreateStore( elt, ptr );
    }
#endif
}

// -----------------------------------------------------------------------------
static Value* getPayloadPtrFromBcpState( Value* bcpState, Type* bcpStatePtrTy, int structIdx, Instruction* insertBefore )
{
    corelib::CoreIRBuilder irb{ insertBefore };
    Value*                 bcpPtr         = irb.CreateIntToPtr( bcpState, bcpStatePtrTy, "bcpPtr" );
    Value*                 valPtr         = irb.CreateStructGEP( bcpPtr, structIdx, "bcpState" );
    Value*                 ptr            = irb.CreateLoad( valPtr, "payloadPtr" );
    return ptr;
}

void RTXCompile::loadValuesFromBcpStateAndReplaceCalls( Function* function ) const
{
    // The bound callable program state has been added as the last parameter
    // to the bound callable program.
    RT_ASSERT( function->arg_size() > 0 );
    auto   arg      = function->arg_end();
    --arg;
    Value* bcpState = &( *arg );

    Module* module = function->getParent();

    // Pass on the boundCallableProgramState to bound callables called from
    // this bound callable program.
    Function* bcpStateProxyFunc = module->getFunction( "lw.optix.bcpState.proxy" );
    for( CallInst* CI : getCallsToFunction( bcpStateProxyFunc, function ) )
    {
        corelib::CoreIRBuilder irb{CI};
        CI->replaceAllUsesWith( irb.CreateIntToPtr( bcpState, CI->getType(), "bcpState" ) );
        CI->eraseFromParent();
    }


    std::map<unsigned short, std::pair<int, std::string>> bcpStateAttributeMappings;
    Type* bcpStatePtrTy = getBoundCallableProgramStateType( module, bcpStateAttributeMappings );
    bcpStatePtrTy       = bcpStatePtrTy->getPointerTo();

    int structIdx = 0;

    std::vector<Function*> toDelete;

    if( !m_params.payloadInRegisters )
    {
        // The read payload proxy is inserted by lowerTrace and marks locations where the actual value
        // of the payload registers is needed (instead of the value stored at the payload pointer).
        Function* readPayloadProxy = module->getFunction( "proxy.lw.rt.read.payload.i32" );
        if( readPayloadProxy )
        {
            for( CallInst* CI : corelib::getCallsToFunction( readPayloadProxy, function ) )
            {
                corelib::CoreIRBuilder irb{ CI };
                Value*                 ptr     = getPayloadPtrFromBcpState( bcpState, bcpStatePtrTy, structIdx, CI );
                ptr                            = irb.CreatePtrToInt( ptr, irb.getInt64Ty() );
                std::pair<Value*, Value*> loHi = corelib::createCast_i64_to_2xi32( ptr, CI );
                ConstantInt*              offsetConst = dyn_cast<ConstantInt>( CI->getArgOperand( 0 ) );
                RT_ASSERT( offsetConst );
                uint64_t offset = offsetConst->getZExtValue();
                RT_ASSERT( offset == 0 || offset == 1 );
                if( offset == 0 )
                    CI->replaceAllUsesWith( loHi.first );
                else
                    CI->replaceAllUsesWith( loHi.second );
                CI->eraseFromParent();
            }
        }

        // Handle payload access from the bound callable program.
        for( Function& F : *module )
        {
            bool isSet        = isPayloadSet( &F );
            bool isGet        = isPayloadGet( &F );
            bool isGetAddress = GetPayloadAddressCall::isIntrinsic( &F );
            if( !isSet && !isGet && !isGetAddress )
                continue;
            for( CallInst* CI : corelib::getCallsToFunction( &F, function ) )
            {
                if( isGetAddress )
                {
                    Value*                 ptr = getPayloadPtrFromBcpState( bcpState, bcpStatePtrTy, structIdx, CI );
                    corelib::CoreIRBuilder irb{ CI };
                    ptr = irb.CreatePtrToInt( ptr, irb.getInt64Ty() );
                    CI->replaceAllUsesWith( ptr );
                }
                else
                {

                    Type* type = CI->getCalledFunction()->getReturnType();
                    if( isSet )
                        type                           = CI->getArgOperand( 2 )->getType();
                    Value*                 offsetValue = CI->getArgOperand( 1 );

                    corelib::CoreIRBuilder irb{ CI };
                    Value*                 ptr = getPayloadPtrFromBcpState( bcpState, bcpStatePtrTy, structIdx, CI );
                    ptr                        = irb.CreateGEP( ptr, offsetValue );
                    ptr                        = irb.CreatePointerCast( ptr, type->getPointerTo() );
                    if( isSet )
                    {
                        irb.CreateStore( CI->getArgOperand( 2 ), ptr );
                    }
                    else
                    {
                        Value* loadedValue = irb.CreateLoad( ptr );
                        CI->replaceAllUsesWith( loadedValue );
                        loadedValue->takeName( CI );
                    }
                }
                CI->eraseFromParent();
            }
            toDelete.push_back( &F );
        }
        ++structIdx;
    }

    // Rewrite transformations to load the matrices from the bound callable program state
    Function* worldToObjectFunc = module->getFunction( "getWorldToObjectTransformMatrix" );
    Function* objectToWorldFunc = module->getFunction( "getObjectToWorldTransformMatrix" );
    RT_ASSERT( worldToObjectFunc && objectToWorldFunc );
    if( !inlineAllCallersOfFunction( worldToObjectFunc, /*eraseUnusedFunctions*/ true ) )
        throw CompileError( RT_EXCEPTION_INFO, "Unable to inline callers of: " + worldToObjectFunc->getName().str() );
    if( !inlineAllCallersOfFunction( objectToWorldFunc, /*eraseUnusedFunctions*/ true ) )
        throw CompileError( RT_EXCEPTION_INFO, "Unable to inline callers of: " + objectToWorldFunc->getName().str() );

    for( CallInst* CI : getCallsToFunction( worldToObjectFunc, function ) )
    {
        corelib::CoreIRBuilder irb{CI};
        Value*                 bcpPtr = irb.CreateIntToPtr( bcpState, bcpStatePtrTy, "bcpPtr" );
        Value*                 valPtr = irb.CreateStructGEP( bcpPtr, structIdx, "bcpState" );
        valPtr                        = irb.CreatePointerCast( valPtr, CI->getType()->getPointerTo() );
        Value* val                    = irb.CreateLoad( valPtr, "worldToObjMat" );
        CI->replaceAllUsesWith( val );
        CI->eraseFromParent();
    }
    ++structIdx;

    for( CallInst* CI : getCallsToFunction( objectToWorldFunc, function ) )
    {
        corelib::CoreIRBuilder irb{CI};
        Value*                 bcpPtr = irb.CreateIntToPtr( bcpState, bcpStatePtrTy, "bcpPtr" );
        Value*                 valPtr = irb.CreateStructGEP( bcpPtr, structIdx, "bcpState" );
        valPtr                        = irb.CreatePointerCast( valPtr, CI->getType()->getPointerTo() );
        Value* val                    = irb.CreateLoad( valPtr, "objToWorldMat" );
        CI->replaceAllUsesWith( val );
        CI->eraseFromParent();
    }

    inlineAllCalls( worldToObjectFunc );
    inlineAllCalls( objectToWorldFunc );

    if( m_inheritedStype == ST_CLOSEST_HIT || m_inheritedStype == ST_ANY_HIT )
    {
        // Attribute handling
        Type*      i8Ty = Type::getInt8Ty( module->getContext() );
        DataLayout DL( module );

        // Rewrite attribute get value calls to read from the bcpState
        for( Function& F : *module )
        {
            if( !GetAttributeValue::isIntrinsic( &F ) )
                continue;
            StringRef varRefUniqueName;
            GetAttributeValue::parseUniqueName( &F, varRefUniqueName );
            const VariableReference* varref = m_programManager->getVariableReferenceByUniversallyUniqueName( varRefUniqueName );
            unsigned short           token = varref->getVariableToken();

            for( CallInst* CI : getCallsToFunction( &F, function ) )
            {
                ConstantInt* constantOffset = dyn_cast<ConstantInt>( CI->getArgOperand( 1 ) );

                const int offset                    = constantOffset->getZExtValue();
                const int firstReg                  = offset / 4;
                const int offsetWithinFirstRegister = offset % 4;
                Type*     attrTargetTy              = CI->getType();

                const unsigned int alignment = DL.getPrefTypeAlignment( attrTargetTy );

                corelib::CoreIRBuilder irb{CI};
                Value*                 bcpPtr = irb.CreateIntToPtr( bcpState, bcpStatePtrTy, "bcpStatePtr" );
                auto                   iter   = bcpStateAttributeMappings.find( token );
                RT_ASSERT( iter != bcpStateAttributeMappings.end() );
                Value* arrayPtr =
                    irb.CreateStructGEP( bcpPtr, iter->second.first, varref->getInputName() + "BcpStatePtr" );
                Value* indices[] = {irb.getInt32( 0 ), irb.getInt32( firstReg )};
                Value* valPtr    = irb.CreateGEP( arrayPtr, indices, varref->getInputName() + "ElementPtr" );

                // Generate struct type to avoid possible misaligned load errors
                // if the value to load has an offset in the alloca.
                SmallVector<Type*, 4> elements;
                for( int i = 0; i < offsetWithinFirstRegister; ++i )
                {
                    // Add byte entries to the struct up to the offset of the value to load.
                    elements.push_back( i8Ty );
                }
                elements.push_back( attrTargetTy );

                Type*     structType  = StructType::create( elements );
                Value*    structPtr   = irb.CreateBitCast( valPtr, structType->getPointerTo() );
                Value*    attrTypePtr = irb.CreateStructGEP( structPtr, offsetWithinFirstRegister );
                LoadInst* load        = irb.CreateLoad( attrTypePtr );
                load->setAlignment( offsetWithinFirstRegister > 0 ? 1 : alignment );
                CI->replaceAllUsesWith( load );
                load->takeName( CI );
                CI->eraseFromParent();
            }
            toDelete.push_back( &F );
        }
    }
    for( Function* func : toDelete )
    {
        RT_ASSERT( func->use_empty() );
        func->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerAttributesForCHandAH( llvm::Function* function ) const
{
    Module*      module      = function->getParent();
    LLVMContext& llvmContext = module->getContext();
    IntegerType* i32Ty       = Type::getInt32Ty( llvmContext );
    DataLayout   DL( module );

    // loop over all the decoders and find the smallest set of bits that we need for the
    // switch.  Throw an error if we run out of bits.
    int mask = -1;
    int bits = 1;
    for( ; bits <= 32; ++bits )
    {
        mask = ( 1u << bits ) - 1;
        SmallSet<int, 16> cases;
        for( unsigned int idx = 0; idx < m_attributeDecoders.size(); ++idx )
        {
            int kind = m_attributeDecoders[idx].attributeKind;
            if( !std::get<1>( cases.insert( kind & mask ) ) )
                // Didn't insert, so it was already there.  Try the next number of bits
                break;
        }
        if( cases.size() == m_attributeDecoders.size() )
            break;
    }
    if( bits > 32 )
        throw CompileError( RT_EXCEPTION_INFO,
                            "Through no fault of your own, the unique identifiers used for your intersection and "
                            "attribute programs are not actually unique in 32 bits.  Please contact LWPU's optix "
                            "team." );


    // Find first non-alloca instruction
    llvm::Instruction* EI = corelib::getFirstNonAlloca(function);
    
    // Insert code that comes before switch
    Value*                 statePtr         = function->arg_begin();
    Function*              getAttributeKind = getFunctionOrAssert( module, "RTX_getAttributeKind" );
    corelib::CoreIRBuilder irb( EI );
    Value*                 kind = irb.CreateCall( getAttributeKind );
    kind                        = irb.CreateAnd( kind, mask, "kind" );

    // Split the block and setup blocks for switch
    BasicBlock* lwrrentBB   = EI->getParent();
    BasicBlock* successorBB = lwrrentBB->splitBasicBlock( EI, "attribute_decoder_end" );
    // Remove terminator inserted by splitBasicBlock
    lwrrentBB->getTerminator()->eraseFromParent();
    BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "ilwalidKind", function, successorBB );
    corelib::CoreIRBuilder{trapBlock}.CreateUnreachable();

    DebugLoc dbgLoc = EI->getDebugLoc();
    if(!dbgLoc)
    {
        DISubprogram* subprogram = function->getSubprogram();
        if(subprogram != nullptr)
            dbgLoc = DILocation::getDistinct( module->getContext(), subprogram->getLine(), 0, subprogram, nullptr );
    }

    SwitchInst* sw = corelib::CoreIRBuilder{lwrrentBB}.CreateSwitch( kind, trapBlock, m_attributeDecoders.size() );
    std::vector<CallInst*> calls( m_attributeDecoders.size() );
    for( unsigned int idx = 0; idx < m_attributeDecoders.size(); ++idx )
    {
        int         attrKind   = m_attributeDecoders[idx].attributeKind;
        int         maskedKind = attrKind & mask;
        BasicBlock* newBlock =
            BasicBlock::Create( llvmContext, stringf( "kind_%d_masked_%d_idx_%u", attrKind, maskedKind, idx ), function, trapBlock );
        sw->addCase( ConstantInt::get( i32Ty, maskedKind ), newBlock );
        corelib::CoreIRBuilder builder( newBlock );
        calls[idx] = builder.CreateCall( m_attributeDecoders[idx].decoder, {statePtr} );
        // We need to set a debug location here. The call that we just inserted is in a newly created BB. It does
        // not have a debug loc at the moment. The call will be inlined a couple of lines further down. This inlining
        // produces invalid debug info if the original call does not have a valid debug location.
        calls[idx]->setDebugLoc( dbgLoc );
        builder.CreateBr( successorBB );
    }

    // Inline the calls and erase the independent functions
    for( unsigned int idx = 0; idx < m_attributeDecoders.size(); ++idx )
    {
        if( !dbgLoc )
        {
            // If we could not set a valid debug location for the call, we clear the debug
            // locations of the function that is about to be inlined. Otherwise the inlined instructions
            // will not have their inlinedAt field set correctly and still point to the inlined function.
            for( llvm::BasicBlock& bb : *m_attributeDecoders[idx].decoder )
            {
                for( llvm::Instruction& instr : bb )
                {
                    instr.setDebugLoc( nullptr );
                }
            }
        }
        InlineFunctionInfo IFI;
        bool               success = InlineFunction( calls[idx], IFI );
        if( !success )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( calls[idx] ), "Cannot inline attribute decoder" );
        m_attributeDecoders[idx].decoder->eraseFromParent();
    }

    // dump( module, function->getName().str(), 4, "after-decoder" );

    std::map<std::string, llvm::AllocaInst*> attributeAllocas;
    bool useUniqueName = false;
    patchAttributesToLocalAllocas( m_programManager, function, useUniqueName, &attributeAllocas );
}

void RTXCompile::lowerPayloadGetAndSet( Module* module ) const
{
    DataLayout   DL( module );
    LLVMContext& llvmContext = module->getContext();
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );
    Type*        voidTy      = Type::getVoidTy( llvmContext );

    FunctionType* readPayloadIntType  = FunctionType::get( i32Ty, i32Ty, false );
    FunctionType* writePayloadIntType = FunctionType::get( voidTy, Types{i32Ty, i32Ty}, false );
    Function*     lwvmReadPayloadInt  = insertOrCreateFunction( module, "lw.rt.read.payload.i32", readPayloadIntType );
    Function* lwvmWritePayloadInt = insertOrCreateFunction( module, "lw.rt.write.payload.i32", writePayloadIntType );

    corelib::CoreIRBuilder irBuilder( llvmContext );

    const int REGISTER_SIZE_IN_BYTES = 4;

    for( Function* F : corelib::getFunctions( module ) )
    {
        bool isSet = isPayloadSet( F );
        bool isGet = isPayloadGet( F );

        if( !isSet && !isGet )
            continue;

        for( CallInst* CI : corelib::getCallsToFunction( F ) )
        {
            Value* offsetValue = CI->getArgOperand( 1 );
            Type*  type        = isGet ? CI->getCalledFunction()->getReturnType() : CI->getArgOperand( 2 )->getType();

            irBuilder.SetInsertPoint( CI );

            if( m_params.payloadInRegisters )
            {
                // We rely on the fact that if payload promotion is enabled the payload offset is constant.
                ConstantInt* offsetConst = dyn_cast<ConstantInt>( offsetValue );
                RT_ASSERT_MSG( offsetConst, "Payload offset is not a constant" );
                uint64_t offset            = offsetConst->getZExtValue();
                uint64_t payloadRegister   = offset / REGISTER_SIZE_IN_BYTES;
                uint64_t offsetInRegister  = offset % REGISTER_SIZE_IN_BYTES;
                uint64_t operandSize       = DL.getTypeAllocSize( type );
                uint64_t operandSizeInBits = operandSize * 8;
                uint64_t offsetEnd         = offset + operandSize;

                // numPayloadPlusSizeRegisters = m_params.propagatePayloadSize ? (m_params.maxPayloadSize / 4 + 1) : numPayloadRegisters
                if( offsetEnd > m_params.maxPayloadSize )
                {
                    // If the range of the payload value is invalid, do not generate the corresponding
                    // rtcore instrinsics since this will trigger a sanity check inside rtcore. Simply
                    // use undef for uses of the get intrinsic.
                    if( isGet )
                    {
                        Value* undef = UndefValue::get( type );
                        CI->replaceAllUsesWith( undef );
                    }
                    // Nothing to do for set intrinsics. Both kinds of intrinsics will be removed at
                    // the end of the loop.
                }
                else
                {
                    // Create an alloca in the entry block
                    corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( CI->getParent()->getParent() )};
                    AllocaInst*            alloca = irb.CreateAlloca( type, nullptr, "payload" );
                    Value*                 i8ptr  = irb.CreateBitCast( alloca, irb.getInt8PtrTy() );

                    // In case of a set intrinsic store the value in the alloca
                    if( isSet )
                    {
                        Value* value = CI->getArgOperand( 2 );
                        corelib::CoreIRBuilder{CI}.CreateStore( value, alloca );
                    }

                    Value* lwrrentRegister = irBuilder.getInt32( payloadRegister );
                    for( uint64_t regOffset = 0; regOffset < operandSize; regOffset += REGISTER_SIZE_IN_BYTES )
                    {
                        Value* gep = irBuilder.CreateInBoundsGEP( i8ptr, irBuilder.getInt32( regOffset ) );

                        if( isSet )
                        {
                            gep        = irBuilder.CreateBitCast( gep, i32Ty->getPointerTo(), "" );
                            Value* elt = irBuilder.CreateLoad( gep );
                            irBuilder.CreateCall( lwvmWritePayloadInt, { lwrrentRegister, elt } );
                        }
                        else
                        {
                            Value* elt = irBuilder.CreateCall( lwvmReadPayloadInt, lwrrentRegister );
                            if( operandSize < REGISTER_SIZE_IN_BYTES )
                            {
                                int numberOfElements = REGISTER_SIZE_IN_BYTES / operandSize;
                                RT_ASSERT_MSG( numberOfElements == 2 || numberOfElements == 4,
                                               "Invalid sub element size" );
                                RT_ASSERT_MSG( offsetInRegister % operandSize == 0, "Unsupported payload access" );
                                Type*  elementType = Type::getIntNTy( llvmContext, operandSizeInBits );
                                Type*  vectorType  = VectorType::get( elementType, numberOfElements );
                                Value* casted      = irBuilder.CreateBitCast( elt, vectorType );
                                Value* index       = irBuilder.getInt32( offsetInRegister / operandSize );
                                elt                = irBuilder.CreateExtractElement( casted, index );
                                gep = irBuilder.CreateInBoundsGEP( gep, irBuilder.getInt32( offsetInRegister ) );
                                gep = irBuilder.CreateBitCast( gep, elementType->getPointerTo() );
                            }
                            else
                            {
                                gep = irBuilder.CreateBitCast( gep, i32Ty->getPointerTo(), "" );
                            }
                            irBuilder.CreateStore( elt, gep );
                        }

                        lwrrentRegister = irBuilder.CreateAdd( lwrrentRegister, irBuilder.getInt32( 1 ), "" );
                    }

                    // In case of a get intrinsic read the value from the alloca
                    if( isGet )
                    {
                        Value* loadedValue = irBuilder.CreateLoad( alloca );
                        CI->replaceAllUsesWith( loadedValue );
                        loadedValue->takeName( CI );
                    }
                }
            }
            else
            {
                bool handled = false;

                ConstantInt* offsetConst = dyn_cast<ConstantInt>( offsetValue );
                if( offsetConst )
                {
                    uint64_t offset      = offsetConst->getZExtValue();
                    uint64_t operandSize = DL.getTypeAllocSize( type );
                    uint64_t offsetEnd   = offset + operandSize;

                    if( offsetEnd > m_params.maxPayloadSize )
                    {
                        // If the range of the payload value is invalid, skip the memory accesses (and simply
                        // use undef for uses of the get intrinsic).
                        if( isGet )
                        {
                            Value* undef = UndefValue::get( type );
                            CI->replaceAllUsesWith( undef );
                        }
                        // Nothing to do for set intrinsics. Both kinds of intrinsics will be removed at the
                        // end of the loop.
                        handled = true;
                    }
                }

                if( !handled )
                {
                    // The payload has not been promoted to register, it is still in local memory.
                    Value* payloadPtrLo   = irBuilder.CreateCall( lwvmReadPayloadInt, irBuilder.getInt32( 0 ) );
                    Value* payloadPtrHi   = irBuilder.CreateCall( lwvmReadPayloadInt, irBuilder.getInt32( 1 ) );
                    Value* payloadPointer = createCast_2xi32_to_i64( payloadPtrLo, payloadPtrHi, CI );
                    Value* ptr            = irBuilder.CreateIntToPtr( payloadPointer, irBuilder.getInt8PtrTy() );
                    ptr                   = irBuilder.CreateGEP( ptr, offsetValue );
                    ptr                   = irBuilder.CreatePointerCast( ptr, type->getPointerTo() );

                    if( isSet )
                    {
                        irBuilder.CreateStore( CI->getArgOperand( 2 ), ptr );
                    }
                    else
                    {
                        Value* loadedValue = irBuilder.CreateLoad( ptr );
                        CI->replaceAllUsesWith( loadedValue );
                        loadedValue->takeName( CI );
                    }
                }
            }

            CI->eraseFromParent();
        }

        RT_ASSERT( F->use_empty() );
        F->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerExceptionDetails( Module* module ) const
{
    Function* optixi_getExceptionDetail   = module->getFunction( "optixi_getExceptionDetail" );
    Function* optixi_getExceptionDetail64 = module->getFunction( "optixi_getExceptionDetail64" );
    Function* rtcore_getExceptionDetail   = getFunctionOrAssert( module, "lw.rt.read.exception.detail" );

    // Colwert optixi_getExceptionDetail64(0..6) and optixi_getExceptionDetail(0..8) into rtcore_getExceptionDetail(0..22)
    //
    // For 0 <= i <= 6, optixi_getExceptionDetail64[i] is mapped to rtcore_getExceptionDetail[2*i]
    //                                                          and rtcore_getExceptionDetail[2*i+1].
    // For 0 <= i <= 8, optixi_getExceptionDetail[i]   is mapped to rtcore_getExceptionDetail[14+i].
    //
    // Keep in sync with RTXExceptionInstrumenter::createException().

    for( CallInst* CI : corelib::getCallsToFunction( optixi_getExceptionDetail64 ) )
    {
        corelib::CoreIRBuilder builder( CI );
        int                    optixIndex = corelib::getConstantValueOrAssertSigned( CI->getArgOperand( 1 ) );
        Value* detailLo = builder.CreateCall( rtcore_getExceptionDetail, builder.getInt32( 2 * optixIndex ) );
        Value* detailHi = builder.CreateCall( rtcore_getExceptionDetail, builder.getInt32( 2 * optixIndex + 1 ) );
        Value* detail   = corelib::createCast_2xi32_to_i64( detailLo, detailHi, CI );
        CI->replaceAllUsesWith( detail );
        CI->eraseFromParent();
    }

    for( CallInst* CI : corelib::getCallsToFunction( optixi_getExceptionDetail ) )
    {
        corelib::CoreIRBuilder builder( CI );
        int                    optixIndex = corelib::getConstantValueOrAssertSigned( CI->getArgOperand( 1 ) );
        Value* detail = builder.CreateCall( rtcore_getExceptionDetail, builder.getInt32( 14 + optixIndex ) );
        CI->replaceAllUsesWith( detail );
        CI->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
// Replace optixi_getLwrrentRay with either the world or object space ray.  See
// OptiX_Programming_Guide section 4.1.6 "Program Variable Transformation" for additional
// information.
void RTXCompile::lowerGetLwrrentRay( Module* module ) const
{
    Function* getLwrrentRay;
    switch( m_stype )
    {
        case ST_CLOSEST_HIT:
        case ST_MISS:
            getLwrrentRay = module->getFunction( "RTX_getWorldSpaceRay" );
            break;
        case ST_ANY_HIT:
        case ST_INTERSECTION:
        case ST_NODE_VISIT:
            getLwrrentRay = module->getFunction( "RTX_getObjectSpaceRay" );
            break;
        default:
            getLwrrentRay = nullptr;
            break;
    }

    for( CallInst* CI : corelib::getCallsToFunction( module->getFunction( "optixi_getLwrrentRay" ) ) )
    {
        if( !getLwrrentRay )
            throw AssertionFailure( RT_EXCEPTION_INFO, "Invalid semantic type (" + semanticTypeToString( m_stype )
                                                           + ") accesses current ray" );
        CallInst* ray = corelib::CoreIRBuilder{CI}.CreateCall( getLwrrentRay );
        ray->takeName( CI );
        CI->replaceAllUsesWith( ray );
        CI->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerGetAttributeData( llvm::Function* function ) const
{

    LLVMContext& llvmContext = function->getContext();
    Module*      module      = function->getParent();

    std::vector<CallInst*> calls;
    for( Function& F : *module )
    {
        if( !F.isDeclaration() )
            continue;

        if( F.getName().startswith( "optixi_getAttributeData." ) )
        {
            const std::vector<CallInst*>& c = getCallsToFunction( &F, function );
            calls.insert( calls.end(), c.begin(), c.end() );
        }
    }

    if( calls.empty() )
        return;

    IntegerType*  i32Ty     = Type::getInt32Ty( llvmContext );
    FunctionType* readType  = FunctionType::get( i32Ty, i32Ty, false );
    Constant*     readRegFn = module->getOrInsertFunction( "lw.rt.read.register.attribute.i32", readType );
    Constant*     readMemFn = module->getOrInsertFunction( "lw.rt.read.memory.attribute.i32", readType );

    for( CallInst* call : calls )
    {
        Type* attributeDataTy         = call->getType();
        int   numAttribute32bitValues = getNumRequiredRegisters( attributeDataTy );

        int numRegister = std::min( m_params.maxAttributeRegisterCount, numAttribute32bitValues );

        std::vector<Value*>    attributeDataVector( numAttribute32bitValues );
        corelib::CoreIRBuilder irb{call};
        for( unsigned int i = 0, e = numRegister; i < e; ++i )
        {
            attributeDataVector[i] = irb.CreateCall( readRegFn, {ConstantInt::get( i32Ty, i )} );
        }
        for( unsigned int i = numRegister, e = numAttribute32bitValues; i < e; ++i )
        {
            attributeDataVector[i] = irb.CreateCall( readMemFn, {ConstantInt::get( i32Ty, i - numRegister )} );
        }

        Value* attributeData;
        if( attributeDataVector.empty() )
            attributeData = UndefValue::get( attributeDataTy );
        else
            attributeData = unflattenAggregateForRTCore( attributeDataTy, attributeDataVector, /*startIndex=*/0, call );

        call->replaceAllUsesWith( attributeData );
        attributeData->takeName( call );
        call->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::lowerIsPotentialIntersection( llvm::Function* function ) const
{
    Module* module = function->getParent();

    std::vector<CallInst*> calls;
    for( Function& F : *module )
    {
        if( !F.isDeclaration() )
            continue;

        if( IsPotentialIntersection::isIntrinsic( &F ) )
        {
            const std::vector<CallInst*>& c = getCallsToFunction( &F, function );
            calls.insert( calls.end(), c.begin(), c.end() );
        }
    }

    if( calls.empty() )
        return;

    Function* rtcReadRayTmin = getFunctionOrAssert( module, "lw.rt.read.ray.tmin" );
    Function* rtcReadRayTmax = getFunctionOrAssert( module, "lw.rt.read.ray.tmax" );

    for( CallInst* call : calls )
    {
        IsPotentialIntersection* rtpi = dyn_cast<IsPotentialIntersection>( call );
        corelib::CoreIRBuilder   irb{rtpi};

        // Replace rtPI by a check if hitT is within the current ray's
        // tmin and tmax (inclusive).
        Value*    hitT    = rtpi->getHitT();
        CallInst* tmin    = irb.CreateCall( rtcReadRayTmin, llvm::None, "tmin" );
        CallInst* tmax    = irb.CreateCall( rtcReadRayTmax, llvm::None, "tmax" );
        Value*    resA    = irb.CreateFCmp( FCmpInst::FCMP_OLE, tmin, hitT, "hit.greater.tmin" );
        Value*    resB    = irb.CreateFCmp( FCmpInst::FCMP_OLE, hitT, tmax, "hit.less.tmax" );
        Value*    hitOkay = irb.CreateBinOp( Instruction::And, resA, resB, "hitT.in.range" );

        call->replaceAllUsesWith( hitOkay );
        hitOkay->takeName( call );
        call->eraseFromParent();
    }
}


// -----------------------------------------------------------------------------
void RTXCompile::lowerReportFullIntersection( llvm::Function* function ) const
{
    LLVMContext& llvmContext = function->getContext();
    Module*      module      = function->getParent();

    std::vector<CallInst*> calls;
    for( Function& F : *module )
    {
        if( !F.isDeclaration() )
            continue;

        if( ReportFullIntersection::isIntrinsic( &F ) )
        {
            const std::vector<CallInst*>& c = getCallsToFunction( &F, function );
            calls.insert( calls.end(), c.begin(), c.end() );
        }
    }

    if( calls.empty() )
        return;

    Type*     i32Ty = Type::getInt32Ty( llvmContext );
    Type*     i1Ty  = Type::getInt1Ty( llvmContext );
    Constant* one   = ConstantInt::get( i1Ty, 1 );

    for( CallInst* call : calls )
    {
        ReportFullIntersection* rfi = dyn_cast<ReportFullIntersection>( call );
        RT_ASSERT( rfi );

        std::vector<llvm::Value*> attributeData = corelib::flattenAggregateTo32BitValuesForRTCore( rfi->getAttributeData(), rfi );

        corelib::CoreIRBuilder irb{rfi};

        // flattenAggregateTo32BitValuesForRTCore doesn't ensure i32 types, so do it now.
        for( int i = 0, e = (int)attributeData.size(); i < e; ++i )
        {
            if( attributeData[i]->getType() != i32Ty )
                attributeData[i] = irb.CreateBitCast( attributeData[i], i32Ty );
        }

        // Partition the attributes into the register and memory groups based on maxAttributeRegisterCount
        int   numRegister               = std::min( m_params.maxAttributeRegisterCount, (int)attributeData.size() );
        int   numMemory                 = (int)attributeData.size() - numRegister;
        Type* registerAttributeStructTy = ArrayType::get( i32Ty, numRegister );
        Type* memoryAttributeStructTy   = ArrayType::get( i32Ty, numMemory );

        Value* registerAttributeStruct = UndefValue::get( registerAttributeStructTy );
        for( unsigned int i = 0, e = numRegister; i < e; ++i )
        {
            registerAttributeStruct = irb.CreateInsertValue( registerAttributeStruct, attributeData[i], i );
        }
        Value* memoryAttributeStruct = UndefValue::get( memoryAttributeStructTy );
        for( unsigned int i = numRegister, e = attributeData.size(); i < e; ++i )
        {
            memoryAttributeStruct = irb.CreateInsertValue( memoryAttributeStruct, attributeData[i], i - numRegister );
        }

        Function* getGeometryInstanceSkip = getFunctionOrAssert( module, "_ZN4cort10getSBTSkipEj" );
        CallInst* sbtSkip = irb.CreateCall( getGeometryInstanceSkip, rfi->getMaterialIndex(), "sbtSkip" );

        // Create type mangled report intersection function declaration
        std::string reportIntersectionName = "lw.rt.report.intersection." + getTypeName( registerAttributeStructTy )
                                             + "." + getTypeName( memoryAttributeStructTy );
        Function* reportIntersectionFunction = module->getFunction( reportIntersectionName );
        if( !reportIntersectionFunction )
        {
            Type*         floatTy    = Type::getFloatTy( llvmContext );
            Type*         i8Ty       = Type::getInt8Ty( llvmContext );
            Type*         argTypes[] = {floatTy, i8Ty, i32Ty, registerAttributeStructTy, memoryAttributeStructTy, i1Ty};
            FunctionType* functionType = FunctionType::get( i1Ty, argTypes, false );
            reportIntersectionFunction =
                Function::Create( functionType, GlobalValue::ExternalLinkage, reportIntersectionName, module );
            reportIntersectionFunction->addFnAttr( Attribute::NoUnwind );
        }

        Constant* checkHitTInterval = one;
        Value*    args[]            = {rfi->getHitT(),          rfi->getHitKind(),     sbtSkip,
                         registerAttributeStruct, memoryAttributeStruct, checkHitTInterval};
        CallInst* rtcRI = irb.CreateCall( reportIntersectionFunction, args );

        rfi->replaceAllUsesWith( rtcRI );
        rfi->eraseFromParent();
    }
}


// -----------------------------------------------------------------------------
static void defineVideoCalls( Module* mod )
{
    for( Function& func : *mod )
    {
        if( !func.getName().startswith( "optix.ptx.video." ) )
            continue;

        // Format video instruction inline string
        bool                     modAB_present = false;
        bool                     modB_present  = false;
        std::vector<std::string> selectors;
        std::string              fName( func.getName() );

        fName.erase( fName.begin(), fName.begin() + sizeof( "optix.ptx.video." ) - 1 /* '\0' terminator */ );

        size_t occ = fName.find( ".negAB" );
        if( occ != std::string::npos )
        {
            fName.erase( occ, sizeof( ".negAB" ) - 1 );
            modAB_present = true;
        }
        occ = fName.find( ".negB" );
        if( occ != std::string::npos )
        {
            fName.erase( occ, sizeof( ".negB" ) - 1 );
            modB_present = true;
        }

        assert( !( modAB_present && modB_present ) && "Invalid video function declaration - double MAD modifiers" );

        occ                      = fName.find( ".selsec" );
        std::string::iterator it = fName.erase( fName.begin() + occ, fName.begin() + occ + sizeof( ".selsec" ) - 1 );

        assert( occ != std::string::npos && "Selectors not in place, invalid video instruction" );

        size_t      selectorsStart = std::distance( fName.begin(), it );
        std::string temp           = fName.substr( selectorsStart );
        size_t      end;
        occ = 0;
        while( ( end = temp.find( '.', occ + 1 ) ) != std::string::npos )
        {
            selectors.push_back( temp.substr( occ, end - occ ) );
            occ = end;
        }
        selectors.push_back( temp.substr( occ ) );

        assert( ( selectors.size() == 3 || selectors.size() == 4 )
                && "Invalid number of parameters in video instruction" );

        fName.erase( selectorsStart );

        std::stringstream inlineStr, constraints;
        inlineStr << fName << " $0";
        if( selectors[0] != ".noSel" )  // Mask
            inlineStr << selectors[0];
        constraints << "=r";

        inlineStr << ", " << ( modAB_present ? "-" : "" ) << "$1";
        if( selectors[1] != ".noSel" )  // a
            inlineStr << selectors[1];
        constraints << ",r";

        inlineStr << ", "
                  << "$2";
        if( selectors[2] != ".noSel" )  // b
            inlineStr << selectors[2];
        constraints << ",r";

        if( func.arg_size() > 2 )
        {
            inlineStr << ", " << ( modB_present ? "-" : "" ) << "$3";  // (optional) c
            constraints << ",r";
        }
        inlineStr << ";";


        InlineAsm* inlineCall = InlineAsm::get( func.getFunctionType(), inlineStr.str(), constraints.str(), true );

        BasicBlock*            BB = BasicBlock::Create( mod->getContext(), "entry", &func );
        corelib::CoreIRBuilder irb( BB );

        func.addFnAttr( Attribute::AlwaysInline );
        func.setLinkage( GlobalValue::LinkOnceAnyLinkage );

        std::vector<Value*> arguments;
        for( Function::arg_iterator it = func.arg_begin(), end = func.arg_end(); it != end; ++it )
            arguments.push_back( &*it );
        Value* ret = irb.CreateCall( inlineCall, arguments );

        irb.CreateRet( ret );
    }
}

// -----------------------------------------------------------------------------
static void rewriteTable( Module*            module,
                          bool               tableInConstMemory,
                          const std::string& variableName,
                          const std::string& runtimeFunctionName,
                          const std::string& constFunctionName )
{
    GlobalVariable* G = module->getGlobalVariable( variableName, true );
    RT_ASSERT_MSG( G != nullptr, "Couldn't find " + variableName + " in module" );
    RT_ASSERT_MSG( G->getType()->getPointerElementType()->isArrayTy()
                       && G->getType()->getPointerElementType()->getArrayElementType()->isIntegerTy( 8 ),
                   "buffer table '" + variableName + "'' has invalid type" );
    if( tableInConstMemory )
    {
        replaceFunctionWithFunction( module, runtimeFunctionName, constFunctionName );
    }
    else
    {
        // We don't need the global variable anymore, and nobody removes it automatically
        // (setting it to InternalLinkage causes mis-compiles).  If we don't actually remove
        // it, lwvm to ptx will still allocate some space for it.  We need to remove the
        // using functions first, though.
        SmallSet<Function*, 10> toDelete;
        getContainingFunctions( G, toDelete );
        for( const auto& I : toDelete )
            I->eraseFromParent();
        G->eraseFromParent();
    }
}

// -----------------------------------------------------------------------------
void RTXCompile::rewriteTableAccessesToUseConstMemory( Module* module, const ConstantMemAllocationFlags& constMemAllocFlags ) const
{
    rewriteTable( module, constMemAllocFlags.objectRecordInConstMemory, "const_ObjectRecord",
                  "_ZN4cort22Global_getObjectRecordEPNS_14CanonicalStateEj", "Megakernel_getObjectRecordFromConst" );

    rewriteTable( module, constMemAllocFlags.bufferTableInConstMemory, "const_BufferHeaderTable",
                  "_ZN4cort22Global_getBufferHeaderEPNS_14CanonicalStateEj", "Megakernel_getBufferHeaderFromConst" );

    rewriteTable( module, constMemAllocFlags.programTableInConstMemory, "const_ProgramHeaderTable",
                  "_ZN4cort23Global_getProgramHeaderEPNS_14CanonicalStateEj", "Megakernel_getProgramHeaderFromConst" );

    rewriteTable( module, constMemAllocFlags.textureTableInConstMemory, "const_TextureHeaderTable",
                  "_ZN4cort30Global_getTextureSamplerHeaderEPNS_14CanonicalStateEj",
                  "Megakernel_getTextureHeaderFromConst" );
}

void RTXCompile::replaceGlobalDeviceCount( Module* module, int deviceCount ) const
{
    Function*    countFunc     = getFunctionOrAssert( module, "_ZN4cort21Global_getDeviceCountEPNS_14CanonicalStateE" );
    LLVMContext& context       = module->getContext();
    Type*        i16Ty         = Type::getInt16Ty( context );
    Value*       devCountValue = ConstantInt::get( i16Ty, deviceCount );
    for( CallInst* call : getCallsToFunction( countFunc ) )
    {
        call->replaceAllUsesWith( devCountValue );
        call->eraseFromParent();
    }
}

void RTXCompile::addReturnsForExceptionThrow( llvm::Module* module, llvm::Function* function ) const
{
    Function* throwFunction = getFunctionOrAssert( module, "lw.rt.throw.exception" );

    inlineAllCallersOfFunction( throwFunction, true );

    std::vector<CallInst*> calls = getCallsToFunction( throwFunction );
    if( calls.empty() )
        return;

    // Some attributes like nounwind cause lwvm70 to optimize away the call to
    // lw.rt.throw.exception which typically results in Undesired Behavior (TM). By
    // forcing the attributes on the call site to match exactly what we need we can avoid
    // this. Clang seems to put nounwind on various places (we think this might be related
    // to 'extern "C"'), and I don't really want to tease that ball of tangled yarn apart,
    // so here we are.
    LLVMContext& context = module->getContext();
    using AttributeContainer = llvm::AttributeList;

    for( CallInst* call : calls )
    {
        AttributeContainer attrs = call->getAttributes();
        attrs = attrs.removeAttribute( context, AttributeContainer::FunctionIndex, llvm::Attribute::NoUnwind );
        attrs = attrs.addAttribute( context, AttributeContainer::FunctionIndex, llvm::Attribute::ReadOnly );
        attrs = attrs.addAttribute( context, AttributeContainer::FunctionIndex, llvm::Attribute::NoReturn );
        call->setAttributes( attrs );

        // Make sure the caller is also marked as being able to throw (remove
        // nounwind). In lwvm70, at least, instcombine will mark calls with NoUnwind if
        // the caller is marked NoUnwind. This eventually leads instcombine to remove the
        // call since it thinks there's no side effect.
        call->getParent()->getParent()->removeFnAttr( llvm::Attribute::NoUnwind );

        Instruction* insertBefore = getInstructionAfter( call );

        BasicBlock* callBlock = insertBefore->getParent();
        callBlock->splitBasicBlock( insertBefore, "afterException" );
        callBlock->getTerminator()->eraseFromParent();

        corelib::CoreIRBuilder irb( callBlock );
        Type*                  returnTy = callBlock->getParent()->getReturnType();
        if( returnTy->isVoidTy() )
            irb.CreateRetVoid();
        else
            irb.CreateRet( UndefValue::get( returnTy ) );
    }
}

// Helper functions.
// -----------------------------------------------------------------------------
void RTXCompile::dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix ) const
{
    addMissingLineInfoAndDump( module, k_saveLLVM.get(), suffix, dumpId, m_launchCounterForDebugging,
                               functionName + "-RTXCompile" );
}

bool RTXCompile::ConstantMemAllocationFlags::operator!=( const ConstantMemAllocationFlags& other ) const
{
    if( objectRecordInConstMemory != other.objectRecordInConstMemory )
        return true;
    if( bufferTableInConstMemory != other.bufferTableInConstMemory )
        return true;
    if( programTableInConstMemory != other.programTableInConstMemory )
        return true;
    if( textureTableInConstMemory != other.textureTableInConstMemory )
        return true;
    return false;
}

bool RTXCompile::ConstantMemAllocationFlags::operator<( const ConstantMemAllocationFlags& other ) const
{
    if( objectRecordInConstMemory != other.objectRecordInConstMemory )
        return objectRecordInConstMemory < other.objectRecordInConstMemory;
    if( bufferTableInConstMemory != other.bufferTableInConstMemory )
        return bufferTableInConstMemory < other.bufferTableInConstMemory;
    if( programTableInConstMemory != other.programTableInConstMemory )
        return programTableInConstMemory < other.programTableInConstMemory;
    if( textureTableInConstMemory != other.textureTableInConstMemory )
        return textureTableInConstMemory < other.textureTableInConstMemory;
    return false;
}

bool RTXCompile::CompileParams::operator!=( const CompileParams& other ) const
{
    if( payloadInRegisters != other.payloadInRegisters )
        return true;
    if( constMemAllocFlags != other.constMemAllocFlags )
        return true;
    if( numCallableParamRegisters != other.numCallableParamRegisters )
        return true;
    if( forceInlineUserFunctions != other.forceInlineUserFunctions )
        return true;
    if( addLimitIndicesCheck != other.addLimitIndicesCheck )
        return true;
    if( exceptionFlags != other.exceptionFlags )
        return true;
    if( maxPayloadSize != other.maxPayloadSize )
        return true;
    if( propagatePayloadSize != other.propagatePayloadSize )
        return true;
    if( maxAttributeRegisterCount != other.maxAttributeRegisterCount )
        return true;
    return false;
}

bool RTXCompile::CompileParams::operator<( const CompileParams& other ) const
{
    if( payloadInRegisters != other.payloadInRegisters )
        return payloadInRegisters < other.payloadInRegisters;
    if( constMemAllocFlags != other.constMemAllocFlags )
        return constMemAllocFlags < other.constMemAllocFlags;
    if( numCallableParamRegisters != other.numCallableParamRegisters )
        return numCallableParamRegisters < other.numCallableParamRegisters;
    if( forceInlineUserFunctions != other.forceInlineUserFunctions )
        return forceInlineUserFunctions < other.forceInlineUserFunctions;
    if( addLimitIndicesCheck != other.addLimitIndicesCheck )
        return addLimitIndicesCheck < other.addLimitIndicesCheck;
    if( exceptionFlags != other.exceptionFlags )
        return exceptionFlags < other.exceptionFlags;
    if( maxPayloadSize != other.maxPayloadSize )
        return maxPayloadSize < other.maxPayloadSize;
    if( propagatePayloadSize != other.propagatePayloadSize )
        return propagatePayloadSize < other.propagatePayloadSize;
    if( maxAttributeRegisterCount != other.maxAttributeRegisterCount )
        return maxAttributeRegisterCount < other.maxAttributeRegisterCount;
    return false;
}

bool RTXCompile::SizeOffsetPair::operator==( const SizeOffsetPair& other ) const
{
    return size == other.size && offset == other.offset && memory == other.memory;
}

bool RTXCompile::SizeOffsetPair::operator<( const SizeOffsetPair& other ) const
{
    if( size != other.size )
        return size < other.size;
    if( offset != other.offset )
        return size < other.offset;
    return memory < other.memory;
}

void optix::readOrWrite( PersistentStream* stream, RTXCompile::ConstantMemAllocationFlags* flags, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "RTXCompile::ConstantMemAllocationFlags" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &flags->objectRecordInConstMemory, "objectRecordInConstMemory" );
    readOrWrite( stream, &flags->bufferTableInConstMemory, "bufferTableInConstMemory" );
    readOrWrite( stream, &flags->programTableInConstMemory, "programTableInConstMemory" );
    readOrWrite( stream, &flags->textureTableInConstMemory, "textureTableInConstMemory" );
}

void optix::readOrWrite( PersistentStream* stream, RTXCompile::CompileParams* params, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "RTXCompile::CompileParams" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &params->payloadInRegisters, "payloadInRegisters" );
    readOrWrite( stream, &params->constMemAllocFlags, "constMemAllocFlags" );
    readOrWrite( stream, &params->numCallableParamRegisters, "numCallableParamRegisters" );
    readOrWrite( stream, &params->forceInlineUserFunctions, "forceInlineUserFunctions" );
    readOrWrite( stream, &params->addLimitIndicesCheck, "addLimitIndicesCheck" );
    readOrWrite( stream, &params->exceptionFlags, "exceptionFlags" );
    readOrWrite( stream, &params->maxPayloadSize, "maxPayloadSize" );
    readOrWrite( stream, &params->propagatePayloadSize, "propagatePayloadSize" );
    readOrWrite( stream, &params->maxAttributeRegisterCount, "maxAttributeRegisterCount" );
}

void optix::readOrWrite( PersistentStream* stream, RTXCompile::SizeOffsetPair* atts, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "SizeOffsetPair" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &atts->size, "size" );
    readOrWrite( stream, &atts->offset, "offset" );
    readOrWrite( stream, &atts->memory, "memory" );
}
