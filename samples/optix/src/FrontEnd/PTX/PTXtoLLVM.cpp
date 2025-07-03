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

#include <FrontEnd/PTX/PTXtoLLVM.h>

#include <FrontEnd/Canonical/LineInfo.h>
#include <FrontEnd/Canonical/Mangle.h>
#include <FrontEnd/PTX/PTXFrontEnd.h>
#include <FrontEnd/PTX/libPTXFrontEnd_bin.h>
#include <Util/CodeRange.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/compiler/PeepholePass.h>
#include <corelib/misc/String.h>
#include <prodlib/compiler/ModuleCache.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/exceptions/IlwalidSource.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Knobs.h>

#include <llvm/Analysis/Passes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>

#include <atomic>

#include "CorrectVprintfTypePass.h"

using namespace corelib;
using namespace prodlib;
using namespace optix;
using namespace llvm;

namespace {
Knob<std::string> k_saveLLVM( RT_DSTRING( "rtx.saveLLVM" ), "", RT_DSTRING( "Save LLVM stages during compilation" ) );
}

// Defined in lwvm\llvm\projects\lwcompiler\lib\lwvm\lwopt\lwopt.h
llvm::Module* optimization( int argc, char* argv[], llvm::Module* orgmodule, char** log );

// clang-format off
Knob<int>               k_ptx2llvm_optLevel( RT_DSTRING("ptx2llvm.optLevel"), 0, RT_DSTRING( "Default optimization level for ptx2llvm" ) );
Knob<std::string>       k_libPTXFrontEndFilename( RT_DSTRING("ptx2llvm.libPTXFrontEnd"), "", RT_DSTRING( "Read frontend library from a file" ) );
Knob<std::string>       k_saveInputPtx( RT_DSTRING( "ptx2llvm.saveInputPtx" ), "", RT_DSTRING( "Save the input PTX into a file" ) );
Knob<bool>              k_ptx2llvm_synthesizeAllocaLifetimes( RT_DSTRING("ptx2llvm.synthesizeAllocaLifetimes"), false, RT_DSTRING( "Determine minimum lifetime of memory reserved by alloca instructions and place corresponding markers if none existed before." ) );
Knob<std::string>       k_ptx2llvm_removeExistingAllocaLifetimes( RT_DSTRING("ptx2llvm.removeExistingAllocaLifetimes"), "safe", RT_DSTRING( "Remove existing lifetime intrinsics before synthesizing new ones. ['all' = remove all, 'safe' = if they cover the entire function (.start in entry block, .end before returns), 'none' to disable." ) );
HiddenPublicKnob<bool>  k_optimizeLLVMInput( RT_PUBLIC_DSTRING("ptx2llvm.optimizeLLVMInput"), true, RT_PUBLIC_DSTRING( "Optimize incoming LLVM input." ) );
// clang-format on

PTXtoLLVM::PTXtoLLVM( llvm::LLVMContext& llvmContext, const llvm::DataLayout* dataLayout )
    : m_llvmContext( llvmContext )
    , m_dataLayout( dataLayout )
    , m_targetArch( 0 )
{
}

PTXtoLLVM::~PTXtoLLVM()
{
}

static void populateFunctionPassManager( unsigned int optLevel, legacy::FunctionPassManager& FPM )
{
    if( optLevel == 0 )
        return;

    // addInitialAliasAnalysisPasses
    // These have shown to be potential bottlenecks, so we disable them at -O1.
    if( optLevel > 1 )
    {
        FPM.add( createTypeBasedAAWrapperPass() );
        FPM.add( createBasicAAWrapperPass() );
    }
    else
    {
        // TODO(Kincaid): Is there a replacement for this pass?
        // FPM.add( createNoAAPass() );
    }

    FPM.add( createCFGSimplificationPass() );

    FPM.add( createSROAPass() );
    FPM.add( createEarlyCSEPass() );
    FPM.add( createLowerExpectIntrinsicPass() );
}

//------------------------------------------------------------------------------
// This is derived from PassManagerBuilder at the time of CL 20171847.
// PassManagerBuilder pmBuilder;
// pmBuilder.OptLevel = optLevel;
// pmBuilder.Inliner = useAlwaysInliner ?
//       createAlwaysInlinerPass() :
//       createFunctionInliningPass( 1000 );
// pmBuilder.populateModulePassManager( PM );
static void populateModulePassManager( unsigned int optLevel, bool useAlwaysInliner, legacy::PassManager& PM )
{
    // Inliner checks OptArchFeatures::supportsRelwrsion(), which seems to be uninitialized if we
    // don't construct OptArchFeatures ourselves.
    PM.add( new OptArchFeatures( "sm_50", false ) );

    Pass* inliner = useAlwaysInliner ? createAlwaysInlinerLegacyPass() : createFunctionInliningPass( 1000 );

    // This is the minimum set of optimizations known to produce the desired output
    // lwstomized for PTXtoLLVM.
    if( optLevel == 0 )
    {
        PM.add( inliner );
        PM.add( createEarlyCSEPass() );  // Catch trivial redundancies
        PM.add( createPeepholePass() );  // Seems to translate local address math to GEP
        PM.add( createInstructionCombiningPass() );  // Clean up after everything.
        PM.add( createEarlyCSEPass() );              // Clean up after instcombine (see Bug 3315634)
        PM.add( createCorrectVprintfTypePass() );    // Correct vprintf type from ptx parsing
        PM.add( createCFGSimplificationPass() );
        return;
    }

// Set up some properties.
// These are basically default values except for force-disabled
// vectorization to keep compile times down (we expect this to not
// find many vectorizable regions anyway).
#define sizeLevel 0
#define disableUnrollLoops 0
#define doBBVectorize 0
#define doSLPVectorize 0
#define loopVectorize 0
#define lateVectorize 1
#define rerollLoops 1
#if doBBVectorize
    const bool useGVNAfterVectorization = false;
#endif

    // addInitialAliasAnalysisPasses
    // These have shown to be potential bottlenecks, so we disable them at -O1.
    if( optLevel > 1 )
    {
        PM.add( createTypeBasedAAWrapperPass() );
        PM.add( createBasicAAWrapperPass() );
    }
    else
    {
        //TODO(Kincaid): Is there a replacement for this pass in lwvm70? Is it needed?
        // PM.add( createNoAAPass() );
    }

    PM.add( createCFGSimplificationPass() );

    // Skipping SROA since the incoming PTX doesn't use structs.

    if( optLevel > 1 )
        PM.add( createEarlyCSEPass() );

    // We don't need this pass, because we don't use llvm.expect
    //PM.add( createLowerExpectIntrinsicPass() );

    PM.add( createGlobalOptimizerPass() );  // Optimize out global vars
    PM.add( createIPSCCPPass() );           // IP SCCP

    PM.add( createDeadArgEliminationPass() );  // Dead argument elimination
    PM.add( createPeepholePass() );
    PM.add( createInstructionCombiningPass() );  // Clean up after IPCP & DAE
    PM.add( createCFGSimplificationPass() );     // Clean up after IPCP & DAE
    PM.add( createPruneEHPass() );               // Remove dead EH info

    PM.add( inliner );
    //PM.add( createGlobalDCEPass() );  // This will be done up after cloning the canonical program, so skip it for now

    PM.add( createPostOrderFunctionAttrsLegacyPass() );  // Set readonly/readnone attrs

    if( optLevel > 2 )
        PM.add( createArgumentPromotionPass() );  // Scalarize uninlined fn args

    // Skipping SROA since the incoming PTX doesn't use structs.
    PM.add( createEarlyCSEPass() );       // Catch trivial redundancies
    PM.add( createJumpThreadingPass() );  // Thread jumps.
    // CVP has shown to be a potential bottleneck, so we disable it at -O1.
    if( optLevel > 1 )
        PM.add( createCorrelatedValuePropagationPass() );  // Propagate conditionals
    PM.add( createCFGSimplificationPass() );               // Merge & remove BBs
    PM.add( createInstructionCombiningPass() );            // Combine silly seq's

    PM.add( createTailCallEliminationPass() );  // Eliminate tail calls
    PM.add( createCFGSimplificationPass() );    // Merge & remove BBs
    PM.add( createReassociatePass() );          // Reassociate expressions
    PM.add( createLoopRotatePass() );           // Rotate Loop
    PM.add( createLICMPass() );                 // Hoist loop ilwariants
    PM.add( createLoopUnswitchPass( sizeLevel || optLevel < 3 ) );
    PM.add( createInstructionCombiningPass() );
    PM.add( createIndVarSimplifyPass() );  // Canonicalize indvars
    PM.add( createLoopIdiomPass() );       // Recognize idioms like memset.
    PM.add( createLoopDeletionPass() );    // Delete dead loops

#if !( lateVectorize ) && loopVectorize
    PM.add( createLoopVectorizePass( disableUnrollLoops ) );
#endif

#if !disableUnrollLoops
    PM.add( createLoopUnrollPass() );  // Unroll small loops
#endif

    if( optLevel > 1 )
        PM.add( createGVNPass() );    // Remove redundancies
    PM.add( createMemCpyOptPass() );  // Remove memcpy / form memset
    PM.add( createSCCPPass() );       // Constant prop with SCCP

    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    PM.add( createInstructionCombiningPass() );
    PM.add( createJumpThreadingPass() );  // Thread jumps
    // CVP and DSE have shown to be potential bottlenecks, so we disable them at -O1.
    if( optLevel > 1 )
    {
        PM.add( createCorrelatedValuePropagationPass() );
        PM.add( createDeadStoreEliminationPass() );  // Delete dead stores
    }

#if rerollLoops
    PM.add( createLoopRerollPass() );
#endif
#if doSLPVectorize
    PM.add( createSLPVectorizerPass() );  // Vectorize parallel scalar chains.
#endif

#if doBBVectorize
    {
        PM.add( createBBVectorizePass() );
        PM.add( createInstructionCombiningPass() );
        if( optLevel > 1 && useGVNAfterVectorization )
            PM.add( createGVNPass() );  // Remove redundancies
        else
            PM.add( createEarlyCSEPass() );  // Catch trivial redundancies

        // BBVectorize may have significantly shortened a loop body; unroll again.
        if( !disableUnrollLoops )
            PM.add( createLoopUnrollPass() );
    }
#endif

    PM.add( createAggressiveDCEPass() );         // Delete dead instructions
    PM.add( createCFGSimplificationPass() );     // Merge & remove BBs
    PM.add( createInstructionCombiningPass() );  // Clean up after everything.

    // FIXME: We shouldn't bother with this anymore.
    PM.add( createStripDeadPrototypesPass() );  // Get rid of dead prototypes

    // GlobalOpt already deletes dead functions and globals, at -O2 try a
    // late pass of GlobalDCE.  It is capable of deleting dead cycles.
    if( optLevel > 1 )
    {
        PM.add( createGlobalDCEPass() );      // Remove dead fns and globals.
        PM.add( createConstantMergePass() );  // Merge dup global constants
    }

    // The PTX parser generates vprintf declarations and calls of the wrong
    // type, add a pass to correct it.
    PM.add( createCorrectVprintfTypePass() );
}

static void markForInlining( Function* F )
{
    if( F )
    {
        F->removeFnAttr( Attribute::NoInline );
        F->addFnAttr( Attribute::AlwaysInline );
        F->setLinkage( GlobalValue::InternalLinkage );
    }
}

static void markRtUndefinedUseInlined( llvm::Module* module )
{
    markForInlining( module->getFunction( "_ZN5optix16rt_undefined_useEi" ) );
    markForInlining( module->getFunction( "_ZN5optix18rt_undefined_use64Ey" ) );
}

static bool isRtiInternalRegister( const llvm::GlobalVariable* G )
{
    return G->getName().startswith( "_ZN21rti_internal_register" );
}

static bool isRtiInternalVariable( const llvm::GlobalVariable* G )
{
    const std::string demangledName = optix::canonicalDemangleVariableName( G->getName().str() );

    if( demangledName.find( "rti_internal_semantic::" ) != std::string::npos )
        return true;
    if( demangledName.find( "rti_internal_typeinfo::" ) != std::string::npos )
        return true;
    if( demangledName.find( "rti_internal_typename::" ) != std::string::npos )
        return true;
    if( demangledName.find( "rti_internal_typeenum::" ) != std::string::npos )
        return true;
    if( demangledName.find( "rti_internal_annotation::" ) != std::string::npos )
        return true;

    return false;
}

static bool isRtDeclareVariable( const llvm::Module* module, const llvm::GlobalVariable* G )
{
    // A variable declared with rtDeclareVariable has extra variables
    // accompanying it in the module.  Check if one of those exists.

    const std::string demangledName  = optix::canonicalDemangleVariableName( G->getName().str() );
    const std::string namespacedName = optix::canonicalPrependNamespace( demangledName, "rti_internal_semantic::" );
    const std::string semanticVariableName = optix::canonicalMangleVariableName( namespacedName );

    const llvm::GlobalVariable* GV = module->getGlobalVariable( semanticVariableName, true );

    return GV && GV->hasInitializer();
}

std::string stripInternalNamespace( const std::string& name )
{
    size_t pos = name.find( "rti_internal" );
    if( pos == std::string::npos )
        return name;
    size_t pos1 = name.find( "::", pos );
    if( pos1 == std::string::npos )
        return name;
    return name.substr( 0, pos ) + name.substr( pos1 + 2 );
}

// Pre-process user-defined global variables to catch link errors not reported by LLVM 3.4.
// throws on error
static void prepareUserGlobalsForLinking( const std::string& name, std::vector<llvm::Module*>& modules )
{
    if( modules.size() <= 1 )
        return;

    std::map<llvm::StringRef, llvm::Type*> globalTypes;

    for( llvm::Module* module : modules )
    {
        for( auto G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
        {
            if( !G->hasExternalLinkage() )
                continue;

            // Skip OptiX globals; these are processed elsewhere
            if( isRtiInternalRegister( &*G ) || isRtiInternalVariable( &*G ) || isRtDeclareVariable( module, &*G ) )
                continue;

            // Enforce that the the types match on multiple declarations with the same name.
            // The linker in LLVM version 3.4 does not check this, it only prevents multiple definitions.

            auto inserted = globalTypes.insert( std::make_pair( G->getName(), G->getType() ) );
            if( !inserted.second )
            {
                const std::string demangledName = optix::canonicalDemangleVariableName( G->getName().str() );

                // Compare types
                if( inserted.first->second != G->getType() )
                {
                    throw IlwalidSource( RT_EXCEPTION_INFO,
                                         "Global variable \"" + demangledName
                                             + "\" has multiple declarations that do not have identical types",
                                         name, "Failed to link globals for input modules" );
                }
            }
        }
    }
}

// Pre-process global variables inserted by OptiX (rti_internal, rtDeclareVariable) to make linking work.
// throws on error
static void prepareOptixGlobalsForLinking( const std::string& name, std::vector<llvm::Module*>& modules )
{
    if( modules.size() <= 1 )
        return;

    struct TypeAndInitializer
    {
        const llvm::Type*     type;
        const llvm::Constant* initializer;
    };

    std::map<llvm::StringRef, TypeAndInitializer> globalTypes;

    for( llvm::Module* module : modules )
    {
        for( auto G = module->global_begin(), GE = module->global_end(); G != GE; ++G )
        {
            if( !G->hasExternalLinkage() )
                continue;

            // Skip user-defined globals; these are processed elsewhere
            if( !isRtiInternalRegister( &*G ) && !isRtiInternalVariable( &*G ) && !isRtDeclareVariable( module, &*G ) )
                continue;

            // Allow multiple identical definitions.  Otherwise we would need 'extern' on some.
            G->setLinkage( GlobalValue::LinkOnceODRLinkage );

            // Skip type checking on internal register vars
            if( isRtiInternalRegister( &*G ) )
                continue;

            // Enforce that the the types and initializers match on multiple definitions.
            // The linker in LLVM version 3.4 does not check this.

            const llvm::Constant* initializer = G->hasInitializer() ? G->getInitializer() : nullptr;
            auto inserted = globalTypes.insert( std::make_pair( G->getName(), TypeAndInitializer{G->getType(), initializer} ) );
            if( !inserted.second )
            {
                const std::string demangledName = optix::canonicalDemangleVariableName( G->getName().str() );
                const std::string userFacingName = isRtiInternalVariable( &*G ) ? stripInternalNamespace( demangledName ) : demangledName;

                // Compare types
                TypeAndInitializer& entry = inserted.first->second;
                if( entry.type != G->getType() )
                {
                    throw IlwalidSource( RT_EXCEPTION_INFO, "OptiX variable \"" + userFacingName
                                                                + "\" has multiple declarations that are not identical",
                                         name, "Failed to link globals for input modules" );
                }

                // Compare and update initializers
                if( entry.initializer && initializer && entry.initializer != initializer )
                {
                    throw IlwalidSource( RT_EXCEPTION_INFO, "OptiX variable \"" + userFacingName
                                                                + "\" has multiple declarations that are not identical",
                                         name, "Failed to link globals for input modules" );
                }
                else if( !entry.initializer && initializer )
                {
                    // Found the first initializer for this variable; insert into map
                    entry.initializer = initializer;
                }
            }
        }
    }
}

static bool isOptixFunction( const llvm::Function* F )
{
    if( F->getName().startswith( "optix.ptx." ) )
        return true;
    if( F->getName().startswith( "optix.lwvm." ) )
        return true;
    if( F->getName().startswith( "_rt_" ) )  // OptiX 6
        return true;
    if( F->getName().startswith( "_rti_" ) )  // OptiX 6
        return true;
    if( F->getName().startswith( "_optix_" ) )  // OptiX 7
        return true;
    if( F->getName() == "vprintf" )
        return true;

    return false;
}

// Basic type matching on external function declarations
// throws on error
static void prepareFunctionsForLinking( const std::string& name, std::vector<llvm::Module*>& modules )
{
    if( modules.size() <= 1 )
        return;

    std::map<llvm::StringRef, llvm::Type*> returnTypes;

    for( llvm::Module* module : modules )
    {
        for( auto F = module->begin(), FE = module->end(); F != FE; ++F )
        {
            // If we have already seen a global function, require a matching return type.  LLVM version 3.4 does not do this for us.
            // Note: mangled ptx names for functions contain the arguments, but not the return type.

            if( isOptixFunction( &*F ) || !F->hasExternalLinkage() )
            {
                continue;
            }

            auto inserted = returnTypes.insert( std::make_pair( F->getName(), F->getReturnType() ) );
            if( !inserted.second )
            {
                // Compare return types
                if( inserted.first->second != F->getReturnType() )
                {
                    throw IlwalidSource( RT_EXCEPTION_INFO,
                                         "Function \"" + F->getName().str()
                                             + "\" has multiple declarations that do not have identical return types",
                                         name, "Failed to link functions for input modules" );
                }
            }
        }
    }
}

static void dump( llvm::Module* module, const std::string& dumpName, int dumpId, const std::string& suffix )
{
    addMissingLineInfoAndDump( module, k_saveLLVM.get(), suffix, dumpId, 0 /*m_launchCounterForDebugging*/,
                               dumpName + "-ptx2llvm" );
}

static void optimizeTranslatedModule( llvm::Module* module )
{
    int optLevel = k_ptx2llvm_optLevel.get();
    // Initialize the Vendor library, which is needed for pass dependency
    // resolution
    llvm::PassRegistry& Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeVendor( Registry );

    llvm::legacy::FunctionPassManager FPM( module );
#if DEBUG  // || DEVELOP ) // TODO figure out a way to enable extra checks for DEVELOP builds
    FPM.add( llvm::createVerifierPass() );
#endif
    FPM.add( llvm::createPromoteMemoryToRegisterPass() );
    llvm::legacy::PassManager MPM;

    markRtUndefinedUseInlined( module );

    populateFunctionPassManager( optLevel, FPM );
    populateModulePassManager( optLevel, true /* useAlwaysInliner */, MPM );

    // Run optimizations
    FPM.doInitialization();
    for( Function& F : *module )
        FPM.run( F );
    FPM.doFinalization();
    MPM.run( *module );

    if( k_ptx2llvm_synthesizeAllocaLifetimes.get() )
    {
        llvm::legacy::FunctionPassManager FPM( module );

        // Make sure that all loops are simplified.
        // Some of our analyses rely on simplified loop structures, e.g. collectWrittenRegisters
        // requires a single back edge per loop.
        FPM.add( createLoopSimplifyPass() );

        // Make sure critical edges are broken. Otherwise, alloca lifetime synthesis
        // may introduce artificially long lifetimes.
        FPM.add( createBreakCriticalEdgesPass() );

        FPM.doInitialization();

        for( Function& F : *module )
        {
            FPM.run( F );
            const bool removeOnlyIfLiveInEntireFunction = k_ptx2llvm_removeExistingAllocaLifetimes.get() == "safe";
            const bool removeExistingLifetimes = k_ptx2llvm_removeExistingAllocaLifetimes.get() != "none";
            synthesizeAllocaLifetimes( &F, removeExistingLifetimes, removeOnlyIfLiveInEntireFunction );
        }
        FPM.doFinalization();
    }
}

llvm::Module* PTXtoLLVM::translate( const std::string& name, const std::string& declString, llvm::Module* module, bool skipOptimization )
{
    // Initialize the Vendor library, which is needed for pass dependency resolution.
    // Do this early since inline ASM handling needs dependencies to by resolved.
    llvm::PassRegistry& Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeVendor( Registry );

    // Create FrontEnd with DEBUG_INFO_OFF to avoid adding a second compile unit. For LLVM input we rely
    // on the debug info that is already present in the module.
    PTXFrontEnd frontEnd( module, m_dataLayout, PTXFrontEnd::DEBUG_INFO_OFF, k_optimizeLLVMInput.get() && skipOptimization );
    int           dumpCount       = 0;
    llvm::Module* processedModule = nullptr;
    dump( module, name, dumpCount++, "-input" );
    {
        bool success = frontEnd.processInputLWVM( declString );
        if( !success )
            throw IlwalidSource( RT_EXCEPTION_INFO, frontEnd.getErrorString(), name,
                                 "Failed to process input LLVM bitcode string" );

        dump( module, name, dumpCount++, "-processed" );

        processedModule = frontEnd.translateModule();
        if( !processedModule )
            throw IlwalidSource( RT_EXCEPTION_INFO, frontEnd.getErrorString(), name,
                                 "Failed to translate input LLVM module" );
    }

    dump( processedModule, name, dumpCount++, "-translated" );

    if( frontEnd.needsPtxInstructionsModuleLinked() )
    {
        llvm::Linker linker( *module );
        std::string errs;

        // Link in the early library
        llvm::Module* earlyLib =
            prodlib::ModuleCache::getOrCreateModule( nullptr, m_llvmContext, "libearly", data::getlibPTXFrontEndData(),
                                                     data::getlibPTXFrontEndDataLength(), k_libPTXFrontEndFilename.get() );
        if( !earlyLib )
            throw IlwalidSource( RT_EXCEPTION_INFO, "Failed to load early lib" );

        std::unique_ptr<Module> lwrrLinkModule( earlyLib );
        if( linker.linkInModule( std::move( lwrrLinkModule ), llvm::Linker::StrictMatch ) )
            throw IlwalidSource( RT_EXCEPTION_INFO, errs, name, "Failed to link early lib" );
    }

    dump( processedModule, name, dumpCount++, "-earlylib" );

    if( k_optimizeLLVMInput.get() && !skipOptimization )
        optimizeTranslatedModule( module );

    dump( processedModule, name, dumpCount++, "-optimized" );

    return processedModule;
}

llvm::Module* PTXtoLLVM::translate( const std::string&                      name,
                                    const std::string&                      declString,
                                    const std::vector<prodlib::StringView>& ptxStrings,
                                    bool                                    parseLineNumbers,
                                    const std::string&                      dumpName,
                                    void*                                   decrypter,
                                    DecryptCall                             decryptCall )
{
    TIMEVIZ_FUNC;
    CodeRange range( std::string( "PTX->LLVM " + name ).c_str() );

    std::vector<llvm::Module*> inputModules;
    {
        RT_ASSERT( !ptxStrings.empty() );

        // Counter to not overwrite dump files from previous modules if k_saveInputPtx is set.
        static unsigned int s_ilwocationCount = 0;

        // Parse the PTX
        inputModules.reserve( ptxStrings.size() );
        m_targetArch = 0;
        CodeRange subrange( "Build input LLVM modules" );
        for( size_t i = 0; i < ptxStrings.size(); ++i )
        {
            if( !k_saveInputPtx.get().empty() )
            {
                std::string filename =
                    corelib::createDumpPath( k_saveInputPtx.get(), s_ilwocationCount, i, "input.ptx", "" );
                std::ofstream file( filename );
                if( file )
                {
                    file << ptxStrings[i].data();
                }
                else
                {
                    lerr << "Could not open file " << filename << " for saving input PTX: " << strerror( errno ) << "\n";
                }
            }
            PTXFrontEnd::Debug_info_mode debugMode = parseLineNumbers ? PTXFrontEnd::DEBUG_INFO_LINE : PTXFrontEnd::DEBUG_INFO_OFF;
            PTXFrontEnd                  frontEnd( m_llvmContext, m_dataLayout, debugMode );

            TIMEVIZ_SCOPE( "Parse PTX" );
            bool success = frontEnd.parsePTX( name, declString, ptxStrings[i], decrypter, decryptCall );
            if( !success )
                throw IlwalidSource( RT_EXCEPTION_INFO, frontEnd.getErrorString(), name,
                                        "Failed to parse input PTX string" );
            m_targetArch = std::max( m_targetArch, frontEnd.getTargetArch() );

            {
                TIMEVIZ_SCOPE( "Colwert to LLVM" );
                llvm::Module* module = frontEnd.translateModule();
                if( !module )
                    throw IlwalidSource( RT_EXCEPTION_INFO, frontEnd.getErrorString(), name,
                                         "Failed to translate PTX input to LLVM" );
                inputModules.push_back( module );
            }
        }
        ++s_ilwocationCount;
    }

    prepareUserGlobalsForLinking( name, inputModules );

    prepareOptixGlobalsForLinking( name, inputModules );

    prepareFunctionsForLinking( name, inputModules );

    // Link into single module
    CodeRange     subrange( "Link input LLVM modules" );
    llvm::Module* module = inputModules[0];
    {
        llvm::Linker linker( *module );
        std::string  errs;

        // Link in any remaining modules and destroy them
        for( size_t i = 1; i < inputModules.size(); ++i )
        {
            if( linker.linkInModule( std::unique_ptr<Module>( inputModules[i] ), llvm::Linker::None ) )
                throw IlwalidSource( RT_EXCEPTION_INFO, errs, name, "Failed to link input module" );

            inputModules[i] = nullptr;
        }

        dump( module, dumpName, 0, "-init" );

        // Link in the early library
        subrange( "Link early lib" );
        llvm::Module* earlyLib =
            prodlib::ModuleCache::getOrCreateModule( nullptr, m_llvmContext, "libearly", data::getlibPTXFrontEndData(),
                                                     data::getlibPTXFrontEndDataLength(), k_libPTXFrontEndFilename.get() );
        if( !earlyLib )
            throw IlwalidSource( RT_EXCEPTION_INFO, "Failed to load early lib" );

        std::unique_ptr<Module> lwrrLinkModule( earlyLib );
        if( linker.linkInModule( std::move( lwrrLinkModule ), llvm::Linker::StrictMatch ) )
            throw IlwalidSource( RT_EXCEPTION_INFO, errs, name, "Failed to link early lib" );

        dump( module, dumpName, 1, "-earlylib" );
    }

    inputModules.clear();

    // Check for unresolved function declarations when loading multiple ptx strings.  This is probably user error.
    if( ptxStrings.size() > 1 )
    {
        std::string        str;
        raw_string_ostream errs( str );
        for( llvm::Module::iterator F = module->begin(), FE = module->end(); F != FE; ++F )
        {
            if( isOptixFunction( &*F ) )
                continue;
            if( F->isDeclaration() && !F->isIntrinsic() )
                errs << "Undefined symbol: " << F->getName().str() << '\n';
        }
        if( !errs.str().empty() )
            throw IlwalidSource( RT_EXCEPTION_INFO, errs.str(), name, "Input module contains unresolved functions." );
    }

    // Build optimization stack
    subrange( "Optimizations" );
    {
        TIMEVIZ_SCOPE( "Optimizations" );
        optimizeTranslatedModule( module );
    }

    dump( module, dumpName, 2, "-optimized" );

    return module;
}
