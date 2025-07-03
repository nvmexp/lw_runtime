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

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#if defined( DEBUG ) || defined( DEVELOP )
#include <FrontEnd/PTX/D2IRPTXInstructions_bin.h>
#endif
#include <FrontEnd/PTX/Intrinsics/D2IRIntrinsicBuilder.h>
#include <FrontEnd/PTX/LinkPTXFrontEndIntrinsics.h>
#include <FrontEnd/PTX/LowerCarryInstructionsPass.h>
#include <FrontEnd/PTX/PTXInstructions_bin.h>
#include <FrontEnd/PTX/libActivemaskEmulate_bin.h>
#include <FrontEnd/PTX/libDevice_bin.h>
#include <FrontEnd/PTX/libDirect2IR_bin.h>
#include <FrontEnd/PTX/libLWPTX_bin.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>

#include <sstream>

namespace {
// clang-format off
Knob<bool> k_generateD2IRWrappers( RT_DSTRING( "compile.newBackend.generateIntrinsics" ), true, RT_DSTRING( "When true, dynamically generate PTX intrinsics instead of linking them" ) );
// clang-format on
}

namespace optix_exp {

static void defineVideoCalls( llvm::Module* mod )
{
    for( llvm::Function& func : *mod )
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


        llvm::InlineAsm* inlineCall = llvm::InlineAsm::get( func.getFunctionType(), inlineStr.str(), constraints.str(), true );

        llvm::BasicBlock*      BB = llvm::BasicBlock::Create( mod->getContext(), "entry", &func );
        corelib::CoreIRBuilder irb( BB );

        func.addFnAttr( llvm::Attribute::AlwaysInline );
        func.setLinkage( llvm::GlobalValue::LinkOnceAnyLinkage );

        std::vector<llvm::Value*> arguments;

        for( llvm::Function::arg_iterator it = func.arg_begin(), end = func.arg_end(); it != end; ++it )
            arguments.push_back( &*it );
        llvm::Value* ret = irb.CreateCall( inlineCall, arguments );

        irb.CreateRet( ret );
    }
}

static bool startsWith( llvm::StringRef name, const std::string& prefix )
{
    if( prefix.size() > name.size() )
        return false;
    auto res = std::mismatch( prefix.begin(), prefix.end(), name.begin() );
    return res.first == prefix.end();
}

static bool isInlinePTXIntrinsic( llvm::StringRef funcName )
{
    std::vector<std::string> inlinePtxPrefixes = {"optix.ptx.", "optix.lwvm."};
    for( const std::string& prefix : inlinePtxPrefixes )
    {
        if( startsWith( funcName, prefix ) )
            return true;
    }
    return false;
}

static bool d2irPathImplementsModuleIntrinsics( llvm::Module* libDirect2IRModule,
                                                llvm::Module* d2irPtxInstructionsModule,
                                                llvm::Module* module,
                                                ErrorDetails& errDetails )
{
    bool providesIntrinsics = true;

    // Check that every optix.ptx.* intrinsic used by this module is provided by one of the D2IR modules.
    for( auto func_it = module->begin(), func_end = module->end(); func_it != func_end; func_it++ )
    {
        if( !isInlinePTXIntrinsic( func_it->getName() ) || !func_it->empty() || func_it->use_empty() )
            continue;

        // If the given intrinsic is an instruction with carry, the "LowerCarryInstructions" pass will handle it.
        if( optix::lowerCarryInstructionsPassHandlesIntrinsic( func_it->getName() ) )
            continue;
        // Otherwise, the instruction must be implemented in one of our D2IR wrapper modules
        else if( !libDirect2IRModule->getFunction( func_it->getName() )
                 && !d2irPtxInstructionsModule->getFunction( func_it->getName() ) )
        {
            errDetails.m_compilerFeedback << "New backend is missing implementation for PTX intrinsic "
                                          << func_it->getName().str() << "\n";
            providesIntrinsics = false;
        }
    }

    return providesIntrinsics;
}

static void runLowerCarryInstructionsPass( llvm::Module* module )
{
    llvm::legacy::PassManager carryLowerPM;
    carryLowerPM.add( optix::createLowerCarryInstructionsPass() );
    carryLowerPM.run( *module );
}

static OptixResult linkInModule( llvm::Linker& linker, llvm::Module* module, const std::string& moduleName, ErrorDetails& errDetails )
{
    if( linker.linkInModule( std::unique_ptr<llvm::Module>( module ), llvm::Linker::LinkOnlyNeeded ) )
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR, "Failed to link " + moduleName + " module" );
    return OPTIX_SUCCESS;
}

OptixResult linkPTXFrontEndIntrinsics( llvm::Module* module, bool enableD2IR, bool enableLWPTXFallback, bool& fellBackToLWPTX, ErrorDetails& errDetails )
{
// Run a dead code elimination pass so we don't worry about implementing
// PTX intrinsics we don't use in practice.
    llvm::legacy::PassManager DCEPM;
    DCEPM.add( llvm::createGlobalDCEPass() );
    DCEPM.run( *module );

    llvm::Linker linker( *module );

    defineVideoCalls( module );

    // Make sure no ".var" names are present otherwise LWVM will spit out a .name PTX variable (invalid)
    for( llvm::Module::global_iterator I = module->global_begin(), E = module->global_end(); I != E; ++I )
    {
        if( !I->getName().empty() && I->getName()[0] == '.' )
        {
            std::string newName( I->getName() );
            newName[0] = 'v';
            I->setName( newName );
        }
    }

    // Link libDevice functions.
    llvm::Module* libDeviceModule = corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getlibDeviceData(),
                                                                        ::optix::data::getlibDeviceDataLength() );

    if( OptixResult res = linkInModule( linker, libDeviceModule, "device", errDetails ) )
        return res;

    fellBackToLWPTX = false;

    if( enableD2IR )
    {
        llvm::Module* libDirect2IRModule =
            corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getlibDirect2IRData(),
                                                ::optix::data::getlibDirect2IRDataLength() );

#if defined( DEBUG ) || defined( DEVELOP )
        if( !k_generateD2IRWrappers.get() )
        {
            llvm::Module* d2irPtxInstructionsModule =
                corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getD2IRPTXInstructionsData(),
                                                    ::optix::data::getD2IRPTXInstructionsDataLength() );

            if( d2irPathImplementsModuleIntrinsics( libDirect2IRModule, d2irPtxInstructionsModule, module, errDetails ) )
            {
                runLowerCarryInstructionsPass( module );

                if( OptixResult res = linkInModule( linker, libDirect2IRModule, "direct2IR", errDetails ) )
                    return res;
                if( OptixResult res = linkInModule( linker, d2irPtxInstructionsModule, "direct2IRPtx", errDetails ) )
                    return res;
                return OPTIX_SUCCESS;
            }
            else
            {
                if( enableLWPTXFallback )
                {
                    fellBackToLWPTX = true;
                    errDetails.m_compilerFeedback << "Unimplemented PTX intrinsics. Falling back to old backend.\n";
                }
                else
                {
                    return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                                  "unable to compile with new backend: unimplemented PTX intrinsics "
                                                  "(see compile feedback for details)" );
                }
            }
        }
        else
#endif
        {
            optix::PTXIntrinsics::D2IRIntrinsicBuilder intrinsicBuilder( module->getContext(), module );

            // Keep track of added intrinsics so we can delete them if we need to fall back to LWPTX
            std::vector<std::string> addedIntrinsics;

            bool d2irPathImplementsIntrinsics = true;
            for( auto func_it = module->begin(), func_end = module->end(); func_it != func_end; func_it++ )
            {
                if( !isInlinePTXIntrinsic( func_it->getName() ) || !func_it->empty() || func_it->use_empty() )
                    continue;

                // Function name might be changed by wrapper builder (functions
                // are replaced to avoid ilwalidating iterators)
                const std::string funcName = func_it->getName().str();

                // If the given intrinsic is an instruction with carry, the "LowerCarryInstructions" pass will handle it.
                if( optix::lowerCarryInstructionsPassHandlesIntrinsic( func_it->getName() ) )
                    continue;
                else if( libDirect2IRModule->getFunction( func_it->getName() ) )
                    continue;
                else if( intrinsicBuilder.addIntrinsic( &*func_it ) )
                {
                    addedIntrinsics.push_back( funcName );
                    continue;
                }

                d2irPathImplementsIntrinsics = false;
                errDetails.m_compilerFeedback << "New backend is missing implementation for PTX intrinsic " << func_it->getName().str() << "\n";
            }

            if( !d2irPathImplementsIntrinsics )
            {
                if( enableLWPTXFallback )
                {
                    // Delete added intrinsics
                    for( const std::string& addedIntrinsic : addedIntrinsics )
                        module->getFunction( addedIntrinsic )->deleteBody();

                    fellBackToLWPTX = true;
                    errDetails.m_compilerFeedback << "Unimplemented PTX intrinsics. Falling back to old backend.\n";
                }
                else
                {
                    return errDetails.logDetails( OPTIX_ERROR_INTERNAL_COMPILER_ERROR,
                                                  "unable to compile with new backend: unimplemented PTX intrinsics" );
                }
            }
            else
            {

                runLowerCarryInstructionsPass( module );
                if( OptixResult res = linkInModule( linker, libDirect2IRModule, "direct2IR", errDetails ) )
                    return res;

                return OPTIX_SUCCESS;
            }
        }
    }

    llvm::Module* libActivemaskEmulateModule =
        corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getlibActivemaskEmulateData(),
                                            ::optix::data::getlibActivemaskEmulateDataLength() );

    if( OptixResult res = linkInModule( linker, libActivemaskEmulateModule, "activemask", errDetails ) )
        return res;

    llvm::Module* libLWPTXModule = corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getlibLWPTXData(),
                                                                       ::optix::data::getlibLWPTXDataLength() );

    if( OptixResult res = linkInModule( linker, libLWPTXModule, "LWPTX", errDetails ) )
        return res;

    // Link wrappers for ptx instructions.
    // Do not use the module cache since (a) the cached module most likely uses the wrong LLVM context and (b) our
    // temporary LLVM context should not end up in the cache.
    llvm::Module* ptxInstructionsModule =
        corelib::loadModuleFromBitcodeLazy( module->getContext(), ::optix::data::getPTXInstructionsData(),
                                            ::optix::data::getPTXInstructionsDataLength() );

    if( OptixResult res = linkInModule( linker, ptxInstructionsModule, "PTX instructions", errDetails ) )
        return res;

    return OPTIX_SUCCESS;
}
}
