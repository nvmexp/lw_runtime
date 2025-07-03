// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <FrontEnd/Canonical/LineInfo.h>

#include <corelib/compiler/LLVMUtil.h>

#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/LLVMContext.h>

#include <cassert>


namespace optix {
// clang-format off
Knob<bool> k_addMissingLineInfo(RT_DSTRING( "compile.addMissingLineInfo" ),false,RT_DSTRING( "Add missing lineinfo, annotating new or rewritten instructions. false: Disabled, true: In between transformations, labeled individually." ) );
// clang-format on
namespace {
const char OPTIX_GENERATED_DIR[] = "OPTIX/generated/";

using namespace prodlib;
using namespace corelib;
using namespace llvm;
}


const char* getGeneratedCodeDirectory()
{
    return OPTIX_GENERATED_DIR;
}

void addLineInfoToFunction( llvm::Function* func, const llvm::StringRef& filename )
{
    auto mod = func->getParent();
    assert( mod );
    return addLineInfoToFunction( func, *mod, filename );
}

void addLineInfoToFunction( llvm::Function* func, llvm::Module& module, const llvm::StringRef& filename )
{
    if( k_addMissingLineInfo.get() )
    {
        unsigned              line = 1;
        llvm::DIBuilder       dib{ module };
        const llvm::StringRef dir{ OPTIX_GENERATED_DIR };
        auto                  file = dib.createFile( filename, dir );
        auto lw = dib.createCompileUnit( llvm::dwarf::DW_LANG_C_plus_plus, file, "LWPU OptiX compiler", false,
                                         llvm::StringRef{}, 0 );

        // not sure if correct parameters are needed here
        auto ty = dib.createSubroutineType( dib.getOrCreateTypeArray( {} ) );
        dib.createFunction( file, func->getName(), llvm::StringRef{}, file, line, ty, false, true, line );
        func->setMetadata( llvm::LLVMContext::MD_dbg, lw );
        dib.finalize();
    }
}

bool hasLineInfo( const Module* module )
{
    for( const Function& f : *module )
    {
        for( const BasicBlock& bb : f )
        {
            for( const Instruction& i : bb )
            {
                if( i.getDebugLoc().get() )
                {
                    const DILocation* lwr      = i.getDebugLoc().get();
                    const auto&       filename = lwr->getFilename();
                    const auto&       dir      = lwr->getDirectory();
                    if( !filename.empty() && filename != "unknown" && dir != OPTIX_GENERATED_DIR )
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void addMissingLineInfo( Module* module, const std::string& filename )
{
    llvm::DIBuilder       builder{*module};
    const llvm::StringRef directory{OPTIX_GENERATED_DIR};
    MDNode*               scope = builder.createFile( filename, directory );

    for( Function& f : *module )
    {
        for( BasicBlock& bb : f )
        {
            for( Instruction& i : bb )
            {
                if( !i.getDebugLoc().get() )
                {
                    i.setDebugLoc( DebugLoc::get( 1, 1, f.getSubprogram() ) );
                }
                else if( i.getDebugLoc().getCol() == 0 )
                {
                    i.setDebugLoc( DebugLoc::get( i.getDebugLoc().getLine(), 1, i.getDebugLoc().getScope(),
                                                  i.getDebugLoc().getInlinedAt() ) );
                }
            }
        }
    }
}

void addMissingLineInfoAndDump( Module*            module,
                                const std::string& outfile_pattern,
                                const std::string& identifier,
                                int                dumpId,
                                int                launchCounter,
                                const std::string& functionName )
{
#if defined( DEBUG ) || defined( DEVELOP )
    if( k_addMissingLineInfo.get() || !outfile_pattern.empty() )
    {
        if( k_addMissingLineInfo.get() )
        {
            dumpId *= 2;  // we write two files: one before and one after debug info holes get filled (for easy diffs)
        }
        std::string filename = createDumpPath( outfile_pattern, launchCounter, dumpId, identifier, functionName );

        if( !outfile_pattern.empty() )
        {
            lprint << "Writing LLVM ASM file to: " << filename << "\n";
            saveModuleToAsmFile( module, filename );
        }

        if( k_addMissingLineInfo.get() )
        {
            addMissingLineInfo( module, identifier );
            if( !outfile_pattern.empty() )
            {
                filename = createDumpPath( outfile_pattern, launchCounter, dumpId + 1, identifier, functionName );
                lprint << "Writing LLVM ASM file (holes in debug info plugged) to: " << filename << "\n";
                saveModuleToAsmFile( module, filename );
            }
        }
    }
#endif
}
}
