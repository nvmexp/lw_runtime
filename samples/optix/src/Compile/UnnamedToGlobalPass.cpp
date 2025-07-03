// Copyright (c) 2019, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Compile/UnnamedToGlobalPass.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

#include <llvm/IR/LegacyPassManager.h>

#include <llvm/Pass.h>

#include <lwvm/ClientInterface/LWVM.h>

using namespace llvm;

namespace {

class UnnamedToGlobalAddressSpacePass : public ModulePass
{
  public:
    static char ID;  // Pass identifier

    UnnamedToGlobalAddressSpacePass()
        : ModulePass( ID )
    {
    }

    void getAnalysisUsage( AnalysisUsage& AU ) const override { AU.setPreservesAll(); }

    virtual bool runOnModule( Module& module )
    {
        // Determine which, if any, global variables we need to replace.
        std::vector<GlobalVariable*> varsInUnnamedAddrSpace;
        for( Module::global_iterator gvar = module.global_begin(), end = module.global_end(); gvar != end; ++gvar )
        {
            if( gvar->getType()->getAddressSpace() == lwvm::ADDRESS_SPACE_GENERIC )
                varsInUnnamedAddrSpace.push_back( &*gvar );
        }

        for( GlobalVariable* oldVar : varsInUnnamedAddrSpace )
        {
            PointerType* oldTy = oldVar->getType();
            PointerType* newTy = PointerType::get( oldTy->getElementType(), lwvm::ADDRESS_SPACE_GLOBAL );

            // Create a new global that is exactly the same as the old one, except it's in the global address space.
            Constant* initializer = oldVar->hasInitializer() ? oldVar->getInitializer() : nullptr;
            GlobalVariable* newVar = new GlobalVariable( module, oldVar->getType()->getElementType(), oldVar->isConstant(),
                                                         oldVar->getLinkage(), initializer, oldVar->getName(),
                                                         nullptr, oldVar->getThreadLocalMode(), lwvm::ADDRESS_SPACE_GLOBAL );

            // Create an addrspacecast from the new global to the type of the old global.
            Constant* newVarGenericPtr = llvm::ConstantExpr::getAddrSpaceCast( newVar, oldTy );

            oldVar->replaceAllUsesWith( newVarGenericPtr );
            newVar->takeName( oldVar );
            oldVar->eraseFromParent();
        }

        return false;
    }
};

char UnnamedToGlobalAddressSpacePass::ID = 0;

}  // anonymous namespace

namespace optix {

ModulePass* createUnnamedToGlobalAddressSpacePass()
{
    return new UnnamedToGlobalAddressSpacePass();
}

void moveVariablesFromUnnamedToGlobalAddressSpace( llvm::Module* module )
{
    llvm::legacy::PassManager PM;
    PM.add( createUnnamedToGlobalAddressSpacePass() );
    PM.run( *module );
}
}
