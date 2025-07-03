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

#pragma once

#include <vector>

#include <Context/BindingManager.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Common/Specializations.h>
#include <ExelwtionStrategy/Common/VariableSpecialization.h>

#include <LWCA/ComputeCapability.h>

namespace llvm {
class Instruction;
class Module;
class Type;
class Value;
}

namespace optix {

namespace RTXSpecializer {
std::string computeDumpName( SemanticType stype, const std::string& functionName );
}  // end namespace RTXSpecializer

// The specializer provides an interface to run specialization passes.
class RTXVariableSpecializer
{
  public:
    RTXVariableSpecializer( const Specializations& specializations,
                            llvm::Function*        entryFunction,
                            SemanticType           stype,
                            SemanticType           inheritedStype,
                            bool                   deviceSupportsLDG,
                            bool                   useConstMemory,
                            const ProgramManager*  programManager,
                            int                    launchCounterForDebugging );

    void runOnModule( llvm::Module* module, const std::string& dumpName );

  private:
    void initializeRuntimeFunctions( llvm::Function* entryFunction );

    //  This specialization is not safe, see the comment in MegakernelCompile::applySpecializations.
    //  void performLDGReadOnlyAnalysis( llvm::Module* module, const std::vector<VariableSpecialization>& specializations );

    void specializeVariable( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    void specializeBuffer( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    void specializeRtxiGetBufferSize( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    void specializeRtxiGetBufferId( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    // Replace all calls to bindless buffer lookups to use the specialized megakernel version.
    // The _linear version of the getElementAddress functions does not take in input a pointer to stack memory.
    // This improves compile times, since the optimizer does not have to remove unneeded alloca instructions.
    void specializeBindlessBuffers( llvm::Module* module );


    // Handle textures separately, because we have to look for all the functions that match
    // Texture_* rather than looking for the function by name directly (we can't construct
    // the function name from the available data, we have to pull it out of the function
    // name itself).
    void applyTextureSpecializations( llvm::Module* module );

    // Find or create the runtime function of the given name
    llvm::Function* findRuntimeFunction( llvm::Module*                   module,
                                         const std::string&              name,
                                         llvm::Type*                     returnType,
                                         const std::vector<llvm::Type*>& argTypes );

    llvm::Value* loadSpecializedVariable( llvm::Module*                 module,
                                          const VariableSpecialization& vs,
                                          llvm::Type*                   returnType,
                                          llvm::Value*                  stateptr,
                                          llvm::Instruction*            insertBefore,
                                          const std::string&            varname,
                                          unsigned short                token,
                                          llvm::Value*                  defaultValue,
                                          llvm::Value*                  offset );

    template <class BufferAccess>
    llvm::Value* createGetElementAddressCall( VariableReferenceID           refID,
                                              llvm::Type*                   valueType,
                                              BufferAccess*                 origCall,
                                              const VariableSpecialization& vs );

    // still needed for atomics
    template <class AccessBufferCall>
    llvm::Value* createGetBufferElementCall( VariableReferenceID           refID,
                                             llvm::Type*                   valueType,
                                             AccessBufferCall*             origCall,
                                             const VariableSpecialization& vs );

    void specializeBuffer_pitchedLinear( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    void specializeBuffer_texHeap( llvm::Module* module, VariableReferenceID refID, const VariableSpecialization& vs );

    void dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix );

  private:
    const Specializations& m_specializations;
    SemanticType           m_stype                     = ST_ILWALID;
    SemanticType           m_inheritedStype            = ST_ILWALID;
    bool                   m_deviceSupportsLDG         = false;
    bool                   m_useConstMemory            = false;
    const ProgramManager*  m_programManager            = nullptr;
    int                    m_launchCounterForDebugging = 0;

    llvm::Type*     m_statePtrTy                  = nullptr;
    llvm::Type*     m_i32Ty                       = nullptr;
    llvm::Type*     m_i8PtrTy                     = nullptr;
    llvm::Type*     m_constMemi8PtrTy             = nullptr;
    llvm::Function* m_genericLookupFunc           = nullptr;
    llvm::Function* m_bufferSizeFromIdFunc        = nullptr;
    llvm::Function* m_bufferElementFromIdFuncs[3] = {nullptr, nullptr, nullptr};
};

class RTXGlobalSpecializer
{
  public:
    RTXGlobalSpecializer( int dimensionality, unsigned int minTransformDepth, unsigned int maxTransformDepth, bool printEnabled, int launchCounterForDebugging );

    void runOnModule( llvm::Module* module, const std::string& dumpName );

  private:
    // Check if specializeTransformDepth will do something.
    // This is used to prevent max transform depth from being moved to the interstate.
    static bool canSpecializeTransformDepthToConstant( unsigned int minTransformDepth, unsigned int maxTransformDepth );

    // Specialize transform code for special cases max transform depth ==0, ==1, or <=1.
    static void specializeTransformDepth( llvm::Module* module, unsigned int minTransformDepth, unsigned int maxTransformDepth );

    // Specialize optixi_isPrintingEnabled calls if printing is disabled or enabled for all launch indices.
    static void specializePrintActive( llvm::Module* module );

    // Specialize optixi_getLaunchIndex based on the dimensionality.
    void specializeGetLaunchIndex( llvm::Module* module );

    void dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix );

  private:
    int          m_dimensionality            = 0;
    unsigned int m_minTransformDepth         = 0;
    unsigned int m_maxTransformDepth         = 0;
    bool         m_printEnabled              = false;
    int          m_launchCounterForDebugging = 0;
};
}
