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

#include <LWCA/ComputeCapability.h>
#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <FrontEnd/PTX/Canonical/CanonicalType.h>
#include <FrontEnd/PTX/Canonical/UberPointer.h>
#include <Util/BitSet.h>

#include <corelib/misc/Concepts.h>

#include <llvm/IR/ValueMap.h>

#include <map>
#include <set>
#include <vector>

namespace llvm {
class BitVector;
class Constant;
class Function;
class FunctionType;
class GlobalValue;
class GlobalVariable;
class LLVMContext;
class Module;
class StringRef;
class Type;
class Value;
}

namespace optix {
class CanonicalProgram;
class Context;
class LLVMManager;
class ModuleCache;
class ObjectManager;
class ProgramManager;
class UberPointerSet;
class VariableReference;
class VariableType;
struct AttributeSegment;

/**
   * Take an LLVM input Function and turn it into a canonical
   * representation.
   */
class C14n : private corelib::NonCopyable
{
  public:
    // Note: llvmManager, programManager and objectManager are passed
    // independently of context so that they can be use in testing
    // without a full context.
    C14n( llvm::Function*         function,
          CanonicalizationType    type,
          lwca::ComputeCapability targetMin,
          lwca::ComputeCapability targetMax,
          size_t                  ptxHash,
          Context*                context,
          LLVMManager*            llvmManager,
          ProgramManager*         programManager,
          ObjectManager*          objectManager );
    ~C14n();
    CanonicalProgram* run();

    static std::string getCanonicalizationOptions( const Context* context );


  private:
    enum class TraceVariant
    {
        PLAIN,
        TIME,
        MASK_FLAGS,
        TIME_MASK_FLAGS
    };

    std::unique_ptr<CanonicalProgram> m_cp;

    llvm::Function*      m_function;
    CanonicalizationType m_type;
    Context*             m_context;
    LLVMManager*         m_llvmManager;
    ProgramManager*      m_programManager;
    ObjectManager*       m_objectManager;

    std::map<std::string, VariableReference*> m_variableReferences;
    std::unique_ptr<UberPointerSet> m_up;
    std::vector<llvm::Value*>       m_globalsToRemove;
    llvm::ValueMap<llvm::Value*, VariableReference*> m_variableValues;

    CanonicalProgram* canonicalizePtx();

    void canonicalizeVariables( llvm::Module* module, const llvm::Module* oldModule );
    void canonicalizeInternalRegister( llvm::GlobalVariable* G );
    void canonicalizeInstanceVariable( llvm::GlobalVariable* G,
                                       const std::string&    semantic,
                                       unsigned int          ptxSize,
                                       VariableType          vtype,
                                       unsigned int          typeenum,
                                       const std::string&    annotation );
    void canonicalizeSemanticVariable( llvm::GlobalVariable* G,
                                       const std::string&    semantic,
                                       unsigned int          ptxSize,
                                       const VariableType&   vtype,
                                       const std::string&    annotation );
    void canonicalizeAttribute( llvm::GlobalVariable* G,
                                const std::string&    semantic,
                                unsigned int          ptxSize,
                                const VariableType&   vtype,
                                const std::string&    annotation );
    VariableReference* getOrAddVariable( llvm::GlobalVariable* G, const VariableType& vtype, bool isAttribute = false );
    int getOrAddUberPointerKind( const std::string&       name,
                                 llvm::Constant*          getter,
                                 llvm::Constant*          setter,
                                 llvm::Constant*          atomicSetter,
                                 llvm::Constant*          getAddress,
                                 UberPointer::PointeeType type,
                                 llvm::Value*             defaultValue = nullptr );

    void canonicalizeComplexFunctions( llvm::Module* module );
    void canonicalizeFunctionPrintActive( llvm::Module* module, llvm::Function* printActiveFunction, std::vector<llvm::Value*>& toDelete );
    void canonicalizeFunctionPrintStart( llvm::Module* moduel, llvm::Function* printStartFunction, std::vector<llvm::Value*>& toDelete );

    void canonicalizeGetBufferSize( llvm::Function* fn );
    void canonicalizeAccessBuffer( llvm::Function* fn, std::vector<llvm::Value*>& toDelete );
    void canonicalizeLoadOrRequest( llvm::Function* fn, std::vector<llvm::Value*>& toDelete );
    void canonicalizeGetBufferSizeFromId( llvm::Function* fn );
    void canonicalizeAccessBufferFromId( llvm::Function* function, std::vector<llvm::Value*>& toDelete );
    void canonicalizeCallableProgramFromId( llvm::Module* module );
    void canonicalizeTextures( llvm::Function* function, std::vector<llvm::Value*>& toDelete );
    void canonicalizeBindlessTextures( llvm::Module* module, std::vector<llvm::Value*>& toDelete );
    void canonicalizeDemandLoadBindlessTextures( llvm::Module* module, std::vector<llvm::Value*>& toDelete );
    void canonicalizeTraceGlobal( llvm::Function* fn, std::vector<llvm::Value*>& toDelete, TraceVariant );

    void makeIntersectionAndAttributeFunctions( llvm::Function* fn );
    bool generateDeferredAttributes( llvm::Function* fn, std::vector<AttributeSegment>& segments );
    void generateDefaultAttributeDecoder( llvm::Type* attributesTy );
    void computeAttributeData32bitValues( llvm::Function* fn );

    void validateFunctionGetTransform( llvm::Function* function );
    void validateFunctionTransformTuple( llvm::Function* function );
    void validateFunctionThrow( llvm::Function* function );

    void validateRemainingFunctions( llvm::Function* function );
    void validateRemainingVariables( llvm::Module* module );

    void changeAddressSpaceTo( llvm::Instruction* inst, unsigned newAddrSpace ) const;

    void checkVariableSizes( llvm::GlobalVariable* G, const VariableType& vtype, unsigned int ptxSize );

    unsigned short registerVariableName( const std::string& name );
    unsigned registerCallableFunctionSignature( llvm::FunctionType* ftype );

    bool getString( const llvm::Module* module, const std::string& name, std::string& returnString ) const;
    bool getTypeInfo( const llvm::Module* module, const std::string& name, unsigned int& returnKind, unsigned int& returnSize ) const;
    bool getInt( const llvm::Module* module, const std::string& name, unsigned int& returnKind ) const;
    const llvm::Constant* getInitializer( const llvm::Module* module, const std::string& name ) const;

    CanonicalProgram*        canonicalizeTraverser();
    const VariableReference* addVariable( const std::string& name, const VariableType& vtype );

    static TraceVariant getTraceVariant( const llvm::StringRef& name );
    static size_t getParameterCount( TraceVariant );
    static bool   hasTime( TraceVariant );
    static bool   hasMaskAndFlags( TraceVariant );
};

}  // namespace optix
