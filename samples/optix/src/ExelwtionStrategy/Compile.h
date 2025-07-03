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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/SemanticType.h>
#include <corelib/compiler/LLVMSupportTypes.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Regex.h>
#include <map>
#include <set>
#include <vector>

namespace llvm {
class CallInst;
class Constant;
class Function;
class GlobalVariable;
class Instruction;
class Module;
class StringRef;
class Type;
class Value;
}

namespace optix {
class CanonicalProgram;
class LLVMManager;
class Plan;
class ProgramManager;
class AtomicSetBufferElement;

enum class AtomicSubElementType
{
    INT32 = 0,
    FLOAT32,
    INT64,
    INVALID
};

enum class AtomicOpType
{
    ADD = 0,
    SUB,
    EXCH,
    MIN,
    MAX,
    INC,
    DEC,
    CAS,
    AND,
    OR,
    XOR,
    INVALID
};

// Helper that colwerts the provided value to a sub-element type. Will assert if val isn't a ConstantInt.
AtomicSubElementType valueToSubElementType( llvm::Value* val );

// Helper that casts the provided value to an i64 in a way that works depending on it's original type.
llvm::Value* castToInt64( llvm::Instruction* insertBefore, llvm::Value* val );

// Helper that gets the operand and compareOperand colwerted to the correct type depending on the
// subElementType argument of the call. Returns a llvm::Type that
// corresponds to the subElementType. Asserts if the subElementType isn't one of the supported types.
// Only valid for template types AtomicSetBufferElement, AtomicSetBufferElementFromId, RtxiAtomicSetBufferElement.
// and RtxiAtomicSetBufferElementFromId.
template <typename T>
llvm::Type* getAtomicOperands( T* call, llvm::Value** oOperand, llvm::Value** oCompareOperand );

// Helper function to generate an atomic call from the given parameters
llvm::CallInst* createAtomicCall( llvm::Type*        returnType,
                                  AtomicOpType       op,
                                  llvm::Value*       addressPtr,
                                  llvm::Value*       compareOperand,
                                  llvm::Value*       atomicOperand,
                                  llvm::Instruction* insertBefore );


// Most accessors such as optixi_getVariableValue and optixi_getBufferElement
// have a common format, as seen in the regex below.
// optixi_getBufferSize, optixi_trace and optixi_getPayload are slightly
// different and are handled by additional regexes.
// NOTE: Can't be const'ified due to Regex::match not being const.
// SGP TODO: This is not acceptable. We must not have static globals that allocate memory. This needs to get moved into a manager class.
// clang-format off
// TODO: placeHolderRegex and PayloadRegex overlap (payload should not be matched by placeholder). One possible
// solution would be to disallow "." in the function name.
const std::string optixi_CanonicalProgramUniqueNameRegex = ".+_ptx0x[0-9A-Fa-f]+";
const std::string optixi_VariableReferenceUniqueNameRegex = optixi_CanonicalProgramUniqueNameRegex + "\\.[^\\.]+";

static llvm::Regex optixi_PlaceholderRegex = llvm::Regex( "optixi_(get|set)(.+)Value\\.(" + optixi_VariableReferenceUniqueNameRegex + ")(\\..+)?$" );
static llvm::Regex optixi_BufferSizeRegex  = llvm::Regex( "optixi_getBufferSize\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );
static llvm::Regex optixi_PayloadRegex = llvm::Regex( "optixi_(get|set)PayloadValue\\.prd([0-9]+)b\\.(" + optixi_VariableReferenceUniqueNameRegex + "\\..+)$" );
static llvm::Regex optixi_TraceRegex = llvm::Regex( "optixi_trace\\.(" + optixi_CanonicalProgramUniqueNameRegex + ")\\.prd([0-9]+)b$" );
static llvm::Regex optixi_CallableRegex = llvm::Regex( "optixi_call(Bound|Bindless)(\\.(" + optixi_VariableReferenceUniqueNameRegex + "))?\\.sig([0-9]+)$" );
// clang-format on

// Regex parsing functions for "demangling" of optixi_ calls.
bool parsePlaceholderName( const llvm::StringRef& fname, llvm::StringRef& kind, bool& isSet, llvm::StringRef& variableReferenceUniqueName );
bool parseBufferSizeName( const llvm::StringRef& fname, llvm::StringRef& variableReferenceUniqueName );
bool parsePayloadName( const llvm::StringRef& fname, bool& isSet, llvm::StringRef& payloadName );
bool parseTraceName( const llvm::StringRef& fname );
bool parseCallableProgramName( const llvm::StringRef& fname, bool& isBound, llvm::StringRef& variableReferenceUniqueName, unsigned& sig );

bool isIntrinsic( llvm::StringRef functionName, const std::string& prefix, const std::string& varRefUniqueName );

typedef std::set<CanonicalProgramID> CPIDSet;

// Functions for setting up attribute offsets.
typedef std::map<unsigned short, unsigned> AttributeOffsetMap;  // token->offset
unsigned int computeAttributeOffsets( AttributeOffsetMap&                 attributeOffsets,
                                      const std::set<CanonicalProgramID>& CPs,
                                      ProgramManager*                     programManager );

// Creates a jump table global variable containing the given functions
void createJumpTable( llvm::Module* module, std::vector<llvm::Function*>& functions );

// Replace placeholder accessors from canonicalization with calls to
// the CommonRuntime.  hwOnlyVarRefSet contains the list of variable
// references that should be specialized for hardware texturing.
// Pass a semantic type if variable address lookup should use this information
// (has to be supported by the runtime, lwrrently only in RTX).
void replacePlaceholderAccessors( llvm::Module*             module,
                                  const AttributeOffsetMap& attributeOffsets,
                                  const ProgramManager*     programManager,
                                  bool                      hwtexonly,
                                  bool                      swtexonly,
                                  SemanticType*             stype          = nullptr,
                                  SemanticType*             inheritedStype = nullptr );
unsigned int replaceTracePlaceholders( llvm::Module* module );  // Returns maximum payload size
void generateCallableProgramDispatch( llvm::Module* module );
void generateFunctionFromPrototype( llvm::Function* dst, llvm::Function* src, int numCopyParameters, const std::string& extName );

// Utility for dealing with runtime functions. Similar to
// llvm::Module::getOrInsert except that it will throw an error on
// type mismatch rather than returning a casted function pointer.
llvm::Function* findOrCreateRuntimeFunction( llvm::Module*                   module,
                                             const std::string&              name,
                                             llvm::Type*                     returnType,
                                             const std::vector<llvm::Type*>& argTypes );

llvm::CallInst* createCallableProgramCall( llvm::CallInst*     CI,
                                           llvm::Value*        pid,
                                           bool                isBound,
                                           unsigned short      sig,
                                           VariableReferenceID refid,
                                           unsigned short      token,
                                           llvm::StringRef     varname );

// Texture generation helpers
void generateTextureDeclarations( llvm::Module* module, const std::string& basename, int count );

// Gives unnamed values a name
void dbgNameUnnamedVals( llvm::Module* module );

// Insert print statements for values specified in compile.dbgPrintVals knob
void dbgInsertPrintsVals( llvm::Module* module );

// Returns true if this GlobalVariable will be uniquified.
bool includeForUniquifyingConstMemoryInitializers( const llvm::GlobalVariable* G );

// Make all statically initialized constant memory values unique (i.e. remove duplicates)
// This calls includeForUniquifyingConstMemoryInitializers().
void uniquifyConstMemoryInitializers( llvm::Module* module );

// Transforms calls to rt_printf into calls to vprintf.
void printfToVprintf( llvm::Module* module );

// If fnTy is non-null, then a declaration of the function is added if not found.
llvm::Function* getAtomicFunction( llvm::Module* module, AtomicOpType atomicOperator, llvm::Type* type, llvm::FunctionType* fnTy = nullptr );

void replaceGetBufferSizeFromId( llvm::Function* function, llvm::Function* const runtimeFunction, std::vector<llvm::Value*>& toDelete );

template <class Intrinsic>
void replaceGetBufferFromId( llvm::Function*                 F,
                             llvm::Function*                 bufferElementFuncs[],
                             bool                            generateAlloca,
                             corelib::ValueVector&           toDelete,
                             std::vector<corelib::InstPair>& toReplace );

template <class Intrinsic>
void replaceSetBufferFromId( llvm::Function*                 F,
                             llvm::Function*                 bufferElementFuncs[],
                             bool                            generateAlloca,
                             corelib::ValueVector&           toDelete,
                             std::vector<corelib::InstPair>& toReplace );

template <class Intrinsic>
void replaceGetBufferElementAddressFromId( llvm::Function*       function,
                                           llvm::Function*       bufferElementFromIdFunc[],
                                           bool                  generateAlloca,
                                           corelib::ValueVector& toDelete );

template <class Intrinsic>
void replaceAtomicIntrinsicFromId( llvm::Function* F, llvm::Function* bufferElementFuncs[], bool generateAlloca, corelib::ValueVector& toDelete );

}  // namespace optix
