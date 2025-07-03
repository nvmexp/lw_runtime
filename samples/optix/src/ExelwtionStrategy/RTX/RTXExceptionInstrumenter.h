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

#include <string>
#include <vector>

#include <o6/optix.h>

#include <corelib/compiler/CoreIRBuilder.h>

#include <Objects/SemanticType.h>

namespace llvm {
class CallInst;
class Function;
class Instruction;
class LLVMContext;
class Module;
class Value;
}

namespace optix {

class Context;
class LLVMManager;
class ProgramManager;

/// This class inserts checks for various OptiX exceptions (if enabled via the context or knobs).
///
/// In order to make the generated exception checks efficient, these checks are inserted before the
/// specializer runs. To avoid repeated computations, the optixi intrinsics are split into more
/// fine-grained RTX intrinsics, such that the information from the RTX intrinsics can be re-used
/// for the exception checks. This splitting of intrinsics happens even if all exception checks are
/// disabled to prevent later stages from having to deal with both kind of intrinsics.
///
/// Note that the stack overflow and trace depth exceeded exceptions are not handled here, but
/// directly in rtcore. For user exceptions we only check the range of the exception code, the
/// intrinsic is implemented by the RTX runtime.
class RTXExceptionInstrumenter
{
  public:
    /// \param exceptionFlags       The exception flags obtained from the context (knobs will be
    ///                             taken into account separately).
    /// \param maxPayloadSize       The maximum payload size taking all programs in the plan into
    ///                             account.
    /// \param payloadInRegisters   Indicates whether the payload lives in memory or has been
    ///                             promoted to registers.
    RTXExceptionInstrumenter( SemanticType stype, uint64_t exceptionFlags, uint64_t maxPayloadSize, bool payloadInRegisters, int launchCounterForDebugging );

    void runOnFunction( llvm::Function* entryFunction );

    /// Used by RTXCompile
    ///
    /// \param exceptionFlags   The exception flags with knobs already taken into account.
    static bool hasProductSpecificExceptionsEnabled( SemanticType stype, uint64_t exceptionFlags );

  private:
    /// Initialize the pointers to runtime functions - creating them if necessary.
    ///
    void initializeRuntimeFunctions();

    /// Find or create a runtime function of the given type.
    llvm::Function* findRuntimeFunction( llvm::Module*                   module,
                                         const std::string&              name,
                                         llvm::Type*                     returnType,
                                         const std::vector<llvm::Type*>& argTypes );

    /// \name Transforms the corresponding optixi intrinsic into RTX intrinsics, typically
    ///       RtxiGetBufferId and another RTX intrinsic doing the real work based on the ID.
    ///       If enabled, checks for buffer ID and buffer index out of bounds are inserted.
    //@{

    void transformGetBufferSize( llvm::Function* F );

    void transformGetBufferElement( llvm::Function* F );

    void transformSetBufferElement( llvm::Function* F );

    void transformGetBufferElementAddress( llvm::Function* F );

    void transformAtomicSetBufferElement( llvm::Function* F );

    void transformLoadOrRequestBufferElement( llvm::Function* function );

    void transformLoadOrRequestTextureElement( llvm::Function* function );

    void transformGetBufferSizeFromId( llvm::Function* F );

    void transformGetBufferElementFromId( llvm::Function* F );

    void transformSetBufferElementFromId( llvm::Function* F );

    void transformGetBufferElementAddressFromId( llvm::Function* F );

    void transformAtomicSetBufferElementFromId( llvm::Function* F );

    //@
    /// \name Inserts checks for the various exception types
    //@{

    void insertTextureIdIlwalidCheck();

    void insertTextureIdIlwalidCheck( llvm::Function* toInstrument, llvm::Function* checkFunc );

    void insertProgramIdIlwalidCheck();

    void insertProgramIdIlwalidCheck( llvm::Function* toInstrument, llvm::Function* checkFunc );

    void insertBufferIdIlwalidCheck( llvm::Value* statePtr, llvm::Value* bufferId, llvm::Instruction* insertBefore );

    void insertIlwalidIdCheck( RTexception exception, llvm::Function* checkFunc, llvm::Value* statePtr, llvm::Value* id, llvm::Instruction* insertBefore );

    void insertBufferIndexOutOfBoundsCheck( llvm::Value*       statePtr,
                                            llvm::Value*       bufferId,
                                            int                dimensions,
                                            llvm::Value*       elementSize,
                                            llvm::Value*       indices[3],
                                            const std::string& bufferName,
                                            llvm::Instruction* insertBefore );

    void insertIlwalidRayCheck();

    void insertIlwalidRayCheck( llvm::CallInst* traceCall );

    void insertIndexOutOfBoundsCheck();

    void insertExceptionCodeOutOfBoundsCheck();

    void insertPayloadAccessOutOfBoundsCheck();

    //@}
    /// \name Indicates whether the various exception types are enabled (knobs override context).
    //@{

    bool textureIdIlwalidEnabled() const;

    bool programIdIlwalidEnabled() const;

    bool bufferIdIlwalidEnabled() const;

    bool bufferIndexOutOfBoundsEnabled() const;

    bool ilwalidRayEnabled() const;

    bool indexOutOfBoundsEnabled() const;

    bool payloadOffsetOutOfBoundsEnabled() const;

    bool exceptionEnabled( bool exceptionKnob, RTexception exception ) const;

    //@}

    /// Returns the buffer details for \p variableName and \p instruction.
    ///
    /// Buffer details are the buffer name and the source location. The buffer name is extracted
    /// from the variable name (if not empty). If neither buffer name nor source location are
    /// available, the string "n/a (ilwoke lwcc with the -lineinfo option)" is returned.
    ///
    /// The string is returned as i64 value that can be casted to "const char*" and passed as %s
    /// argument to rtPrintf().
    llvm::Value* getBufferDetails( const std::string& variableName, llvm::Instruction* instruction );

    /// Returns the source location of \p instruction from the debug metadata.
    ///
    /// If no exact source location is available, an approximate source location is considered. If no
    /// approximate source location is available either, the string "n/a (ilwoke lwcc with the
    /// -lineinfo option, or no useful information for that block present)" if returned.
    ///
    /// The string is returned as i64 value that can be casted to "const char*" and passed as %s
    /// argument to rtPrintf().
    llvm::Value* getSourceLocation( llvm::Instruction* instruction );

    /// Returns an approximate source location of \p instruction from the debug metadata.
    ///
    /// If no exact source location can be found (not available or the filename is blacklisted),
    /// an approximate source location is returned. Both are formatted as filename:line:column, plus
    /// an "(approximately)" as suffix in the latter case. If no approximate source location is
    /// found, the empty string is returned.
    ///
    /// The approximate source location is found by inspecting predecessors and successors of
    /// \p instruction within the same basic block.
    std::string getApproximateSourceLocationAsString( llvm::Instruction* instruction );

    /// Returns the exact source location of \p instruction from the debug metadata.
    ///
    /// The source location is formatted as filename:line:column. If no source location is found,
    /// the empty string is returned.
    std::string getExactSourceLocationAsString( llvm::Instruction* instruction );

    /// Indicates whether the given \p filename is blacklisted for source locations.
    ///
    /// Black listed are those filenames from OptiX headers that would otherwise appear as source
    /// locations due to inlining, e.g., optix_internal.h due to rt_trace(). These locations are
    /// not helpful for the user. Instead, an approximate location from their own .lw files is much
    /// more helpful.
    static bool filenameIsBlacklisted( const std::string& filename );

    /// Indicates whether the given \p directory is blacklisted for source locations.
    ///
    /// Black listed is the artificial directory that is used for generated source locations. These
    /// locations are not helpful for the user. Instead, an approximate location from their own .lw
    /// files is much more helpful.
    static bool directoryIsBlacklisted( const std::string& directory );

    /// Dump the module if the knob rtx.saveLLVM is enabled.
    void dump( const std::string& functionName, int dumpId, const std::string& suffix );

    llvm::Function*    m_entryFunction;
    llvm::Module*      m_module;
    const SemanticType m_stype;
    const uint64_t     m_exceptionFlags;
    const uint64_t     m_maxPayloadSize;
    const bool         m_payloadInRegisters;
    const int          m_launchCounterForDebugging;

    /// Map used by getBufferDetails() and getSource Location() to avoid repeated generation of
    /// identical strings.
    std::map<std::string, llvm::Value*> m_stringsCache;

    /// Function pointers to runtime functions used during rewrite.
    llvm::Function* m_checkTextureIdFunc      = nullptr;
    llvm::Function* m_checkBufferIdFunc       = nullptr;
    llvm::Function* m_checkProgramIdFunc      = nullptr;
    llvm::Function* m_bufferGetSizeFunc       = nullptr;
    llvm::Function* m_getGeometryInstanceFunc = nullptr;
    llvm::Function* m_getNumMaterialsFunc     = nullptr;
    llvm::Function* m_throwExceptionFunc      = nullptr;

    llvm::Function* m_throwBufferIndexOutOfBoundsException   = nullptr;
    llvm::Function* m_throwExceptionCodeOutOfBoundsException = nullptr;
    llvm::Function* m_throwIlwalidIdException                = nullptr;
    llvm::Function* m_throwIlwalidRayException               = nullptr;
    llvm::Function* m_throwMaterialIndexOutOfBoundsException = nullptr;
    llvm::Function* m_throwPayloadAccessOutOfBoundsException = nullptr;
};

}  // namespace optix
