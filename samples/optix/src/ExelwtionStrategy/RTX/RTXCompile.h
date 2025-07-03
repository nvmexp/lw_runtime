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

#include <ExelwtionStrategy/Common/ConstantMemAllocations.h>
#include <ExelwtionStrategy/Common/Specializations.h>
#include <FrontEnd/Canonical/CallSiteIdentifier.h>
#include <Memory/DemandLoad/PagingMode.h>
#include <Objects/SemanticType.h>

#include <rtcore/interface/types.h>

#include <map>
#include <memory>
#include <set>
#include <string>

namespace llvm {
class BasicBlock;
class CallInst;
class Function;
class Module;
class Type;
class Value;
class Instruction;
}

namespace optix {
class CanonicalProgram;
class PersistentStream;
class ProfileManager;
class ProgramManager;
class LWDADevice;

class RTXCompile
{
  public:
    struct ConstantMemAllocationFlags
    {
        //
        // WARNING: This is a persistent class. If you change anything you
        // should also update the readOrWrite function.
        //
        ConstantMemAllocationFlags( const ConstantMemAllocations& allocs )
        {
            objectRecordInConstMemory = allocs.objectRecordSize > 0;
            bufferTableInConstMemory  = allocs.bufferTableSize > 0;
            programTableInConstMemory = allocs.programTableSize > 0;
            textureTableInConstMemory = allocs.textureTableSize > 0;
        }

        ConstantMemAllocationFlags() {}

        bool objectRecordInConstMemory = false;
        bool bufferTableInConstMemory  = false;
        bool programTableInConstMemory = false;
        bool textureTableInConstMemory = false;

        bool operator!=( const ConstantMemAllocationFlags& other ) const;
        bool operator<( const ConstantMemAllocationFlags& other ) const;
    };

    struct CompileParams
    {
        //
        // WARNING: This is a persistent class. If you change anything you
        // should also update the readOrWrite function and bump the the
        // version number.
        //
        bool                       payloadInRegisters = false;
        ConstantMemAllocationFlags constMemAllocFlags;

        int numCallableParamRegisters = 0;

        bool forceInlineUserFunctions = true;

        bool addLimitIndicesCheck = false;

        // Exception flags (bitmask as in Context::getExceptionFlags(), but including knobs).
        uint64_t exceptionFlags = 0;

        // The maximum payload size (in bytes) over all programs.
        uint64_t maxPayloadSize = 0;

        // Indicates whether the payload size is propagated through trace calls.
        bool propagatePayloadSize = false;

        int maxAttributeRegisterCount = 0;

        bool operator!=( const CompileParams& other ) const;
        bool operator<( const CompileParams& other ) const;
    };
    struct SizeOffsetPair
    {
        //
        // WARNING: This is a persistent class. If you change anything you
        // should also update the readOrWrite function and bump the the
        // version number.
        //
        SizeOffsetPair( int size, int offset, bool memory )
            : size( size )
            , offset( offset )
            , memory( memory )
        {
        }
        SizeOffsetPair() {}

        int  size   = 0;
        int  offset = 0;
        bool memory = false;

        bool operator==( const SizeOffsetPair& other ) const;
        bool operator<( const SizeOffsetPair& other ) const;
    };

    struct AttributeDecoderData
    {
        llvm::Function* decoder;
        int             attributeKind;
        SemanticType    stype;
    };
    typedef std::vector<AttributeDecoderData> AttributeDecoderList;

    struct Options
    {
        const LWDADevice*                    device;
        CompileParams                        compileParams;
        RtcCompileOptions                    rtcoreCompileOptions;
        std::vector<const CanonicalProgram*> attributeDecoders;
        std::set<std::string>                heavyweightCallSiteNames;
        Specializations                      specializations;
        SemanticType                         stype;
        SemanticType                         inheritedStype;
        int                                  numConlwrrentLaunchDevices;
        PagingMode                           pagingMode;
        const CanonicalProgram*              cp;
    };

    RTXCompile( const Options& options, const AttributeDecoderList& attributeDecoders, const ProgramManager* programManager, int launchCounterForDebugging );
    ~RTXCompile();

    // Compile the given function so that it implements the rtcore interface.
    // Returns the new name of the function: it is going to be mangled based on the semantic type.
    std::string runOnFunction( llvm::Function* function, bool& fellBackToLWPTX );

    /// Link PTX wrapper libraries
    static void linkPTXFrontEndIntrinsics( llvm::Module* module, bool useD2IR, bool enableLWPTXFallback, bool& fellBackToLWPTX );

    // Constant used to specify that we shouldn't perform device count specialization when supplied as
    // numConlwrrentLaunchDevices parameter.
    static const int DONT_SPECIALIZE_NUM_DEVICES = 0;

  private:
    void optimizeModule( llvm::Module* module ) const;

    /// Adds a check to the beginning of the given function that exits if the current launch index is
    /// outside of those specified via the launch.limitActiveIndices knob.
    void addLimitIndicesCheck( llvm::Module* module, llvm::Function* function );


    /// Changes the function signature and adapts the function body accordingly.
    ///
    /// For bounding box programs and callable programs, the canonical state parameter is removed
    /// from the signature. All parameters types and the return type are flattened into an array of
    /// i32. For other programs, all parameters are removed.
    ///
    /// If a bounding box program or callable program uses more than
    /// m_compileOptions.numCallableParamRegisters flattened i32 arguments or return values, the
    /// last two registers are used for the pointer to the spill buffer. In case of argument
    /// spilling, the unspilled and spilled arguments are combined into one aggregate value for
    /// simpler unflattening. In case of return value spilling, all but the first
    /// m_compileOptions.numCallableParamRegisters-2 return values are spilled.
    ///
    /// This pass is run last, after optimizing the module, to ensure that the canonical state (and
    /// other parameters for programs other than bounding box and callable programs) are no longer
    /// used.
    void changeFunctionSignature( llvm::Function* function ) const;

    /// Transforms bounding box programs.
    ///
    /// The signature is changed to (CanonicalState*, i32 gi, i32 primitive, i32 motion, i64 aabb).
    /// All callers of cort::getGeometryInstanceHandle() are inlined and the calls are replaced by
    /// the "gi" argument. All calls to optixi_getPrimitiveArgToComputeAABB() are replaced by the
    /// "primitive" argument. All calls to optixi_getMotionIndexArgToComputeAABB() are replaced by
    /// the "motion" argument. All calls to optixi_getAABBArgToComputeAABB() are replaced by the
    /// "aabb" argument casted to float*. Returns the new function.
    llvm::Function* lowerBoundingBoxProgram( llvm::Function* function ) const;

    /// Adds a check to the given raygen function that ensures the function is not launching outside
    /// of the dimensions of the launch.
    void addLaunchBoundsCheck( llvm::Module* module, llvm::Function* function ) const;

    /// Transforms AABB iterator programs.
    ///
    /// All calls to optixi_computeAABB() in the given function (typically exactly one) are replaced
    /// by a call to RTX_computeAABB().
    ///
    /// In addition, the call to RTX_computeAABB_BoundingBoxProgramStub() in the runtime function
    /// RTX_computeAABB_BoundingBoxProgram() is replaced by a call to the corresponding bounding box
    /// program and lowered using lowerCalls().
    void lowerAABBIteratorProgram( llvm::Function* function ) const;

    /// Packs the incoming payload such that it can be passed to the rtcore trace call.
    ///
    /// \param localPayload                       Indicates whether the incoming payload resides in
    ///                                           a local variable or is passed in through another
    ///                                           trace call ("global").
    /// \param payloadSize                        The size of the payload in bytes.
    /// \param payloadArg                         The incoming payload (used if \p localPayload is
    ///                                           \c true).
    /// \param lwvmReadPayload                    Function to read the incoming payload (used if
    ///                                           \p localPayload is \c false).
    /// \param[out] numPayloadRegisters           Number of registers used for the packed payload
    ///                                           or payload pointer.
    /// \param[out] numPayloadPlusSizeRegisters   Number of registers used for the packed payload
    ///                                           payload pointer plus payload size.
    /// \param[out] payloadValue                  The packed payload.
    /// \param[out] payloadTy                     The type of the packed payload.
    void packPayload( bool              localPayload,
                      unsigned int      payloadSize,
                      llvm::Value*      payloadArg,
                      llvm::Function*   lwvmReadPayload,
                      unsigned int&     numPayloadRegisters,
                      unsigned int&     numPayloadPlusSizeRegisters,
                      llvm::Value*&     payloadValue,
                      llvm::Type*&      payloadTy,
                      llvm::BasicBlock* insertBefore ) const;

    /// Unpacks the payload from an rtcore trace call.
    ///
    /// \param localPayload                       Indicates whether the incoming payload resides in
    ///                                           a local variable or is passed in through another
    ///                                           trace call ("global").
    /// \param numPayloadRegisters                Number of registers used for the packed payload
    ///                                           or payload pointer.
    /// \param result                             The payload to be unpacked.
    /// \param payloadArg                         The outgoing payload (used if \p localPayload is
    ///                                           \c true).
    /// \param lwvmWritePayload                   Function to write the outgoing payload (used if
    ///                                           \p localPayload is \c false).
    void unpackPayload( bool              localPayload,
                        unsigned int      numPayloadRegisters,
                        llvm::Value*      result,
                        llvm::Value*      payloadArg,
                        llvm::Function*   lwvmWritePayload,
                        llvm::BasicBlock* insertBefore ) const;

    void lowerTrace( llvm::Module* module ) const;

    /// Transforms bound callable programs.
    ///
    /// Bound callable programs receive (potentially) two additional parameters:
    /// the caller's SBT pointer for scoped variable lookup and a struct that
    /// contains data to allow bound callable programs access to the payload,
    /// the transformations and the attributes. Bound callable programs
    /// called from raygen or a bindless callable program do not get the
    /// additional struct.
    llvm::Function* lowerBoundCallableProgram( llvm::Function* function ) const;
    void rewriteBoundCallableProgramParentSbtPointer( llvm::Function* function ) const;
    void loadValuesFromBcpStateAndReplaceCalls( llvm::Function* module ) const;

    /// Ilwokes lowerCalls() for all callable programs.
    void lowerCallableProgramCalls( llvm::Module* module ) const;

    /// Replaces all calls to function F by calls to a function named rtcoreName.
    ///
    /// In addition, parameter and return value spilling is done. If more than
    /// m_compileOptions.numCallableParamRegisters flattened i32 arguments or return values are
    /// used, a spill buffer is allocated and all but the first
    /// m_compileOptions.numCallableParamRegisters-2 arguments are spilled. The last two registers
    /// are used for the pointer to the spill buffer. In case of return value spilling, the
    /// unspilled and spilled return values are combined into one aggregate value for simpler
    /// unflattening.
    void lowerCalls( llvm::Module* module, llvm::Function& F, const std::string& rtcoreName, bool isBound ) const;

    void lowerAttributesForCHandAH( llvm::Function* function ) const;

    void lowerAttributesForBoundCallableProgram( llvm::Module*      module,
                                                 unsigned short     token,
                                                 llvm::Value*       alloca,
                                                 llvm::Instruction* insertBefore,
                                                 int                totalNumAttrRegs,
                                                 int                totalNumAttrMemRegs ) const;

    void lowerGetAttributeData( llvm::Function* function ) const;
    void lowerIsPotentialIntersection( llvm::Function* function ) const;
    void lowerReportFullIntersection( llvm::Function* function ) const;

    void lowerExceptionDetails( llvm::Module* module ) const;

    void lowerGetLwrrentRay( llvm::Module* module ) const;

    void lowerPayloadGetAndSet( llvm::Module* module ) const;

    void rewriteTableAccessesToUseConstMemory( llvm::Module* module, const ConstantMemAllocationFlags& constMemAllocFlags ) const;

    void replaceGlobalDeviceCount( llvm::Module* module, int deviceCount ) const;

    void addReturnsForExceptionThrow( llvm::Module* module, llvm::Function* function ) const;

    void dump( llvm::Module* module, const std::string& functionName, int dumpId, const std::string& suffix ) const;

    // Bound callable programs receive additional data they may need through
    // a struct called "BoundCallableProgramState". The layout of that struct
    // is determined dynamically based on the semantic type and the
    // attribute assignments. It is filled by the caller of a bound
    // callable program.
    // The out parameter bcpStateAttributeMappings maps the token of the attribute
    // as stored in the attribute assignments to its index in the
    // BoundCallableProgramState and its name.
    llvm::Type* getBoundCallableProgramStateType( llvm::Module* module,
                                                  std::map<unsigned short, std::pair<int, std::string>>& bcpStateAttributeMappings ) const;
    llvm::Value* fillBoundCallableProgramState( llvm::Function& callable, llvm::Instruction* insertBefore ) const;

    AttributeDecoderList m_attributeDecoders;

    // Calls to bindless callable programs from call sites identified with these
    // names are compiled as continuation calls.
    std::set<std::string> m_heavyWeightCallSites;

    SemanticType          m_stype          = ST_ILWALID;
    SemanticType          m_inheritedStype = ST_ILWALID;
    CompileParams         m_params;
    const ProgramManager* m_programManager;
    int                   m_launchCounterForDebugging  = 0;
    int                   m_numConlwrrentLaunchDevices = 0;
    PagingMode            m_pagingMode                 = PagingMode::UNKNOWN;

    // Replace references to demand load paging mode with constants in
    // the given module.
    void replacePagingMode( llvm::Module* module );

    // NOTE: SBT record offset and stride have only 4 bit precision, but LWVM
    // does not support i4, so we use 8 bits.
    static const int SBT_RECORD_OFFSET_NBITS = 8;  // = REGPACK_SBT_RECORD_OFFSET_NBITS
    static const int SBT_RECORD_STRIDE_NBITS = 8;  // = REGPACK_SBT_RECORD_STRIDE_NBITS

    // Whether or not we should use D2IR.
    bool m_useD2IR = false;
};

// Persistence support
void readOrWrite( PersistentStream* stream, RTXCompile::ConstantMemAllocationFlags* flags, const char* label );
void readOrWrite( PersistentStream* stream, RTXCompile::CompileParams* params, const char* label );
void readOrWrite( PersistentStream* stream, RTXCompile::SizeOffsetPair* so, const char* label );
}
