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

#include <corelib/compiler/CoreIRBuilder.h>

#include <FrontEnd/Canonical/IntrinsicsManager.h>

namespace optix {

/// General note to all RTX intrinsics.
///
/// The RTX instrinsics can be divided into four groups of instrinsics.
///
///    RtxiGetBufferId
///
///    RtxiGetBufferSize
///    RtxiGetBufferElement
///    RtxiSetBufferElement
///    RtxiGetBufferElementAddress
///    RtxiAtomicSetBufferElement
///
///    RtxiGetBufferSizeFromId
///    RtxiGetBufferElementFromId
///    RtxiSetBufferElementFromId
///    RtxiGetBufferElementAddressFromId
///    RtxiAtomicSetBufferElementFromId
///
/// The first one, RtxiGetBufferId, is used as first step to obtain the buffer ID which is a
/// parameter to all instrinsics in the second and third group.
///
/// The intrinsics in the second and third group are the RTX counterparts to the corresponding
/// optixi intrinsics. Since the methods in the second group also have the buffer ID as parameter,
/// the methods in both groups are very similar. One difference is the function name. The name of
/// the methods from the second group contains the mangled variable name, whereas the names of the
/// methods from the third group don't.

/// RTX intrinsic
///
///     i32 rtxiGetBufferId.<varRefUniqueName>(%"struct.cort::CanonicalState"*)
class RtxiGetBufferId : public OptixIntrinsic
{
  public:
    std::string getVarRefUniqueName();

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }

    static std::string getFunctionName( const std::string& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 1;

    enum Args
    {
        StatePtr = 0,
        END
    };

    friend class RtxiGetBufferIdBuilder;
};

class RtxiGetBufferIdBuilder
{
  public:
    RtxiGetBufferIdBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName );
    llvm::FunctionType* createType();

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     size3 rtxiGetBufferSize.<varRefUniqueName>(%"struct.cort::CanonicalState"*, i32 bufferId)
class RtxiGetBufferSize : public OptixIntrinsic
{
  public:
    std::string getVarRefUniqueName();

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }

    static std::string getFunctionName( const std::string& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 2;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        END
    };

    friend class RtxiGetBufferSizeBuilder;
};

class RtxiGetBufferSizeBuilder
{
  public:
    RtxiGetBufferSizeBuilder( llvm::Module* module, llvm::Type* size3Ty, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName );
    llvm::FunctionType* createType();

    llvm::Module*          m_module;
    llvm::Type*            m_size3Ty;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr = nullptr;
    llvm::Value* m_bufferId = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     <elementType> rtxiGetBufferElement.<varRefUniqueName>.<elementTypeName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiGetBufferElement : public OptixIntrinsic
{
  public:
    std::string  getVarRefUniqueName();
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( const std::string& varRefUniqueName, const std::string& elementTypeName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiGetBufferElementBuilder;
};

class RtxiGetBufferElementBuilder
{
  public:
    RtxiGetBufferElementBuilder( llvm::Module* module, llvm::Type* returnTy, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName, int dimensionality );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName, int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_returnTy;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_offset      = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
};

// This class models the output of RTXExceptionInstrumenter, which transforms
// the LoadOrRequestBufferElement intrinsic from C14n into the RtxiLoadOrRequestBufferElement
// intrinsic.
//
//     i8* rtxiLoadOrRequestBufferElement.<varRefUniqueName>(
//         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 ptr, i64 x, i64 y, i64 z )
//
// The last two arguments are optional depending on the dimensionality.  Because the
// dimensionality of the access is associated with the dimensionality of the corresponding
// buffer, there is no need to differentiate this function beyond it's unique variable
// name reference.
//
class RtxiLoadOrRequestBufferElement : public OptixIntrinsic
{
  public:
    std::string  getVarRefUniqueName();
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getPtr() const { return getArgOperand( Ptr ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( const std::string& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Ptr,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiLoadOrRequestBufferElementBuilder;
};

class RtxiLoadOrRequestBufferElementBuilder
{
  public:
    RtxiLoadOrRequestBufferElementBuilder( llvm::Module* module, llvm::Type* returnTy, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setPtr( llvm::Value* ptr ) { m_ptr = ptr; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName, int dimensionality );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName, int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_returnTy;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
    llvm::Value* m_ptr         = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     void rtxiSetBufferElement.<varRefUniqueName>.<elementTypeName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset, i64 x, i64 y, i64 z,
///         <elementType> valueToSet)
///
/// The second-last and third-last arguments are optional depending on the dimensionality.
class RtxiSetBufferElement : public OptixIntrinsic
{
  public:
    std::string  getVarRefUniqueName();
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getValueToSet() const { return getArgOperand( Offset + getDimensionality() + 1 ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( const std::string& varRefUniqueName, const std::string& elementTypeName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 6;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        X,
        Y,
        Z,
        ValueToSet,
        END
    };

    friend class RtxiSetBufferElementBuilder;
};

class RtxiSetBufferElementBuilder
{
  public:
    RtxiSetBufferElementBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Type* valueToSetTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }
    void setValueToSet( llvm::Value* valueToSet ) { m_valueToSet = valueToSet; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName, int dimensionality );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName, int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    llvm::Type*            m_valueToSetTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_offset      = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
    llvm::Value* m_valueToSet  = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     i64 rtxiGetBufferElementAddress.<varRefUniqueName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiGetBufferElementAddress : public OptixIntrinsic
{
  public:
    std::string  getVarRefUniqueName();
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( const std::string& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiGetBufferElementAddressBuilder;
};

class RtxiGetBufferElementAddressBuilder
{
  public:
    RtxiGetBufferElementAddressBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName, int dimensionality );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName, int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_offset      = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     i64 rtxiAtomicSetBufferElement.<varRefUniqueName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset,
///         i32 operation, <opType> compareOperand, <opType> operand, i8 subElementType, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiAtomicSetBufferElement : public OptixIntrinsic
{
  public:
    std::string  getVarRefUniqueName();
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getOperation() const { return getArgOperand( Operation ); }
    llvm::Value* getCompareOperand() const { return getArgOperand( CompareOperand ); }
    llvm::Value* getOperand() const { return getArgOperand( Operand ); }
    llvm::Value* getSubElementType() const { return getArgOperand( SubElementType ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( const std::string& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 9;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        Operation,
        CompareOperand,
        Operand,
        SubElementType,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiAtomicSetBufferElementBuilder;
};

class RtxiAtomicSetBufferElementBuilder
{
  public:
    RtxiAtomicSetBufferElementBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Type* opTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setOperation( llvm::Value* operation ) { m_operation = operation; }
    void setCompareOperand( llvm::Value* compareOperand ) { m_compareOperand = compareOperand; }
    void setOperand( llvm::Value* operand ) { m_operand = operand; }
    void setSubElementType( llvm::Value* subElementType ) { m_subElementType = subElementType; }

    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( const std::string& varRefUniqueName, int dimensionality );

  private:
    llvm::Function* createFunction( const std::string& varRefUniqueName, int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    llvm::Type*            m_opTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr       = nullptr;
    llvm::Value* m_bufferId       = nullptr;
    llvm::Value* m_elementSize    = nullptr;
    llvm::Value* m_offset         = nullptr;
    llvm::Value* m_operation      = nullptr;
    llvm::Value* m_compareOperand = nullptr;
    llvm::Value* m_operand        = nullptr;
    llvm::Value* m_subElementType = nullptr;
    llvm::Value* m_x              = nullptr;
    llvm::Value* m_y              = nullptr;
    llvm::Value* m_z              = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     size3 rtxiGetBufferSizeFromId(%"struct.cort::CanonicalState"*, i32 bufferId)
class RtxiGetBufferSizeFromId : public OptixIntrinsic
{
  public:
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }

    static std::string getFunctionName();

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 2;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        END
    };

    friend class RtxiGetBufferSizeFromIdBuilder;
};

class RtxiGetBufferSizeFromIdBuilder
{
  public:
    RtxiGetBufferSizeFromIdBuilder( llvm::Module* module, llvm::Type* size3Ty, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }

    llvm::CallInst* createCall();

  private:
    llvm::Function*     createFunction();
    llvm::FunctionType* createType();

    llvm::Module*          m_module;
    llvm::Type*            m_size3Ty;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr = nullptr;
    llvm::Value* m_bufferId = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     <elementType> rtxiGetBufferElementFromId.<dimensionality>.<elementTypeName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiGetBufferElementFromId : public OptixIntrinsic
{
  public:
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( int dimensionality, const std::string& elementTypeName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiGetBufferElementFromIdBuilder;
};

class RtxiGetBufferElementFromIdBuilder
{
  public:
    RtxiGetBufferElementFromIdBuilder( llvm::Module* module, llvm::Type* returnTy, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( int dimensionality );

  private:
    llvm::Function* createFunction( int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_returnTy;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_offset      = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     void rtxiSetBufferElementFromId.<dimensionality>.<elementTypeName>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset, i64 x, i64 y, i64 z,
///         <elementType> valueToSet)
///
/// The second-last and third-last arguments are optional depending on the dimensionality.
class RtxiSetBufferElementFromId : public OptixIntrinsic
{
  public:
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getValueToSet() const { return getArgOperand( Offset + getDimensionality() + 1 ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( int dimensionality, const std::string& elementTypeName );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 6;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        X,
        Y,
        Z,
        ValueToSet,
        END
    };

    friend class RtxiSetBufferElementFromIdBuilder;
};

class RtxiSetBufferElementFromIdBuilder
{
  public:
    RtxiSetBufferElementFromIdBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Type* valueToSetTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }
    void setValueToSet( llvm::Value* valueToSet ) { m_valueToSet = valueToSet; }

    llvm::CallInst* createCall( int dimensionality );

  private:
    llvm::Function* createFunction( int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    llvm::Type*            m_valueToSetTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_offset      = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
    llvm::Value* m_valueToSet  = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     i64 rtxiGetBufferElementAddressFromId.<dimensionality>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiGetBufferElementAddressFromId : public OptixIntrinsic
{
  public:
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( int dimensionality );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 4;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiGetBufferElementAddressFromIdBuilder;
};

class RtxiGetBufferElementAddressFromIdBuilder
{
  public:
    RtxiGetBufferElementAddressFromIdBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( int dimensionality );

  private:
    llvm::Function* createFunction( int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr    = nullptr;
    llvm::Value* m_bufferId    = nullptr;
    llvm::Value* m_elementSize = nullptr;
    llvm::Value* m_x           = nullptr;
    llvm::Value* m_y           = nullptr;
    llvm::Value* m_z           = nullptr;
};

/// -----------------------------------------------------------------------------------------------

/// RTX intrinsic
///
///     i64 rtxiAtomicSetBufferElementFromId.<dimensionality>(
///         %"struct.cort::CanonicalState"*, i32 bufferId, i32 elementSize, i64 offset,
///         i32 operation, <opType> compareOperand, <opType> operand, i8 subElementType, i64 x, i64 y, i64 z)
///
/// The last two arguments are optional depending on the dimensionality.
class RtxiAtomicSetBufferElementFromId : public OptixIntrinsic
{
  public:
    unsigned int getDimensionality() const;

    llvm::Value* getStatePtr() const { return getArgOperand( StatePtr ); }
    llvm::Value* getBufferId() const { return getArgOperand( BufferId ); }
    llvm::Value* getElementSize() const { return getArgOperand( ElementSize ); }
    llvm::Value* getOffset() const { return getArgOperand( Offset ); }
    llvm::Value* getOperation() const { return getArgOperand( Operation ); }
    llvm::Value* getCompareOperand() const { return getArgOperand( CompareOperand ); }
    llvm::Value* getOperand() const { return getArgOperand( Operand ); }
    llvm::Value* getSubElementType() const { return getArgOperand( SubElementType ); }
    llvm::Value* getX() const { return getArgOperand( X ); }
    llvm::Value* getY() const { return getArgOperand( Y ); }
    llvm::Value* getZ() const { return getArgOperand( Z ); }
    llvm::Value* getIndex( unsigned int dimension ) const { return getArgOperand( X + dimension ); }

    static std::string getFunctionName( int dimenstionality );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef functionName );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 9;

    enum Args
    {
        StatePtr = 0,
        BufferId,
        ElementSize,
        Offset,
        Operation,
        CompareOperand,
        Operand,
        SubElementType,
        X,
        Y,
        Z,
        END
    };

    friend class RtxiAtomicSetBufferElementFromIdBuilder;
};

class RtxiAtomicSetBufferElementFromIdBuilder
{
  public:
    RtxiAtomicSetBufferElementFromIdBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Type* opTy, llvm::Instruction* insertBefore );

    void setStatePtr( llvm::Value* statePtr ) { m_statePtr = statePtr; }
    void setBufferId( llvm::Value* bufferId ) { m_bufferId = bufferId; }
    void setElementSize( llvm::Value* elementSize ) { m_elementSize = elementSize; }
    void setOffset( llvm::Value* offset ) { m_offset = offset; }
    void setOperation( llvm::Value* operation ) { m_operation = operation; }
    void setCompareOperand( llvm::Value* compareOperand ) { m_compareOperand = compareOperand; }
    void setOperand( llvm::Value* operand ) { m_operand = operand; }
    void setSubElementType( llvm::Value* subElementType ) { m_subElementType = subElementType; }

    void setX( llvm::Value* x ) { m_x = x; }
    void setY( llvm::Value* y ) { m_y = y; }
    void setZ( llvm::Value* z ) { m_z = z; }

    llvm::CallInst* createCall( int dimensionality );

  private:
    llvm::Function* createFunction( int dimensionality );
    llvm::FunctionType* createType( int dimensionality );

    llvm::Module*          m_module;
    llvm::Type*            m_statePtrTy;
    llvm::Type*            m_opTy;
    corelib::CoreIRBuilder m_builder;

    llvm::Value* m_statePtr       = nullptr;
    llvm::Value* m_bufferId       = nullptr;
    llvm::Value* m_elementSize    = nullptr;
    llvm::Value* m_offset         = nullptr;
    llvm::Value* m_operation      = nullptr;
    llvm::Value* m_compareOperand = nullptr;
    llvm::Value* m_operand        = nullptr;
    llvm::Value* m_subElementType = nullptr;
    llvm::Value* m_x              = nullptr;
    llvm::Value* m_y              = nullptr;
    llvm::Value* m_z              = nullptr;
};

}  // namespace optix
