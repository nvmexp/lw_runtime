//
// Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
#pragma once

#include <FrontEnd/Canonical/CanonicalProgramID.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Regex.h>

#include <string>

namespace optix {

class LLVMManager;
class ProgramManager;
class VariableReference;

const std::string GET_BUFFER_SIZE_ID = "optixi_getBufferSizeFromId";
const std::string GET_BUFFER_SIZE    = "optixi_getBufferSize";
const std::string GET_PAYLOAD_PREFIX = "optixi_getPayloadValue";
const std::string SET_PAYLOAD_PREFIX = "optixi_setPayloadValue";

bool isOptixIntrinsic( const llvm::Function* function );

bool isOptixAtomicIntrinsic( const llvm::Function* function );

unsigned short int getOptixAtomicToken( const llvm::Function* function, const ProgramManager* programManager );

bool isPayloadGet( const llvm::Function* function );
bool isPayloadSet( const llvm::Function* function );

// -----------------------------------------------------------------------------
class OptixIntrinsic : public llvm::CallInst
{
  public:
    // Objects of type OptixIntrinsic should only be create by casting from CallInst, never directly.
    OptixIntrinsic()                        = delete;
    OptixIntrinsic( const OptixIntrinsic& ) = delete;
    OptixIntrinsic( OptixIntrinsic&& )      = delete;
    OptixIntrinsic& operator=( const OptixIntrinsic& ) = delete;
    OptixIntrinsic& operator=( OptixIntrinsic&& ) = delete;

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static const std::string PREFIX;
};

// -----------------------------------------------------------------------------
class AtomicSetBufferElement : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    llvm::Value* getOffset() const;
    unsigned     getDimensionality() const;

    llvm::Value* getOperation();
    llvm::Value* getCompareOperand();  // Specific for CAS.
    llvm::Value* getOperand();
    llvm::Value* getSubElementType();
    std::string  parseUniqueName() const;

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static std::string parseUniqueName( llvm::StringRef name );
    static std::string createUniqueName( const VariableReference* vref );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 8;

    enum Args
    {
        CanonicalState = 0,
        Operation,
        CompareOperand,
        Operand,
        ElementSize,
        SubElementType,
        Offset,
        x,
        y,
        z,
        END
    };

    friend class AtomicSetBufferElementBuilder;
};

class AtomicSetBufferElementBuilder
{
  public:
    AtomicSetBufferElementBuilder( llvm::Module* module );

    llvm::CallInst* create( llvm::Function* function, llvm::Instruction* insertBefore );
    llvm::Constant* createFunction( llvm::Type* opType, VariableReference* varRef, unsigned dimensions, LLVMManager* llvmManager );
    AtomicSetBufferElementBuilder& setCanonicalState( llvm::Value* canonicalState );
    AtomicSetBufferElementBuilder& setOperation( llvm::Value* operation );
    AtomicSetBufferElementBuilder& setCompareOperand( llvm::Value* compareOperand );
    AtomicSetBufferElementBuilder& setOperand( llvm::Value* operand );
    AtomicSetBufferElementBuilder& setElementSize( llvm::Value* elementSize );
    AtomicSetBufferElementBuilder& setSubElementType( llvm::Value* subElementType );
    AtomicSetBufferElementBuilder& setOffset( llvm::Value* offset );
    AtomicSetBufferElementBuilder& setX( llvm::Value* x );
    AtomicSetBufferElementBuilder& setY( llvm::Value* y );
    AtomicSetBufferElementBuilder& setZ( llvm::Value* z );

    static unsigned getDimensionality( const llvm::Function* function );

  private:
    llvm::FunctionType* createType( llvm::Type* opType, unsigned dimensions, LLVMManager* llvmManager );

  private:
    llvm::Module* m_module = nullptr;

    llvm::Value* m_canonicalState = nullptr;
    llvm::Value* m_operation      = nullptr;
    llvm::Value* m_compareOperand = nullptr;
    llvm::Value* m_operand        = nullptr;
    llvm::Value* m_elementSize    = nullptr;
    llvm::Value* m_subElementType = nullptr;
    llvm::Value* m_offset         = nullptr;
    llvm::Value* m_x              = nullptr;
    llvm::Value* m_y              = nullptr;
    llvm::Value* m_z              = nullptr;
};

// -----------------------------------------------------------------------------
class AtomicSetBufferElementFromId : public OptixIntrinsic
{
  public:
    unsigned getDimensionality();

    llvm::Value* getStatePtr();
    llvm::Value* getBufferId();
    llvm::Value* getOperation();
    llvm::Value* getCompareOperand();  // Specific for CAS.
    llvm::Value* getOperand();
    llvm::Value* getElementSize();
    llvm::Value* getOffset();
    llvm::Value* getX();
    llvm::Value* getY();
    llvm::Value* getZ();
    llvm::Value* getIndex( unsigned dimension );
    llvm::Value* getSubElementType();

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static std::string createUniqueName( unsigned int dimensions, size_t elementSize );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 9;

    enum Args
    {
        CanonicalState = 0,
        BufferId,
        Operation,
        CompareOperand,
        Operand,
        ElementSize,
        SubElementType,
        Offset,
        x,
        y,
        z,
        END
    };

    friend class AtomicSetBufferElementFromIdBuilder;
};

class AtomicSetBufferElementFromIdBuilder
{
  public:
    AtomicSetBufferElementFromIdBuilder( llvm::Module* module );

    llvm::CallInst* create( llvm::Function* function, llvm::Instruction* insertBefore );
    llvm::Constant* createFunction( llvm::Type* opType, size_t elementSize, unsigned dimensions, LLVMManager* llvmManager );
    AtomicSetBufferElementFromIdBuilder& setCanonicalState( llvm::Value* canonicalState );
    AtomicSetBufferElementFromIdBuilder& setBufferId( llvm::Value* bufferId );
    AtomicSetBufferElementFromIdBuilder& setOperation( llvm::Value* operation );
    AtomicSetBufferElementFromIdBuilder& setCompareOperand( llvm::Value* compareOperand );
    AtomicSetBufferElementFromIdBuilder& setOperand( llvm::Value* operand );
    AtomicSetBufferElementFromIdBuilder& setElementSize( llvm::Value* elementSize );
    AtomicSetBufferElementFromIdBuilder& setSubElementType( llvm::Value* subElementSize );
    AtomicSetBufferElementFromIdBuilder& setOffset( llvm::Value* offset );
    AtomicSetBufferElementFromIdBuilder& setX( llvm::Value* x );
    AtomicSetBufferElementFromIdBuilder& setY( llvm::Value* y );
    AtomicSetBufferElementFromIdBuilder& setZ( llvm::Value* z );

    static unsigned getDimensionality( const llvm::Function* function );

  private:
    llvm::FunctionType* createType( llvm::Type* opType, unsigned dimensions, LLVMManager* llvmManager );

  private:
    llvm::Module* m_module = nullptr;

    llvm::Value* m_canonicalState = nullptr;
    llvm::Value* m_bufferId       = nullptr;
    llvm::Value* m_operation      = nullptr;
    llvm::Value* m_compareOperand = nullptr;
    llvm::Value* m_operand        = nullptr;
    llvm::Value* m_elementSize    = nullptr;
    llvm::Value* m_subElementType = nullptr;
    llvm::Value* m_offset         = nullptr;
    llvm::Value* m_x              = nullptr;
    llvm::Value* m_y              = nullptr;
    llvm::Value* m_z              = nullptr;
};

// -----------------------------------------------------------------------------
class TraceGlobalPayloadCall : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getNode() const;
    llvm::Value* getOx() const;
    llvm::Value* getOy() const;
    llvm::Value* getOz() const;
    llvm::Value* getDx() const;
    llvm::Value* getDy() const;
    llvm::Value* getDz() const;
    llvm::Value* getRayType() const;
    llvm::Value* getTMin() const;
    llvm::Value* getTMax() const;
    llvm::Value* getTime() const;
    llvm::Value* getHasTime() const;
    llvm::Value* getRayMask() const;
    llvm::Value* getRayFlags() const;
    llvm::Value* getElementSize() const;

    static int getPayloadSize( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static llvm::Regex nameRegex;
    static bool matchName( llvm::StringRef name, llvm::SmallVector<llvm::StringRef, 2>& matches );

    enum Args
    {
        CanonicalState = 0,
        Node,
        Ox,
        Oy,
        Oz,
        Dx,
        Dy,
        Dz,
        RayType,
        Tmin,
        Tmax,
        Time,
        HasTime,
        RayMask,
        RayFlags,
        ElementSize,
        END
    };

    friend class TraceGlobalPayloadBuilder;
};

class TraceGlobalPayloadBuilder
{
  public:
    TraceGlobalPayloadBuilder( llvm::Module* module );

    llvm::CallInst* create( const std::string& cpUUName, int elementSize, LLVMManager* llvmManager, llvm::Instruction* insertBefore );
    TraceGlobalPayloadBuilder& setCanonicalState( llvm::Value* node );
    TraceGlobalPayloadBuilder& setNode( llvm::Value* node );
    TraceGlobalPayloadBuilder& setOx( llvm::Value* ox );
    TraceGlobalPayloadBuilder& setOy( llvm::Value* oy );
    TraceGlobalPayloadBuilder& setOz( llvm::Value* oz );
    TraceGlobalPayloadBuilder& setDx( llvm::Value* dx );
    TraceGlobalPayloadBuilder& setDz( llvm::Value* dy );
    TraceGlobalPayloadBuilder& setDy( llvm::Value* dz );
    TraceGlobalPayloadBuilder& setRayType( llvm::Value* rayType );
    TraceGlobalPayloadBuilder& setTMin( llvm::Value* tMin );
    TraceGlobalPayloadBuilder& setTMax( llvm::Value* tMax );
    TraceGlobalPayloadBuilder& setTime( llvm::Value* t );
    TraceGlobalPayloadBuilder& setHasTime( llvm::Value* t );
    TraceGlobalPayloadBuilder& setRayMask( llvm::Value* );
    TraceGlobalPayloadBuilder& setRayFlags( llvm::Value* );
    TraceGlobalPayloadBuilder& setElementSize( llvm::Value* elementSize );

    static llvm::Type*  getRayFlagsType( llvm::LLVMContext& );
    static llvm::Value* getDefaultRayFlags( llvm::LLVMContext& );
    static llvm::Type*  getRayMaskType( llvm::LLVMContext& );
    static llvm::Value* getDefaultRayMask( llvm::LLVMContext& );

  private:
    llvm::FunctionType* createType( LLVMManager* llvmManager );
    llvm::Function* createFunction( const std::string& name, LLVMManager* llvmManager );

  private:
    llvm::Module* m_module = nullptr;

    llvm::Value* m_canonicalState = nullptr;
    llvm::Value* m_node           = nullptr;
    llvm::Value* m_ox             = nullptr;
    llvm::Value* m_oy             = nullptr;
    llvm::Value* m_oz             = nullptr;
    llvm::Value* m_dx             = nullptr;
    llvm::Value* m_dy             = nullptr;
    llvm::Value* m_dz             = nullptr;
    llvm::Value* m_rayType        = nullptr;
    llvm::Value* m_tMin           = nullptr;
    llvm::Value* m_tMax           = nullptr;
    llvm::Value* m_time           = nullptr;
    llvm::Value* m_hasTime        = nullptr;
    llvm::Value* m_rayMask        = nullptr;
    llvm::Value* m_rayFlags       = nullptr;
    llvm::Value* m_elementSize    = nullptr;
};


// -----------------------------------------------------------------------------
class GetBufferElementAddress : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    llvm::Value* getOffset() const;
    unsigned     getDimensionality() const;
    std::string  parseUniqueName() const;

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );
    static std::string parseUniqueName( llvm::StringRef name );
    static std::string createUniqueName( const VariableReference* varRef );

  private:
    static bool matchName( llvm::StringRef name );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 3;

    enum Args
    {
        CanonicalState = 0,
        ElementSize,
        x,
        y,
        z,
        END
    };

    friend class GetBufferElementAddressBuilder;
};

class GetBufferElementAddressBuilder
{
  public:
    GetBufferElementAddressBuilder( llvm::Module* module );

    unsigned getDimensionality( const llvm::Function* function ) const;

    llvm::CallInst* create( llvm::Function* function, llvm::Instruction* insertBefore );
    llvm::Function* createFunction( VariableReference* varRef, unsigned dimensions, LLVMManager* llvmManager );
    GetBufferElementAddressBuilder& setCanonicalState( llvm::Value* canonicalState );
    GetBufferElementAddressBuilder& setElementSize( llvm::Value* elementSize );
    GetBufferElementAddressBuilder& setX( llvm::Value* x );
    GetBufferElementAddressBuilder& setY( llvm::Value* y );
    GetBufferElementAddressBuilder& setZ( llvm::Value* z );

  private:
    llvm::FunctionType* createType( unsigned dimensions, LLVMManager* llvmManager );

  private:
    llvm::Module* m_module         = nullptr;
    llvm::Value*  m_canonicalState = nullptr;
    llvm::Value*  m_elementSize    = nullptr;
    llvm::Value*  m_x              = nullptr;
    llvm::Value*  m_y              = nullptr;
    llvm::Value*  m_z              = nullptr;
};

// -----------------------------------------------------------------------------
class GetBufferElementAddressFromId : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getBufferId() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    unsigned getDimensionality() const;

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static std::string createUniqueName( unsigned int dimensionality );

  private:
    static bool matchName( llvm::StringRef name );

    static const std::string PREFIX;
    static const int         MANDATORY_ARGS_NUMBER = 4;

    enum Args
    {
        CanonicalState = 0,
        BufferId,
        ElementSize,
        x,
        y,
        z,
        END
    };

    friend class GetBufferElementAddressFromIdBuilder;
};

class GetBufferElementAddressFromIdBuilder
{
  public:
    GetBufferElementAddressFromIdBuilder( llvm::Module* module );

    unsigned getDimensionality( const llvm::Function* function ) const;

    llvm::CallInst* create( llvm::Function* function, llvm::Instruction* insertBefore );
    llvm::Function* createFunction( unsigned dimensions, LLVMManager* llvmManager );
    GetBufferElementAddressFromIdBuilder& setCanonicalState( llvm::Value* canonicalState );
    GetBufferElementAddressFromIdBuilder& setBufferId( llvm::Value* id );
    GetBufferElementAddressFromIdBuilder& setElementSize( llvm::Value* elementSize );
    GetBufferElementAddressFromIdBuilder& setX( llvm::Value* x );
    GetBufferElementAddressFromIdBuilder& setY( llvm::Value* y );
    GetBufferElementAddressFromIdBuilder& setZ( llvm::Value* z );

  private:
    llvm::FunctionType* createType( unsigned dimensions, LLVMManager* llvmManager );

  private:
    llvm::Module* m_module         = nullptr;
    llvm::Value*  m_canonicalState = nullptr;
    llvm::Value*  m_id             = nullptr;
    llvm::Value*  m_elementSize    = nullptr;
    llvm::Value*  m_x              = nullptr;
    llvm::Value*  m_y              = nullptr;
    llvm::Value*  m_z              = nullptr;
};

// -----------------------------------------------------------------------------
class GetPayloadAddressCall : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

    static llvm::CallInst* create( llvm::Function* function, llvm::Value* statePtr, llvm::Instruction* insertBefore );
    static llvm::Function* createFunction( LLVMManager* llvmManager, llvm::Module* module );

  private:
    static const std::string GET_PAYLOAD_ADDRESS;
    static bool matchName( llvm::StringRef name );

    static llvm::FunctionType* createType( LLVMManager* llvmManager );

    enum Args
    {
        CanonicalState = 0,
        END
    };
};

// -----------------------------------------------------------------------------
class GetBufferElement : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getOffset() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    unsigned    getDimensionality() const;
    std::string parseUniqueName() const;

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );
    static std::string parseUniqueName( llvm::StringRef name );
    static std::string createUniqueName( const VariableReference* varRef );

  private:
    static bool matchName( llvm::StringRef name );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 4;

    enum Args
    {
        CanonicalState = 0,
        ElementSize,
        offset,
        x,
        y,
        z,
        END
    };
};

// This class models the output of C14n for _rt_load_or_request_64.
//
//      i32 optixi_loadOrRequestBuffer.variableUniversallyUniqueName(
//          statePtrTy canonicalState,
//          i32 elementSize,
//          i64 ptr,
//          i64 x, i64 y )
// for dimensionality = 2.
//
class LoadOrRequestBufferElement : public OptixIntrinsic
{
  public:
    enum Args
    {
        CanonicalState = 0,
        ElementSize,
        Pointer,
        x,
        y,
        z,
        w,
        END
    };

    llvm::Value* getStatePtr() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getPointer() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    // w value is always the constant zero
    std::string  parseUniqueName() const;
    unsigned int getDimensionality() const;

    static unsigned int getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );
    static std::string parseUniqueName( llvm::StringRef name );
    static std::string createUniqueName( const VariableReference* vref );

  private:
    static bool matchName( llvm::StringRef name );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 4;
};


// This class models the output of C14n for rt_texture_load_or_request_*_id, for example:
//
//    optixi_textureLoadOrRequest2(
//          statePtrTy canonicalState,
//          i32 texId,
//          float %x, float %y,
//          i32* isResident )
//
// for dimensionality = 2.
//
class LoadOrRequestTextureElement : public OptixIntrinsic
{
  public:
    enum Kind
    {
        Nomip,
        Lod,
        Grad
    };

    Kind         getKind() const;
    unsigned int getDimensionality() const;

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static bool matchName( llvm::StringRef name );

    static llvm::Regex nameRegex;
    static const int   MANDATORY_ARGS_NUMBER = 4;
};

// -----------------------------------------------------------------------------
class SetBufferElement : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    llvm::Value* getOffset() const;
    unsigned     getDimensionality() const;
    llvm::Value* getValueToSet() const;
    std::string  parseUniqueName() const;

    static unsigned getDimensionality( const llvm::Function* function );

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static bool isIntrinsic( const llvm::Function* function, const std::string& varRefUniqueName );
    static std::string parseUniqueName( llvm::StringRef name );
    static std::string createUniqueName( const VariableReference* vref );

  private:
    static bool matchName( llvm::StringRef name );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        CanonicalState = 0,
        ElementSize,
        offset,
        x,
        y,
        z,
        toSet,
        END
    };
};

// -----------------------------------------------------------------------------
class GetBufferElementFromId : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getBufferId() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getOffset() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    unsigned getDimensionality() const;

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static std::string createUniqueName( unsigned int dimensionality, const llvm::Type* valueType );

  private:
    static bool matchName( llvm::StringRef name );
    static unsigned getDimensionality( const llvm::Function* function );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 5;

    enum Args
    {
        CanonicalState = 0,
        BufferId,
        ElementSize,
        offset,
        x,
        y,
        z,
        END
    };
};

// -----------------------------------------------------------------------------
class SetBufferElementFromId : public OptixIntrinsic
{
  public:
    llvm::Value* getStatePtr() const;
    llvm::Value* getBufferId() const;
    llvm::Value* getElementSize() const;
    llvm::Value* getX() const;
    llvm::Value* getY() const;
    llvm::Value* getZ() const;
    llvm::Value* getIndex( unsigned dimension ) const;
    llvm::Value* getOffset() const;
    unsigned     getDimensionality() const;
    llvm::Value* getValueToSet() const;

    static bool classof( const llvm::CallInst* inst );
    static bool classof( const llvm::Value* value );
    static bool isIntrinsic( const llvm::Function* function );
    static std::string createUniqueName( unsigned int dimensionality, const llvm::Type* valueType );

  private:
    static bool matchName( llvm::StringRef name );
    static unsigned getDimensionality( const llvm::Function* function );

    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
    static const int         MANDATORY_ARGS_NUMBER = 6;

    enum Args
    {
        CanonicalState = 0,
        BufferId,
        ElementSize,
        offset,
        x,
        y,
        z,
        toSet,
        END
    };
};

// -----------------------------------------------------------------------------
class SetAttributeValue : public OptixIntrinsic
{
  public:
    bool parseUniqueName( llvm::StringRef& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool isIntrinsic( const llvm::Function* function );
    static bool parseUniqueName( const llvm::Function* function, llvm::StringRef& varRefUniqueName );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
};

// -----------------------------------------------------------------------------
class GetAttributeValue : public OptixIntrinsic
{
  public:
    bool parseUniqueName( llvm::StringRef& varRefUniqueName );

    static bool classof( const llvm::CallInst* inst );
    static bool isIntrinsic( const llvm::Function* function );
    static bool parseUniqueName( const llvm::Function* function, llvm::StringRef& varRefUniqueName );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;
};

// -----------------------------------------------------------------------------
class ReportFullIntersection : public OptixIntrinsic
{
  public:
    bool parseUniqueName( llvm::StringRef& name );

    llvm::Value* getHitT() const;
    llvm::Value* getMaterialIndex() const;
    llvm::Value* getHitKind() const;
    llvm::Value* getAttributeData() const;

    static bool classof( const llvm::CallInst* inst );
    static bool isIntrinsic( const llvm::Function* function );
    static bool parseUniqueName( const llvm::Function* function, llvm::StringRef& name );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;

    enum Args
    {
        CanonicalState = 0,
        HitT,
        MaterialIndex,
        HitKind,
        AttributeData,
        END
    };
};

// -----------------------------------------------------------------------------
class IsPotentialIntersection : public OptixIntrinsic
{
  public:
    llvm::Value* getHitT() const;

    static bool classof( const llvm::CallInst* inst );
    static bool isIntrinsic( const llvm::Function* function );

  private:
    static const std::string PREFIX;
    static llvm::Regex       nameRegex;

    enum Args
    {
        CanonicalState = 0,
        HitT,
        END
    };
};


}  // namespace optix
