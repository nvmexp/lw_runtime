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

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringRef.h>

#include <map>
#include <string>
#include <vector>

namespace llvm {
class BitVector;
class CallInst;
class Constant;
class ExtractValueInst;
class Function;
class GlobalVariable;
class InsertValueInst;
class Instruction;
class LoadInst;
class Module;
class StructType;
class Twine;
class Type;
class Value;
template <typename T, unsigned N>
class SmallVector;
}

namespace optix {
class LLVMManager;

// -----------------------------------------------------------------------------
class UberPointer
{
  public:
    // Make UberPointer non-copyable.
    UberPointer( UberPointer& ) = delete;
    UberPointer& operator=( UberPointer& ) = delete;

  public:
    enum class PointeeType
    {
        Attribute,
        Buffer,
        BufferID,
        LwrrentRay,
        LwrrentTime,
        ExceptionDetail,
        LaunchDim,
        LaunchIndex,
        Payload,
        Raw,
        TMax,
        Variable,
        SubframeIndex,
        Unknown
    };

    typedef enum {
        Offset      = 0,
        UPKind      = 1,
        ID          = 2,
        BufferIndex = 3,
        ElementSize = 4,
        RawPointer  = 5,
    } Field;

    // struct UberPointer {
    //   size_t offset;  // Offset within the element
    //   int kind;       // Index into UberPointerSet
    //   int id;         // buffer id, exception index
    //   size_t3 buffer_index; // for BufferGet
    // };

    template <Field    F>
    static llvm::Type* getType( const LLVMManager* llvm_manager );
    // For UberPointer itself
    static llvm::StructType* getType( const LLVMManager* llvm_manager );

    // These are designed to create the actual type rather than simply getting them.  Only
    // the LLVMManager should call these functions.
    template <Field    F>
    static llvm::Type* createType( const LLVMManager* llvm_manager );
    // For UberPointer itself
    static llvm::StructType* createType( const LLVMManager* llvm_manager );

    template <Field     F>
    static llvm::Value* get( llvm::Value* UberP, const llvm::Twine& NameStr, llvm::Instruction* insertBefore );

    template <Field        F>
    static llvm::Constant* get( llvm::Constant* UberP );

    template <Field F>
    static llvm::Value* set( llvm::Value* UberP, llvm::Value* new_value, const llvm::Twine& NameStr, llvm::Instruction* insertBefore );

    template <Field        F>
    static llvm::Constant* set( llvm::Constant* UberP, llvm::Constant* new_value );

    static llvm::Constant* getUberPointer( LLVMManager* llvm_manager, int upkind, size_t elementSize, size_t initialOffset );
    static llvm::Constant* getUberPointer( LLVMManager* llvm_manager, int upkind, llvm::Constant* index, size_t elementSize, size_t initialOffset );
    static llvm::Value* getUberPointer( LLVMManager*                     llvm_manager,
                                        int                              upkind,
                                        const std::vector<llvm::Value*>& dims,
                                        size_t                           elementSize,
                                        size_t                           initialOffset,
                                        llvm::Instruction*               insertBefore );
    static llvm::Value* getUberPointer( LLVMManager*                     llvm_manager,
                                        int                              upkind,
                                        llvm::Value*                     bufferId,
                                        const std::vector<llvm::Value*>& dims,
                                        size_t                           elementSize,
                                        size_t                           initialOffset,
                                        llvm::Instruction*               insertBefore );
    static llvm::Value* getRawUberPointer( LLVMManager* llvm_manager, int upkind, llvm::Value* raw_pointer, llvm::Instruction* insertBefore );
};

// -----------------------------------------------------------------------------
class UberPointerTransform
{
  public:
    // Make UberPointerTransform non-copyable.
    UberPointerTransform( const UberPointerTransform& ) = delete;
    UberPointerTransform& operator=( const UberPointerTransform& ) = delete;

  public:
    UberPointerTransform( llvm::Module* module, LLVMManager* llvm_manager, const std::string& name, UberPointer::PointeeType pointeeType );

    bool hasAnyLoadsAndStores() { return m_hasLoads || m_hasStores; }
    bool hasLoads() { return m_hasLoads; }
    bool hasStores() { return m_hasStores; }
    bool pointerEscapes() { return m_pointerEscapes; }

    const std::vector<llvm::Value*>& getLoads() { return m_insertedLoads; }

    void translate( llvm::Value* start, int upkind, size_t elementSize, size_t initialOffset );
    void translate( llvm::Value* start, int upkind, llvm::Constant* index, size_t elementSize, size_t initialOffset );
    void translate( llvm::Instruction* start, int upkind, const std::vector<llvm::Value*>& dims, size_t elementSize, size_t initialOffset );
    void translate( llvm::Instruction* start, int upkind, llvm::Value* bufferId, const std::vector<llvm::Value*>& args, size_t elementSize, size_t initialOffset );

    static const std::string UBER_POINTER_MEM_READ;
    static const std::string UBER_POINTER_MEM_READ_TO_ALLOCA;
    static const std::string UBER_POINTER_MEM_WRITE;
    static const std::string UBER_POINTER_ATOMIC_OP;
    static const std::string UBER_POINTER_GET_ADDRESS;

  private:
    llvm::Module*            m_module      = nullptr;
    LLVMManager*             m_llvmManager = nullptr;
    std::string              m_name;
    bool                     m_hasLoads       = false;
    bool                     m_hasStores      = false;
    bool                     m_pointerEscapes = false;
    UberPointer::PointeeType m_pointeeType    = UberPointer::PointeeType::Unknown;
    llvm::SmallSet<llvm::Value*, 16> m_visited;  // needed to prevent loop when doing dataflow
    std::vector<llvm::Value*> m_insertedLoads;

    enum class AccessType
    {
        READ = 0,
        WRITE,
        ATOMIC_OP,
        GET_ADDRESS
    };

    void translate( llvm::Value* V_start, llvm::Value* UberP_in );

    llvm::CallInst* createUberMemAccessCall( llvm::Value*       UberP,
                                             llvm::Instruction* insertBefore,
                                             llvm::Function*    parent,
                                             llvm::Type*        type,
                                             AccessType         accessType,
                                             llvm::Value*       storeBuffer );

    llvm::Value* createUberPointerCast( llvm::Value* V, llvm::Instruction* insertBefore );
};

// -----------------------------------------------------------------------------
class UberPointerSet
{
  public:
    // Make UberPointerSet non-copyable.
    UberPointerSet( const UberPointerSet& ) = delete;
    UberPointerSet& operator=( const UberPointerSet& ) = delete;

    UberPointerSet( LLVMManager* llvmManager )
        : m_llvmManager( llvmManager ){};

  public:
    void finalizeUberPointerGetsAndSets( llvm::Module* module );
    int getOrAddUberPointerKind( const std::string&       name,
                                 llvm::Function*          getter,
                                 llvm::Function*          setter,
                                 llvm::Function*          atomic,
                                 llvm::Function*          getAddress,
                                 UberPointer::PointeeType type,
                                 llvm::Value*             defaultValue );

  private:
    struct Info
    {
        std::string              name;
        UberPointer::PointeeType type;
        llvm::Function*          getter;
        llvm::Function*          setter;
        llvm::Function*          atomic;
        llvm::Function*          getAddress;
        llvm::Value*             variableDefault;

        Info( const std::string&       name,
              UberPointer::PointeeType type,
              llvm::Function*          getter,
              llvm::Function*          setter,
              llvm::Function*          atomic,
              llvm::Function*          getAddress,
              llvm::Value*             variableDefault );
    };

    void finalizeUberPointerAtomic( llvm::Function* function );
    void finalizeUberPointerGetsOrSets( llvm::Function* getterOrSetter, bool isPut );
    void finalizeUberPointerGetAddress( llvm::Function* getAddress );
    void finalizeUberPointerGetsToAlloca( llvm::Function* getterOrSetter );

    void findValidUberPointerKinds( llvm::BitVector& validKinds, llvm::Value* UberP );

    void generateSingleGetOrSet( int                idx,
                                 llvm::Value*       UberP,
                                 llvm::Value*       statePtr,
                                 llvm::Value*       allocaptr,
                                 llvm::Type*        type,
                                 llvm::Value*       toStore,
                                 llvm::Instruction* insertBefore );
    void generateSingleGetOrSetPayload( Info&              info,
                                        llvm::Value*       UberP,
                                        llvm::Value*       statePtr,
                                        llvm::Value*       allocaptr,
                                        llvm::Type*        type,
                                        llvm::Value*       toStore,
                                        llvm::Instruction* insertBefore );
    void generateSingleGetOrSetBufferElement( Info&              info,
                                              llvm::Value*       UberP,
                                              llvm::Value*       statePtr,
                                              llvm::Value*       allocaptr,
                                              llvm::Type*        type,
                                              llvm::Value*       toStore,
                                              llvm::Instruction* insertBefore );
    void generateSingleGetOrSetBufferElementFromId( Info&              info,
                                                    llvm::Value*       UberP,
                                                    llvm::Value*       statePtr,
                                                    llvm::Value*       allocaptr,
                                                    llvm::Type*        type,
                                                    llvm::Value*       toStore,
                                                    llvm::Instruction* insertBefore );
    void generateSingleGetVariable( Info&              info,
                                    llvm::Value*       UberP,
                                    llvm::Value*       statePtr,
                                    llvm::Value*       allocaptr,
                                    llvm::Type*        type,
                                    llvm::Instruction* insertBefore );
    void generateFreePointerAccess( llvm::Value* UberP, llvm::Value* allocaptr, bool isPut, llvm::Instruction* insertBefore );
    llvm::CallInst* generateRawAtomicCall( llvm::Value*       uberP,
                                           llvm::Value*       operatorCode,
                                           llvm::Value*       subElementType,
                                           llvm::Value*       compareOperand,
                                           llvm::Value*       operand,
                                           const Info&        info,
                                           llvm::Instruction* insertBefore );
    llvm::CallInst* generateSingleAtomicCall( llvm::Function*    atomicFunction,
                                              llvm::Value*       statePtr,
                                              llvm::Value*       uberP,
                                              llvm::Value*       operatorCode,
                                              llvm::Value*       subElementType,
                                              llvm::Value*       compareOperand,
                                              llvm::Value*       operand,
                                              const Info&        info,
                                              llvm::Instruction* insertBefore );
    void generateSingleGetOrSetAttribute( Info&              info,
                                          llvm::Value*       UberP,
                                          llvm::Value*       statePtr,
                                          llvm::Value*       allocaptr,
                                          llvm::Type*        type,
                                          llvm::Value*       toStore,
                                          llvm::Instruction* insertBefore );

    llvm::Value* generateSingleGetAddr( int kind, llvm::Value* UberP, llvm::Value* statePtr, llvm::Instruction* insertBefore );
    int getOrInsertRawAccessEntry( llvm::Module* module );
    void checkAliasingRules( const llvm::BitVector& validKinds );

    struct InfoCompare
    {
        bool operator()( const Info& a, const Info& b ) const;
    };
    typedef std::map<Info, int, InfoCompare> InfoMapType;
    InfoMapType       m_uptrInfo;
    std::vector<Info> m_uptrList;
    LLVMManager*      m_llvmManager;
};

}  // end namespace optix
