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

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace llvm {
class DataLayout;
class LLVMContext;
class Module;
class Regex;
class StructType;
class Type;
}

namespace prodlib {
class ModuleCache;
}

namespace optix {
class Context;

class LLVMManager
{
  public:
    LLVMManager( Context* context );
    ~LLVMManager();  // needed to avoid having to know the types inside the std::unique_ptrs.

    llvm::Module* getC14nRuntime( const std::string& overrideFilename = "" );
    llvm::Module* getRTXRuntime( const std::string& overrideFilename = "" );
    llvm::Module* getTraverserRuntime( const std::string& overrideFilename = "" );

    // Get a fresh copy of the module in the specified context.
    static std::unique_ptr<llvm::Module> getRTXRuntime( llvm::LLVMContext& context,
                                                        const std::string& overrideFilename = "" );


    prodlib::ModuleCache* getLLVMModuleCache() const;
    llvm::LLVMContext&    llvmContext() const;
    llvm::DataLayout&     llvmDataLayout() const;

    llvm::Type*       getStatePtrType() const;
    llvm::StructType* getUint3Type() const;
    llvm::StructType* getUint4Type() const;
    llvm::StructType* getFloat2Type() const;
    llvm::StructType* getFloat3Type() const;
    llvm::StructType* getFloat4Type() const;
    llvm::Type*       getVoidType() const;
    llvm::Type*       getSizeTType() const;
    llvm::Type*       getI1Type() const;
    llvm::Type*       getI8Type() const;
    llvm::Type*       getI32Type() const;
    llvm::Type*       getI64Type() const;
    llvm::Type*       getFloatType() const;
    llvm::StructType* getSize3Type() const;
    llvm::StructType* getBufferHeaderType() const;
    llvm::StructType* getOptixRayType() const;
    llvm::StructType* getUberPointerType() const;

    llvm::Regex* getCachedRegex( const std::string& name );

    void enablePassTiming();
    bool isPassTimingEnabled();

  private:
    llvm::Module* getOrCreateInternalModule( const std::string& name,
                                             const char*        bitcode,
                                             size_t             size,
                                             const std::string& overrideFilename = "" );


    std::unique_ptr<prodlib::ModuleCache> m_llvmModuleCache;
    std::unique_ptr<llvm::DataLayout>     m_llvmDataLayout;

    // The LLVM context owned by the LLVM manager.
    std::unique_ptr<llvm::LLVMContext> m_llvmContext;

    Context* m_context = nullptr;

    llvm::StructType* m_stateTy        = nullptr;
    llvm::StructType* m_uint3Ty        = nullptr;
    llvm::StructType* m_uint4Ty        = nullptr;
    llvm::Type*       m_floatTy        = nullptr;
    llvm::StructType* m_float2Ty       = nullptr;
    llvm::StructType* m_float3Ty       = nullptr;
    llvm::StructType* m_float4Ty       = nullptr;
    llvm::Type*       m_voidTy         = nullptr;
    llvm::Type*       m_sizeTTy        = nullptr;
    llvm::Type*       m_i1Ty           = nullptr;
    llvm::Type*       m_i8Ty           = nullptr;
    llvm::Type*       m_i32Ty          = nullptr;
    llvm::Type*       m_i64Ty          = nullptr;
    llvm::StructType* m_size3Ty        = nullptr;
    llvm::StructType* m_bufferHeaderTy = nullptr;
    llvm::StructType* m_optixRayTy     = nullptr;
    llvm::StructType* m_uberPointerTy  = nullptr;

    std::map<std::string, std::unique_ptr<llvm::Regex>> m_cachedRegexes;
};
}  // end namespace optix
