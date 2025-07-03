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

#include <Context/LLVMManager.h>

#include <AS/TraverserRuntime_bin.h>
#include <ExelwtionStrategy/RTX/RTXRuntime_bin.h>
#include <FrontEnd/Canonical/C14nRuntime_bin.h>
#include <FrontEnd/PTX/Canonical/UberPointer.h>
#include <FrontEnd/PTX/DataLayout.h>

#include <exp/context/DeviceContext.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/compiler/ModuleCache.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/Threading.h>


using namespace optix;
using namespace prodlib;

LLVMManager::LLVMManager( Context* context )
    : m_context( context )
{
    {
        optix_exp::LlvmStart& initializer = optix_exp::LlvmStart::get();
        RT_ASSERT_MSG( initializer.started(), "LLVM is built without threading support" );
    }

    m_llvmModuleCache.reset( new prodlib::ModuleCache() );
    m_llvmContext.reset( new llvm::LLVMContext() );
    m_llvmDataLayout.reset( new llvm::DataLayout( createDataLayoutForLwrrentProcess() ) );

    // Initialize the Vendor library, which is needed for pass dependency
    // resolution
    llvm::PassRegistry& registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeVendor( registry );

    // Base types.
    m_floatTy = llvm::Type::getFloatTy( llvmContext() );
    m_i1Ty    = llvm::Type::getInt1Ty( llvmContext() );
    m_i8Ty    = llvm::Type::getInt8Ty( llvmContext() );
    m_i32Ty   = llvm::Type::getInt32Ty( llvmContext() );
    m_i64Ty   = llvm::Type::getInt64Ty( llvmContext() );
    m_voidTy  = llvm::Type::getVoidTy( llvmContext() );

    {
        // State
        std::string name = "struct.cort::CanonicalState";
        m_stateTy        = llvm::StructType::create( llvmContext(), name );
    }

    {
        // uint3
        std::string name       = "struct.cort::uint3";
        llvm::Type* elements[] = {m_i32Ty, m_i32Ty, m_i32Ty};
        m_uint3Ty              = llvm::StructType::create( elements, name );
    }

    {
        // uint4
        std::string name       = "struct.cort::uint4";
        llvm::Type* elements[] = {m_i32Ty, m_i32Ty, m_i32Ty, m_i32Ty};
        m_uint4Ty              = llvm::StructType::create( elements, name );
    }

    {
        // float2
        std::string name       = "struct.cort::float2";
        llvm::Type* elements[] = {m_floatTy, m_floatTy};
        m_float2Ty             = llvm::StructType::create( elements, name );
    }

    {
        // float3
        std::string name       = "struct.cort::float3";
        llvm::Type* elements[] = {m_floatTy, m_floatTy, m_floatTy};
        m_float3Ty             = llvm::StructType::create( elements, name );
    }

    {
        // float4
        std::string name       = "struct.cort::float4";
        llvm::Type* elements[] = {m_floatTy, m_floatTy, m_floatTy, m_floatTy};
        m_float4Ty             = llvm::StructType::create( elements, name );
    }

    {
        // size_t
        m_sizeTTy = sizeof( void* ) == 8 ? m_i64Ty : m_i32Ty;
    }

    {
        // size3
        std::string name       = "struct.cort::size3";
        llvm::Type* elements[] = {m_sizeTTy, m_sizeTTy, m_sizeTTy};
        m_size3Ty              = llvm::StructType::create( elements, name );
    }

    {
        // Buffer
        std::string name       = "struct.cort::Buffer";
        llvm::Type* elements[] = {m_i8Ty->getPointerTo(), m_i32Ty, m_size3Ty};
        m_bufferHeaderTy       = llvm::StructType::create( elements, name );
    }

    {
        // OptixRay
        std::string name       = "struct.cort::OptixRay";
        llvm::Type* elements[] = {m_float3Ty, m_float3Ty, m_i32Ty, m_floatTy, m_floatTy};
        m_optixRayTy           = llvm::StructType::create( elements, name );
    }

    // This needs to happen after all the other types have been created
    {
        // UberPointer
        m_uberPointerTy = UberPointer::createType( this );
    }
}

void LLVMManager::enablePassTiming()
{
    llvm::TimePassesIsEnabled = true;
}

bool LLVMManager::isPassTimingEnabled()
{
    return llvm::TimePassesIsEnabled;
}

LLVMManager::~LLVMManager()
{
}

prodlib::ModuleCache* LLVMManager::getLLVMModuleCache() const
{
    return m_llvmModuleCache.get();
}

llvm::LLVMContext& LLVMManager::llvmContext() const
{
    return *m_llvmContext;
}

llvm::DataLayout& LLVMManager::llvmDataLayout() const
{
    return *m_llvmDataLayout;
}

llvm::Type* LLVMManager::getStatePtrType() const
{
    return m_stateTy->getPointerTo( 0 );
}

llvm::StructType* LLVMManager::getUint3Type() const
{
    return m_uint3Ty;
}
llvm::StructType* LLVMManager::getUint4Type() const
{
    return m_uint4Ty;
}
llvm::StructType* LLVMManager::getFloat2Type() const
{
    return m_float2Ty;
}
llvm::StructType* LLVMManager::getFloat3Type() const
{
    return m_float3Ty;
}
llvm::StructType* LLVMManager::getFloat4Type() const
{
    return m_float4Ty;
}
llvm::Type* LLVMManager::getVoidType() const
{
    return m_voidTy;
}
llvm::Type* LLVMManager::getSizeTType() const
{
    return m_sizeTTy;
}
llvm::Type* LLVMManager::getI1Type() const
{
    return m_i1Ty;
}
llvm::Type* LLVMManager::getI8Type() const
{
    return m_i8Ty;
}
llvm::Type* LLVMManager::getI32Type() const
{
    return m_i32Ty;
}
llvm::Type* LLVMManager::getI64Type() const
{
    return m_i64Ty;
}
llvm::Type* LLVMManager::getFloatType() const
{
    return m_floatTy;
}
llvm::StructType* LLVMManager::getSize3Type() const
{
    return m_size3Ty;
}
llvm::StructType* LLVMManager::getBufferHeaderType() const
{
    return m_bufferHeaderTy;
}
llvm::StructType* LLVMManager::getOptixRayType() const
{
    return m_optixRayTy;
}
llvm::StructType* LLVMManager::getUberPointerType() const
{
    return m_uberPointerTy;
}

llvm::Regex* LLVMManager::getCachedRegex( const std::string& name )
{
    auto iter = m_cachedRegexes.find( name );
    if( iter != m_cachedRegexes.end() )
        return iter->second.get();
    m_cachedRegexes[name].reset( new llvm::Regex( name ) );
    return m_cachedRegexes.at( name ).get();
}

llvm::Module* LLVMManager::getC14nRuntime( const std::string& overrideFilename )
{
    return getOrCreateInternalModule( "C14nRuntime", data::getC14nRuntimeData(), data::getC14nRuntimeDataLength(), overrideFilename );
}

llvm::Module* LLVMManager::getTraverserRuntime( const std::string& overrideFilename /*= "" */ )
{
    return getOrCreateInternalModule( "TraverserRuntime", optix::data::getTraverserRuntimeData(),
                                      optix::data::getTraverserRuntimeDataLength(), overrideFilename );
}

llvm::Module* LLVMManager::getRTXRuntime( const std::string& overrideFilename )
{
    return getOrCreateInternalModule( "RTXRuntime", data::getRTXRuntimeData(), data::getRTXRuntimeDataLength(), overrideFilename );
}

static std::unique_ptr<llvm::Module> getFreshModule( llvm::LLVMContext& llvmContext,
                                                     const char*        bitcode,
                                                     size_t             bitcodeSize,
                                                     const std::string& overrideFilename )
{
    std::unique_ptr<llvm::Module> module;
    // Load from file or in-memory copy
    if( !overrideFilename.empty() )
    {
        module.reset( corelib::loadModuleFromAsmFile( llvmContext, overrideFilename ) );
    }
    else
    {
        module.reset( corelib::loadModuleFromBitcodeLazy( llvmContext, bitcode, bitcodeSize ) );
    }
    if( llvm::Error err = module->materializeAll() )
        throw CompileError( RT_EXCEPTION_INFO, "internal compilation error" );
    return std::move( module );
}

std::unique_ptr<llvm::Module> LLVMManager::getRTXRuntime( llvm::LLVMContext& llvmContext, const std::string& overrideFilename )
{
    return getFreshModule( llvmContext, data::getRTXRuntimeData(), data::getRTXRuntimeDataLength(), overrideFilename );
}

llvm::Module* LLVMManager::getOrCreateInternalModule( const std::string& name, const char* bitcode, size_t size, const std::string& overrideFilename )
{
    llvm::Module* module = m_llvmModuleCache->getModule( name );
    if( !module )
    {
        module = ModuleCache::getOrCreateModule( m_llvmModuleCache.get(), llvmContext(), name, bitcode, size, overrideFilename );
        if( llvm::Error err = module->materializeAll() )
            throw CompileError( RT_EXCEPTION_INFO, "internal compilation error" );
        m_llvmModuleCache->addModule( name, module );
    }
    return module;
}
