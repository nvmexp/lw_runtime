// Copyright LWPU Corporation 2014
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "ModuleCache.h"

#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>

using namespace prodlib;

ModuleCache::ModuleCache()
{
}

ModuleCache::~ModuleCache()
{
}

void ModuleCache::addModule( const std::string& name, llvm::Module* module )
{
    if( module == NULL )
        throw IlwalidValue( RT_EXCEPTION_INFO, "module is NULL" );

    m_modules[name] = module;
}

llvm::Module* ModuleCache::getModule( const std::string& name )
{
    ModuleMap::iterator it = m_modules.find( name );
    if( it != m_modules.end() )
        return ( *it ).second;
    else
        return NULL;
}

llvm::Module* ModuleCache::getOrCreateModule( ModuleCache*       moduleCache,
                                              llvm::LLVMContext& context,
                                              const std::string& name,
                                              const char*        bitcode,
                                              size_t             bitcodeSize,
                                              const std::string& knobfile )
{
    // Get from cache?
    llvm::Module* module = nullptr;
    if( moduleCache )
    {
        module = moduleCache->getModule( name );
        if( module )
            return module;
    }

    // Load from file or in-memory copy
    if( !knobfile.empty() )
    {
        module = corelib::loadModuleFromAsmFile( context, knobfile );
    }
    else
    {
        module = corelib::loadModuleFromBitcodeLazy( context, bitcode, bitcodeSize );
    }

    // Return to cache
    if( moduleCache )
    {
        moduleCache->addModule( name, module );
    }

    return module;
}
