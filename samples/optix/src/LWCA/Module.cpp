// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO MODULE SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Module.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <LWCA/TexRef.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Module::Module()
    : m_module( nullptr )
{
}

Module::Module( LWmodule module )
    : m_module( module )
{
}

LWmodule Module::get()
{
    return m_module;
}

const LWmodule Module::get() const
{
    return m_module;
}


Function Module::getFunction( const std::string& name, LWresult* returnResult ) const
{
    RT_ASSERT( m_module != nullptr );
    LWfunction func = nullptr;
    CHECK( lwdaDriver().LwModuleGetFunction( &func, m_module, name.c_str() ) );
    return Function( func );
}

LWdeviceptr Module::getGlobal( size_t* bytes, const std::string& name, LWresult* returnResult ) const
{
    RT_ASSERT( m_module != nullptr );
    size_t fakeBytes;
    if( !bytes )
        bytes       = &fakeBytes;
    LWdeviceptr ptr = 0;
    CHECK( lwdaDriver().LwModuleGetGlobal( &ptr, bytes, m_module, name.c_str() ) );
    return ptr;
}

TexRef Module::getTexRef( const std::string& name, LWresult* returnResult ) const
{
    RT_ASSERT( m_module != nullptr );
    LWtexref texref = nullptr;
    CHECK( lwdaDriver().LwModuleGetTexRef( &texref, m_module, name.c_str() ) );
    return TexRef( texref );
}


// TODO: implement surfref
#if 0
SurfRef
Module::getSurfRef( const std::string& name, LWresult* returnResult )
{
  RT_ASSERT( m_module != 0 );
  LWsurfref surfref = 0;
  CHECK( lwdaDriver().lwModuleGetSurfRef( &surfref, m_module, name.c_str() ) );
  return SurfRef(surfref);
}
#endif

Module Module::load( const std::string& fname, LWresult* returnResult )
{
    LWmodule module = nullptr;
    CHECK( lwdaDriver().LwModuleLoad( &module, fname.c_str() ) );
    return Module( module );
}

Module Module::loadData( const void* image, LWresult* returnResult )
{
    LWmodule module = nullptr;
    CHECK( lwdaDriver().LwModuleLoadData( &module, image ) );
    return Module( module );
}

Module Module::loadDataEx( const void* image, unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult )
{
    LWmodule module = nullptr;
    CHECK( lwdaDriver().LwModuleLoadDataEx( &module, image, numOptions, options, optiolwalues ) );
    return Module( module );
}

Module Module::loadDataExHidden( const void* image, unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult )
{
    LWmodule module = nullptr;
    CHECK( lwdaDriver().LwModuleLoadDataExHidden( &module, image, numOptions, options, optiolwalues ) );
    return Module( module );
}

Module Module::loadFatBinary( const void* fatLwbin, LWresult* returnResult )
{
    LWmodule module = nullptr;
    CHECK( lwdaDriver().LwModuleLoadFatBinary( &module, fatLwbin ) );
    return Module( module );
}

void Module::unload( LWresult* returnResult )
{
    RT_ASSERT( m_module != nullptr );
    CHECK( lwdaDriver().LwModuleUnload( m_module ) );
    m_module = nullptr;
}
