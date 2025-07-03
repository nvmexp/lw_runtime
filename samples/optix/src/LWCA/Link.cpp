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

#include <LWCA/Link.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Module.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Link::Link()
    : m_linkState( nullptr )
{
}

Link::Link( LWlinkState linkState )
    : m_linkState( linkState )
{
}

LWlinkState Link::get()
{
    return m_linkState;
}

const LWlinkState Link::get() const
{
    return m_linkState;
}

// Creates a pending JIT linker invocation.
Link Link::create( unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult )
{
    LWlinkState linkState = nullptr;
    CHECK( lwdaDriver().LwLinkCreate( numOptions, options, optiolwalues, &linkState ) );
    return Link( linkState );
}

// Add an input to a pending linker invocation.
void Link::addData( LWjitInputType     type,
                    void*              data,
                    size_t             size,
                    const std::string& name,
                    unsigned int       numOptions,
                    LWjit_option*      options,
                    void**             optiolwalues,
                    LWresult*          returnResult )
{
    RT_ASSERT( m_linkState != nullptr );
    CHECK( lwdaDriver().LwLinkAddData( m_linkState, type, data, size, name.c_str(), numOptions, options, optiolwalues ) );
}

// Add a file input to a pending linker invocation.
void Link::addFile( LWjitInputType type, const std::string& path, unsigned int numOptions, LWjit_option* options, void** optiolwalues, LWresult* returnResult )
{
    RT_ASSERT( m_linkState != nullptr );
    CHECK( lwdaDriver().LwLinkAddFile( m_linkState, type, path.c_str(), numOptions, options, optiolwalues ) );
}

// Complete a pending linker invocation.
void Link::complete( void** lwbinOut, size_t* sizeOut, LWresult* returnResult )
{
    RT_ASSERT( m_linkState != nullptr );
    CHECK( lwdaDriver().LwLinkComplete( m_linkState, lwbinOut, sizeOut ) );
}

// Complete a pending linker invocation - alternate version that returns a loaded module
Module Link::complete( LWresult* returnResult )
{
    void*  lwbinOut;
    size_t sizeOut;
    complete( &lwbinOut, &sizeOut, returnResult );
    if( returnResult && *returnResult != LWDA_SUCCESS )
        return Module();

    return Module::loadDataEx( lwbinOut, 0, nullptr, nullptr, returnResult );
}

// Destroys state for a JIT linker invocation.
void Link::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_linkState != nullptr );
    CHECK( lwdaDriver().LwLinkDestroy( m_linkState ) );
}
