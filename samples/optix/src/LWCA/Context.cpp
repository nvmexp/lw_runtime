// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO CONTEXT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Context.h>

#include <LWCA/Device.h>
#include <LWCA/ErrorCheck.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Context::Context()
    : m_context( nullptr )
{
}

Context::Context( LWcontext context )
    : m_context( context )
{
}

LWcontext Context::get()
{
    return m_context;
}

const LWcontext Context::get() const
{
    return m_context;
}

Context Context::create( unsigned int flags, const Device& dev, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    LWcontext context = nullptr;
    CHECK( lwdaDriver().LwCtxCreate( &context, flags, dev.get() ) );
    return Context( context );
}

void Context::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_context != nullptr );
    CHECK( lwdaDriver().LwCtxDestroy( m_context ) );
    m_context = nullptr;
}

unsigned int Context::getApiVersion( LWresult* returnResult )
{
    RT_ASSERT( m_context != nullptr );
    unsigned int version = 0;
    CHECK( lwdaDriver().LwCtxGetApiVersion( m_context, &version ) );
    return version;
}

LWfunc_cache Context::getCacheConfig( LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Context::getCacheConfig not implemented" );
#if 0
  LWfunc_cache result = 0;
  CHECK( lwdaDriver().lwCtxGetCacheConfig ( &result ) );
  return result;
#endif
}

Context Context::getLwrrent( LWresult* returnResult )
{
    LWcontext context = nullptr;
    CHECK( lwdaDriver().LwCtxGetLwrrent( &context ) );
    return Context( context );
}

Device Context::getDevice( LWresult* returnResult )
{
    LWdevice device = 0;
    CHECK( lwdaDriver().LwCtxGetDevice( &device ) );
    return Device( device );
}

size_t Context::getLimit( LWlimit limit, LWresult* returnResult )
{
    size_t value = 0;
    CHECK( lwdaDriver().LwCtxGetLimit( &value, limit ) );
    return value;
}

bool Context::getFlag( unsigned int flag, LWresult* returnResult )
{
    unsigned int flags;
    CHECK( lwdaDriver().LwCtxGetFlags( &flags ) );

    return flags & flag;
}

LWsharedconfig Context::getSharedMemConfig( LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Context::getSharedMemConfig not implemented" );
#if 0
  LWsharedconfig config = 0;
  CHECK( lwdaDriver().lwCtxGetSharedMemConfig ( &config ) );
  return config;
#endif
}

void Context::getStreamPriorityRange( int* leastPriority, int* greatestPriority, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Context::getStreamPriorityRange not implemented" );
#if 0
  CHECK( lwdaDriver().lwCtxGetStreamPriorityRange ( leastPriority, greatestPriority, ) );
#endif
}

Context Context::popLwrrent( LWresult* returnResult )
{
    LWcontext context = nullptr;
    CHECK( lwdaDriver().LwCtxPopLwrrent( &context ) );
    return Context( context );
}

void Context::pushLwrrent( LWresult* returnResult )
{
    RT_ASSERT( m_context != nullptr );
    CHECK( lwdaDriver().LwCtxPushLwrrent( m_context ) );
}

void Context::setCacheConfig( LWfunc_cache config, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Context::setCacheConfig not implemented" );
#if 0
  CHECK( lwdaDriver().lwCtxSetCacheConfig ( config ) );
#endif
}

void Context::setLwrrent( LWresult* returnResult )
{
    RT_ASSERT( m_context != nullptr );
    LWcontext context = nullptr;
    CHECK( lwdaDriver().LwCtxGetLwrrent( &context ) );
    if( context != m_context )
        CHECK( lwdaDriver().LwCtxSetLwrrent( m_context ) );
}

void Context::unsetLwrrent( LWresult* returnResult )
{
    CHECK( lwdaDriver().LwCtxSetLwrrent( nullptr ) );
}

void Context::setLimit( LWlimit limit, size_t value, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwCtxSetLimit( limit, value ) );
}

void Context::setSharedMemConfig( LWsharedconfig config, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Context::setSharedMemConfig not implemented" );
#if 0
  CHECK( lwdaDriver().lwCtxSetSharedMemConfig ( config ) );
#endif
}

void Context::synchronize( LWresult* returnResult )
{
    CHECK( lwdaDriver().LwCtxSynchronize() );
}

void Context::devicePrimaryCtxGetState( const Device& dev, unsigned int* flags, int* active, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    CHECK( lwdaDriver().LwDevicePrimaryCtxGetState( dev.get(), flags, active ) );
}

void Context::devicePrimaryCtxRelease( const Device& dev, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    CHECK( lwdaDriver().LwDevicePrimaryCtxRelease( dev.get() ) );
}

void Context::devicePrimaryCtxReset( const Device& dev, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    CHECK( lwdaDriver().LwDevicePrimaryCtxReset( dev.get() ) );
}

Context Context::devicePrimaryCtxRetain( const Device& dev, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    LWcontext context = nullptr;
    CHECK( lwdaDriver().LwDevicePrimaryCtxRetain( &context, dev.get() ) );
    return Context( context );
}

void Context::devicePrimaryCtxSetFlags( const Device& dev, unsigned int flags, LWresult* returnResult )
{
    RT_ASSERT( dev.isValid() );
    CHECK( lwdaDriver().LwDevicePrimaryCtxSetFlags( dev.get(), flags ) );
}

void Context::disablePeerAccess( const Context& peerContext, LWresult* returnResult ) const
{
    CHECK( lwdaDriver().LwCtxDisablePeerAccess( peerContext.get() ) );
}

void Context::enablePeerAccess( const Context& peerContext, unsigned int Flags, LWresult* returnResult ) const
{
    CHECK( lwdaDriver().LwCtxEnablePeerAccess( peerContext.get(), Flags ) );
}

bool Context::isValid() const
{
    return this->get() != nullptr;
}
