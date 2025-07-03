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

#include <LWCA/Stream.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Event.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Stream::Stream()
    : m_stream( nullptr )
{
}

Stream::Stream( LWstream stream )
    : m_stream( stream )
{
}

LWstream Stream::get()
{
    return m_stream;
}

const LWstream Stream::get() const
{
    return m_stream;
}

Stream Stream::create( unsigned int Flags, LWresult* returnResult )
{
    LWstream stream = nullptr;
    CHECK( lwdaDriver().LwStreamCreate( &stream, Flags ) );
    return Stream( stream );
}

Stream Stream::createWithPriority( unsigned int flags, int priority, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Stream::createWithPriority not implemented" );
#if 0
  LWstream stream = 0;
  CHECK( lwdaDriver().lwStreamCreateWithPriority( &stream, flags, priority ) );
  return Stream(stream);
#endif
}

void Stream::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_stream != nullptr );
    CHECK( lwdaDriver().LwStreamDestroy( m_stream ) );
    m_stream = nullptr;
}

unsigned int Stream::getFlags( LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Stream::getFlags not implemented" );
#if 0
  RT_ASSERT(m_stream != 0);
  unsigned int flags = 0;
  CHECK( lwdaDriver().lwStreamGetFlags( m_stream, &flags ) );
  return flags;
#endif
}

int Stream::getPriority( LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Stream::getPriority not implemented" );
#if 0
  RT_ASSERT(m_stream != 0);
  int prio = 0;
  CHECK( lwdaDriver().lwStreamGetPriority( m_stream, &prio ) );
  return prio;
#endif
}

bool Stream::query( LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "Stream::query not implemented" );
#if 0
  RT_ASSERT(m_stream != 0);
  LWresult err = lwdaDriver().lwdaDriver().lwStreamQuery( m_stream );
  if(err == LWDA_SUCCESS)
    return true;
  if(err == LWDA_ERROR_NOT_READY)
    return false;

  // A real error
  if(errorReturn) {
    errorReturn.set(err);
    return false;
  } else {
    throw prodlib::LwdaError( RT_EXCEPTION_INFO, "lwdaDriver().lwStreamQuery( m_stream )", err );
  }
#endif
}

void Stream::synchronize( LWresult* returnResult )
{
    RT_ASSERT( m_stream != nullptr );
    CHECK( lwdaDriver().LwStreamSynchronize( m_stream ) );
}

void Stream::waitEvent( const Event& event, unsigned int Flags, LWresult* returnResult ) const
{
    RT_ASSERT( m_stream != nullptr );
    CHECK( lwdaDriver().LwStreamWaitEvent( m_stream, event.get(), Flags ) );
}