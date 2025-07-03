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

#include <LWCA/TexObject.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <LWCA/Module.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;


TexObject::TexObject()
    : m_texObject( ~0ull )
{
}

TexObject::TexObject( LWtexObject ref )
    : m_texObject( ref )
{
}

LWtexObject TexObject::get() const
{
    return m_texObject;
}

bool TexObject::isNull() const
{
    return m_texObject == ~0ull;
}

TexObject TexObject::create( const LWDA_RESOURCE_DESC&      pResDesc,
                             const LWDA_TEXTURE_DESC&       pTexDesc,
                             const LWDA_RESOURCE_VIEW_DESC* pResViewDesc,
                             LWresult*                      returnResult )
{
    LWtexObject result = 0;
    CHECK( lwdaDriver().LwTexObjectCreate( &result, &pResDesc, &pTexDesc, pResViewDesc ) );
    return TexObject( result );
}

void TexObject::destroy( LWresult* returnResult )
{
    if( !isNull() )
        CHECK( lwdaDriver().LwTexObjectDestroy( m_texObject ) );
}

void TexObject::getResourceDesc( LWDA_RESOURCE_DESC& pResDesc, LWresult* returnResult ) const
{
    CHECK( lwdaDriver().LwTexObjectGetResourceDesc( &pResDesc, m_texObject ) );
}

void TexObject::getResourceViewDesc( LWDA_RESOURCE_VIEW_DESC& pResViewDesc, LWresult* returnResult ) const
{
    RT_ASSERT_FAIL_MSG( "TexObject::getResourceViewDesc not implemented" );
#if 0
  CHECK( lwdaDriver().lwTexObjectGetResourceViewDesc( &pResViewDesc, m_texObject ) );
#endif
}

void TexObject::getTextureDesc( LWDA_TEXTURE_DESC& pTexDesc, LWresult* returnResult ) const
{
    RT_ASSERT_FAIL_MSG( "TexObject::getTextureDesc not implemented" );
#if 0
  CHECK( lwdaDriver().lwTexObjectGetTextureDesc( &pTexDesc, m_texObject ) );
#endif
}
