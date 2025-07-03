
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

#include <Context/ObjectManager.h>
#include <Objects/Buffer.h>
#include <Objects/ManagedObject.h>
#include <Util/LinkedPtr.h>

using namespace optix;

ManagedObject::ManagedObject()
    : m_context( nullptr )
    , m_class( RT_OBJECT_UNKNOWN )
{
}

ManagedObject::ManagedObject( Context* context, ObjectClass objClass )
    : m_context( context )
    , m_class( objClass )
{
}

ManagedObject::~ManagedObject() NOEXCEPT_FALSE
{
    RT_ASSERT_MSG( m_linkedPointers.empty(), "Managed Object destroyed while references remain" );
}

void ManagedObject::addLink( LinkedPtr_Link* ptr )
{
    m_linkedPointers.addItem( ptr );
}

void ManagedObject::removeLink( LinkedPtr_Link* ptr )
{
    m_linkedPointers.removeItem( ptr );
}
