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

#include <gmock/gmock.h>

#include <optix_world.h>

using namespace optix;

class TestObjectDestruction : public testing::Test
{
  public:
    Context  m_context;
    Selector m_selector;
    Group    m_group;

    void SetUp()
    {
        m_context  = Context::create();
        m_selector = m_context->createSelector();
        m_group    = m_context->createGroup();
    }

    void TearDown() { ASSERT_NO_THROW( m_context->destroy() ); }
};

// Selectors.
// -----------------------------------------------------------------------------
TEST_F( TestObjectDestruction, CanDestroyEmptySelector )
{
    m_selector->setChildCount( 0 );
    EXPECT_NO_THROW( m_selector->destroy() );
}

// -----------------------------------------------------------------------------
TEST_F( TestObjectDestruction, CanDestroyNonFullSelector )
{
    m_selector->setChildCount( 2 );
    EXPECT_NO_THROW( m_selector->destroy() );
}

// -----------------------------------------------------------------------------
TEST_F( TestObjectDestruction, CanDestroySelectorBeforeChild )
{
    m_selector->setChildCount( 1 );
    Group child = m_context->createGroup();
    m_selector->setChild( 0, child );

    EXPECT_NO_THROW( m_selector->destroy() );
    EXPECT_NO_THROW( child->destroy() );
}

// -----------------------------------------------------------------------------
TEST_F( TestObjectDestruction, CanDestroyChildBeforeSelector )
{
    m_selector->setChildCount( 1 );
    Group child = m_context->createGroup();
    m_selector->setChild( 0, child );

    EXPECT_NO_THROW( child->destroy() );
    EXPECT_NO_THROW( m_selector->destroy() );
}
