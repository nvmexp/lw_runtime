//
// Copyright (c) 2019, LWPU CORPORATION.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES
//

#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>

#include <FrontEnd/Canonical/tests/IntrinsicNameTest.h>

using namespace optix;
using namespace optix::testing;

class RtxiLoadOrRequestBufferElementNameTest : public VariableReferenceUniqueNameTest
{
  public:
    RtxiLoadOrRequestBufferElementNameTest()
        : VariableReferenceUniqueNameTest( "rtxiLoadOrRequestBufferElement" )
    {
    }
};

TEST_F( RtxiLoadOrRequestBufferElementNameTest, class_of_value )
{
    ASSERT_TRUE( RtxiLoadOrRequestBufferElement::classof( createValue() ) );
}

TEST_F( RtxiLoadOrRequestBufferElementNameTest, class_of_call )
{
    ASSERT_TRUE( RtxiLoadOrRequestBufferElement::classof( createCall() ) );
}

TEST_F( RtxiLoadOrRequestBufferElementNameTest, isIntrinsic_function )
{
    ASSERT_TRUE( RtxiLoadOrRequestBufferElement::isIntrinsic( createFunction() ) );
}

TEST_F( RtxiLoadOrRequestBufferElementNameTest, isIntrinsic_unique_name )
{
    ASSERT_TRUE( RtxiLoadOrRequestBufferElement::isIntrinsic( createFunction(), m_uniqueName + m_uniqueNameSuffix ) );
    ASSERT_FALSE( RtxiLoadOrRequestBufferElement::isIntrinsic( createFunction(), "mismatch" ) );
}
