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

#include <FrontEnd/Canonical/IntrinsicsManager.h>

#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>

#include <srcTests.h>

#include <FrontEnd/Canonical/tests/IntrinsicNameTest.h>

using namespace optix;
using namespace optix::testing;

class OptixIntrinsicNameTest : public IntrinsicNameTest
{
  public:
    OptixIntrinsicNameTest()
        : IntrinsicNameTest( "optixi_some_intrinsic" )
    {
    }
};

TEST_F( OptixIntrinsicNameTest, class_of_call )
{
    ASSERT_TRUE( OptixIntrinsic::classof( createCall() ) );
}

TEST_F( OptixIntrinsicNameTest, class_of_value )
{
    ASSERT_TRUE( OptixIntrinsic::classof( createValue() ) );
}

TEST_F( OptixIntrinsicNameTest, isIntrinsic )
{
    ASSERT_TRUE( OptixIntrinsic::isIntrinsic( createFunction() ) );
}

class AtomicSetBufferElementNameTest : public VariableReferenceUniqueNameTest
{
  public:
    AtomicSetBufferElementNameTest()
        : VariableReferenceUniqueNameTest( "optixi_atomicSetBufferElement" )
    {
    }
};

TEST_F( AtomicSetBufferElementNameTest, class_of_call )
{
    ASSERT_TRUE( AtomicSetBufferElement::classof( createCall() ) );
}

TEST_F( AtomicSetBufferElementNameTest, class_of_value )
{
    ASSERT_TRUE( AtomicSetBufferElement::classof( createValue() ) );
}

TEST_F( AtomicSetBufferElementNameTest, createUniqueName )
{
    const CanonicalProgram  parent( nullptr, 0U );
    const VariableReference varRef( &parent );

    const std::string uniqueName = AtomicSetBufferElement::createUniqueName( &varRef );

    ASSERT_EQ( m_prefix + "..", uniqueName );
}

TEST_F( AtomicSetBufferElementNameTest, class_of_call_ilwalid_variable_reference_unique_name )
{
    const char* ilwalidUniqueNames[] = {"_missing_separating_dot_ptx0xdeadbeef.no_more_dots",
                                        ".arbitrary_text_no_p_t_x0xdeadbeef.no_more_dots",
                                        ".arbitrary_text_ptx_not_hex.no_more_dots",
                                        ".arbitrary_text_ptx0xdeadbeef_missing_dot",
                                        ".arbitrary_text_ptx0xdeadbeef.extra.dot"};

    for( const char* name : ilwalidUniqueNames )
    {
        ASSERT_FALSE( AtomicSetBufferElement::classof( createCall( m_prefix + name ) ) );
    }
}

TEST_F( AtomicSetBufferElementNameTest, isIntrinsic )
{
    llvm::Function* fn = createFunction();

    ASSERT_TRUE( AtomicSetBufferElement::isIntrinsic( fn ) );
}

TEST_F( AtomicSetBufferElementNameTest, parseUniqueName )
{
    llvm::Function* fn = createFunction();

    ASSERT_EQ( m_uniqueName, AtomicSetBufferElement::parseUniqueName( fn->getName() ) );
}

class AtomicSetBufferElementFromIdNameTest : public VariableReferenceUniqueNameTest
{
  public:
    AtomicSetBufferElementFromIdNameTest()
        : VariableReferenceUniqueNameTest( "optixi_atomicSetBufferElementFromId" )
    {
    }
};

TEST_F( AtomicSetBufferElementFromIdNameTest, class_of_call )
{
    ASSERT_TRUE( AtomicSetBufferElementFromId::classof( createCall() ) );
}

TEST_F( AtomicSetBufferElementFromIdNameTest, class_of_value )
{
    ASSERT_TRUE( AtomicSetBufferElementFromId::classof( createValue() ) );
}

TEST_F( AtomicSetBufferElementFromIdNameTest, isIntrinsic )
{
    ASSERT_TRUE( AtomicSetBufferElementFromId::isIntrinsic( createFunction() ) );
}

TEST_F( AtomicSetBufferElementFromIdNameTest, createUniqueName )
{
    const unsigned int dimensions  = 2U;
    const size_t       elementSize = 16U;

    const std::string uniqueName = AtomicSetBufferElementFromId::createUniqueName( dimensions, elementSize );

    ASSERT_EQ( m_prefix + "2.16", uniqueName );
}

class TraceGlobalPayloadCallNameTest : public CanonicalProgramUniqueNameTest
{
  public:
    TraceGlobalPayloadCallNameTest()
        : CanonicalProgramUniqueNameTest( "optixi_trace_global_payload" )
    {
    }
};

TEST_F( TraceGlobalPayloadCallNameTest, class_of_call )
{
    ASSERT_TRUE( TraceGlobalPayloadCall::classof( createCall() ) );
}

TEST_F( TraceGlobalPayloadCallNameTest, class_of_value )
{
    ASSERT_TRUE( TraceGlobalPayloadCall::classof( createValue() ) );
}

TEST_F( TraceGlobalPayloadCallNameTest, isIntrinsic )
{
    ASSERT_TRUE( TraceGlobalPayloadCall::isIntrinsic( createFunction() ) );
}

class GetBufferElementAddressNameTest : public VariableReferenceUniqueNameTest
{
  public:
    GetBufferElementAddressNameTest()
        : VariableReferenceUniqueNameTest( "optixi_getBufferElementAddress", "arbitrary_suffix" )
    {
    }
};

TEST_F( GetBufferElementAddressNameTest, class_of_call )
{
    ASSERT_TRUE( GetBufferElementAddress::classof( createCall() ) );
}

TEST_F( GetBufferElementAddressNameTest, class_of_value )
{
    ASSERT_TRUE( GetBufferElementAddress::classof( createValue() ) );
}

TEST_F( GetBufferElementAddressNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetBufferElementAddress::isIntrinsic( createFunction() ) );
}

TEST_F( GetBufferElementAddressNameTest, isIntrinsic_unique_name )
{
    ASSERT_TRUE( GetBufferElementAddress::isIntrinsic( createFunction(), m_uniqueName + m_uniqueNameSuffix ) );
    ASSERT_FALSE( GetBufferElementAddress::isIntrinsic( createFunction(), "mismatch" ) );
}

TEST_F( GetBufferElementAddressNameTest, parseUniqueName )
{
    ASSERT_EQ( m_uniqueName + m_uniqueNameSuffix, GetBufferElementAddress::parseUniqueName( createFunction()->getName() ) );
}

TEST_F( GetBufferElementAddressNameTest, createUniqueName )
{
    const CanonicalProgram  parent( nullptr, 0U );
    const VariableReference varRef( &parent );


    const std::string uniqueName = GetBufferElementAddress::createUniqueName( &varRef );

    ASSERT_EQ( m_prefix + "..", uniqueName );
}

class GetBufferElementAddressFromIdNameTest : public IntrinsicNameTest
{
  public:
    GetBufferElementAddressFromIdNameTest()
        : IntrinsicNameTest( "optixi_getBufferElementAddressFromId", "arbitrary_text" )
    {
    }
};

TEST_F( GetBufferElementAddressFromIdNameTest, class_of_call )
{
    ASSERT_TRUE( GetBufferElementAddressFromId::classof( createCall() ) );
}

TEST_F( GetBufferElementAddressFromIdNameTest, class_of_value )
{
    ASSERT_TRUE( GetBufferElementAddressFromId::classof( createValue() ) );
}

TEST_F( GetBufferElementAddressFromIdNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetBufferElementAddressFromId::isIntrinsic( createFunction() ) );
}

TEST_F( GetBufferElementAddressFromIdNameTest, createUniqueName )
{
    const unsigned int dimensionality = 2;

    const std::string uniqueName = GetBufferElementAddressFromId::createUniqueName( dimensionality );

    ASSERT_EQ( m_prefix + ".2", uniqueName );
}

class GetPayloadAddressCallNameTest : public IntrinsicNameTest
{
  public:
    GetPayloadAddressCallNameTest()
        : IntrinsicNameTest( "optixi_get_payload_address" )
    {
    }
};

TEST_F( GetPayloadAddressCallNameTest, class_of_call )
{
    ASSERT_TRUE( GetPayloadAddressCall::classof( createCall() ) );
}

TEST_F( GetPayloadAddressCallNameTest, class_of_value )
{
    ASSERT_TRUE( GetPayloadAddressCall::classof( createValue() ) );
}

TEST_F( GetPayloadAddressCallNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetPayloadAddressCall::isIntrinsic( createFunction() ) );
}

class GetBufferElementNameTest : public VariableReferenceUniqueNameTest
{
  public:
    GetBufferElementNameTest()
        : VariableReferenceUniqueNameTest( "optixi_getBufferElement", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( GetBufferElementNameTest, class_of_call )
{
    ASSERT_TRUE( GetBufferElement::classof( createCall() ) );
}

TEST_F( GetBufferElementNameTest, class_of_value )
{
    ASSERT_TRUE( GetBufferElement::classof( createValue() ) );
}

TEST_F( GetBufferElementNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetBufferElement::isIntrinsic( createFunction() ) );
}

TEST_F( GetBufferElementNameTest, isIntrinsic_with_unique_name )
{
    ASSERT_TRUE( GetBufferElement::isIntrinsic( createFunction(), m_uniqueName ) );
}

TEST_F( GetBufferElementNameTest, parseqUniqueName )
{
    ASSERT_EQ( m_uniqueName, GetBufferElement::parseUniqueName( createFunction()->getName() ) );
}

TEST_F( GetBufferElementNameTest, createUniqueName )
{
    const CanonicalProgram  parent( nullptr, 0U );
    const VariableReference varRef( &parent );

    const std::string uniqueName = GetBufferElement::createUniqueName( &varRef );

    ASSERT_EQ( m_prefix + "..", uniqueName );
}

class LoadOrRequestBufferElementNameTest : public VariableReferenceUniqueNameTest
{
  public:
    LoadOrRequestBufferElementNameTest()
        : VariableReferenceUniqueNameTest( "optixi_loadOrRequestBufferElement" )
    {
    }
};

TEST_F( LoadOrRequestBufferElementNameTest, class_of_call )
{
    ASSERT_TRUE( LoadOrRequestBufferElement::classof( createCall() ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, class_of_value )
{
    ASSERT_TRUE( LoadOrRequestBufferElement::classof( createValue() ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, isIntrinsic )
{
    ASSERT_TRUE( LoadOrRequestBufferElement::isIntrinsic( createFunction() ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, isIntrinsic_example )
{
    ASSERT_TRUE( LoadOrRequestBufferElement::isIntrinsic( createFunction(
        "optixi_loadOrRequestBufferElement._Z29closest_hit_radiance_texturedv_ptx0xcbb2beb694d249f2.demanded" ) ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, isIntrinsic_with_unique_name )
{
    ASSERT_TRUE( LoadOrRequestBufferElement::isIntrinsic( createFunction(), m_uniqueName ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, parseUniqueName )
{
    ASSERT_EQ( m_uniqueName, LoadOrRequestBufferElement::parseUniqueName( createFunction()->getName() ) );
}

TEST_F( LoadOrRequestBufferElementNameTest, createUniqueName )
{
    const CanonicalProgram  parent( nullptr, 0U );
    const VariableReference varRef( &parent );

    const std::string uniqueName = LoadOrRequestBufferElement::createUniqueName( &varRef );

    ASSERT_EQ( m_prefix + "..", uniqueName );
}

TEST_F( LoadOrRequestBufferElementNameTest, getDimensionality )
{
    llvm::Type*              int32Ty = llvm::Type::getInt32Ty( m_context );
    std::vector<llvm::Type*> argTypes{6, int32Ty};
    llvm::FunctionType*      fnType = llvm::FunctionType::get( int32Ty, argTypes, false );
    llvm::Function*          fn     = llvm::Function::Create( fnType, llvm::Function::ExternalLinkage, m_name.c_str() );

    ASSERT_EQ( 3U, LoadOrRequestBufferElement::getDimensionality( fn ) );
}

class LoadOrRequestTextureElementTest : public IntrinsicNameTest
{
  public:
    LoadOrRequestTextureElementTest()
        : IntrinsicNameTest( "optixi_textureLoadOrRequest" )
    {
    }
};

TEST_F( LoadOrRequestTextureElementTest, isIntrinsic )
{
    ASSERT_TRUE( LoadOrRequestTextureElement::isIntrinsic( createFunction( "optixi_textureLoadOrRequest2" ) ) );
}

TEST_F( LoadOrRequestTextureElementTest, isIntrinsicLod )
{
    ASSERT_TRUE( LoadOrRequestTextureElement::isIntrinsic( createFunction( "optixi_textureLodLoadOrRequest2" ) ) );
}

TEST_F( LoadOrRequestTextureElementTest, isIntrinsicGrad )
{
    ASSERT_TRUE( LoadOrRequestTextureElement::isIntrinsic( createFunction( "optixi_textureGradLoadOrRequest2" ) ) );
}

TEST_F( LoadOrRequestTextureElementTest, getKind )
{
    const char*                  name = "optixi_textureLoadOrRequest2";
    LoadOrRequestTextureElement* call = llvm::dyn_cast<LoadOrRequestTextureElement>( createCall( name ) );
    ASSERT_TRUE( call != nullptr );
    EXPECT_EQ( LoadOrRequestTextureElement::Nomip, call->getKind() );
}

TEST_F( LoadOrRequestTextureElementTest, getKindLod )
{
    const char*                  name = "optixi_textureLodLoadOrRequest2";
    LoadOrRequestTextureElement* call = llvm::dyn_cast<LoadOrRequestTextureElement>( createCall( name ) );
    ASSERT_TRUE( call != nullptr );
    EXPECT_EQ( LoadOrRequestTextureElement::Lod, call->getKind() );
}

TEST_F( LoadOrRequestTextureElementTest, getKindGrad )
{
    const char*                  name = "optixi_textureGradLoadOrRequest2";
    LoadOrRequestTextureElement* call = llvm::dyn_cast<LoadOrRequestTextureElement>( createCall( name ) );
    ASSERT_TRUE( call != nullptr );
    EXPECT_EQ( LoadOrRequestTextureElement::Grad, call->getKind() );
}

class SetBufferElementNameTest : public VariableReferenceUniqueNameTest
{
  public:
    SetBufferElementNameTest()
        : VariableReferenceUniqueNameTest( "optixi_setBufferElement", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( SetBufferElementNameTest, class_of_call )
{
    ASSERT_TRUE( SetBufferElement::classof( createCall() ) );
}

TEST_F( SetBufferElementNameTest, class_of_value )
{
    ASSERT_TRUE( SetBufferElement::classof( createValue() ) );
}

TEST_F( SetBufferElementNameTest, isIntrinsic )
{
    ASSERT_TRUE( SetBufferElement::isIntrinsic( createFunction() ) );
}

TEST_F( SetBufferElementNameTest, isIntrinsic_with_unique_name )
{
    ASSERT_TRUE( SetBufferElement::isIntrinsic( createFunction(), m_uniqueName ) );
}

TEST_F( SetBufferElementNameTest, parseqUniqueName )
{
    ASSERT_EQ( m_uniqueName, SetBufferElement::parseUniqueName( createFunction()->getName() ) );
}

TEST_F( SetBufferElementNameTest, createUniqueName )
{
    const CanonicalProgram  parent( nullptr, 0U );
    const VariableReference varRef( &parent );

    const std::string uniqueName = SetBufferElement::createUniqueName( &varRef );

    ASSERT_EQ( m_prefix + "..", uniqueName );
}

class GetBufferElementFromIdNameTest : public IntrinsicNameTest
{
  public:
    GetBufferElementFromIdNameTest()
        : IntrinsicNameTest( "optixi_getBufferElementFromId", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( GetBufferElementFromIdNameTest, class_of_call )
{
    ASSERT_TRUE( GetBufferElementFromId::classof( createCall() ) );
}

TEST_F( GetBufferElementFromIdNameTest, class_of_value )
{
    ASSERT_TRUE( GetBufferElementFromId::classof( createValue() ) );
}

TEST_F( GetBufferElementFromIdNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetBufferElementFromId::isIntrinsic( createFunction() ) );
}

TEST_F( GetBufferElementFromIdNameTest, createUniqueName )
{
    const unsigned int dimensionality = 2;
    llvm::Type*        valueType      = llvm::Type::getInt32Ty( m_context );

    const std::string uniqueName = GetBufferElementFromId::createUniqueName( dimensionality, valueType );

    ASSERT_EQ( m_prefix + ".2.i32", uniqueName );
}

class SetBufferElementFromIdNameTest : public IntrinsicNameTest
{
  public:
    SetBufferElementFromIdNameTest()
        : IntrinsicNameTest( "optixi_setBufferElementFromId", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( SetBufferElementFromIdNameTest, class_of_call )
{
    ASSERT_TRUE( SetBufferElementFromId::classof( createCall() ) );
}

TEST_F( SetBufferElementFromIdNameTest, class_of_value )
{
    ASSERT_TRUE( SetBufferElementFromId::classof( createValue() ) );
}

TEST_F( SetBufferElementFromIdNameTest, isIntrinsic )
{
    ASSERT_TRUE( SetBufferElementFromId::isIntrinsic( createFunction() ) );
}

TEST_F( SetBufferElementFromIdNameTest, createUniqueName )
{
    const unsigned int dimensionality = 2;
    llvm::Type*        valueType      = llvm::Type::getInt32Ty( m_context );

    const std::string uniqueName = SetBufferElementFromId::createUniqueName( dimensionality, valueType );

    ASSERT_EQ( m_prefix + ".2.i32", uniqueName );
}

class SetAttributeValueNameTest : public VariableReferenceUniqueNameTest
{
  public:
    SetAttributeValueNameTest()
        : VariableReferenceUniqueNameTest( "optixi_setAttributeValue", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( SetAttributeValueNameTest, class_of_call )
{
    ASSERT_TRUE( SetAttributeValue::classof( createCall() ) );
}

TEST_F( SetAttributeValueNameTest, isIntrinsic )
{
    ASSERT_TRUE( SetAttributeValue::isIntrinsic( createFunction() ) );
}

TEST_F( SetAttributeValueNameTest, parseUniqueName_valid_name )
{
    llvm::StringRef uniqueName;

    ASSERT_TRUE( SetAttributeValue::parseUniqueName( createFunction(), uniqueName ) );
    ASSERT_EQ( m_uniqueName, uniqueName );
}

TEST_F( SetAttributeValueNameTest, parseUniqueName_not_valid_name )
{
    llvm::StringRef uniqueName;

    ASSERT_FALSE( SetAttributeValue::parseUniqueName( createFunction( "some_other_function" ), uniqueName ) );
    ASSERT_EQ( llvm::StringRef{}, uniqueName );
}

class GetAttributeValueNameTest : public VariableReferenceUniqueNameTest
{
  public:
    GetAttributeValueNameTest()
        : VariableReferenceUniqueNameTest( "optixi_getAttributeValue", ".arbitrary_suffix_starting_with_dot" )
    {
    }
};

TEST_F( GetAttributeValueNameTest, class_of_call )
{
    ASSERT_TRUE( GetAttributeValue::classof( createCall() ) );
}

TEST_F( GetAttributeValueNameTest, isIntrinsic )
{
    ASSERT_TRUE( GetAttributeValue::isIntrinsic( createFunction() ) );
}

TEST_F( GetAttributeValueNameTest, parseUniqueName_valid_name )
{
    llvm::StringRef uniqueName;

    ASSERT_TRUE( GetAttributeValue::parseUniqueName( createFunction(), uniqueName ) );
    ASSERT_EQ( m_uniqueName, uniqueName );
}

TEST_F( GetAttributeValueNameTest, parseUniqueName_not_valid_name )
{
    llvm::StringRef uniqueName;

    ASSERT_FALSE( GetAttributeValue::parseUniqueName( createFunction( "some_other_function" ), uniqueName ) );
    ASSERT_EQ( llvm::StringRef{}, uniqueName );
}

class ReportFullIntersectionNameTest : public UniqueNameTest
{
  public:
    ReportFullIntersectionNameTest()
        : UniqueNameTest( "optixi_reportFullIntersection",
                          "identifier_with_underscores_and_d1g1t5",
                          ".arbitrary_text_after_dot" )
    {
    }
};

TEST_F( ReportFullIntersectionNameTest, class_of_call )
{
    EXPECT_TRUE( ReportFullIntersection::classof( createCall() ) );
}

TEST_F( ReportFullIntersectionNameTest, isIntrinsic )
{
    EXPECT_TRUE( ReportFullIntersection::isIntrinsic( createFunction() ) );
}

TEST_F( ReportFullIntersectionNameTest, parseUniqueName_valid_name )
{
    llvm::StringRef uniqueName;

    EXPECT_TRUE( ReportFullIntersection::parseUniqueName( createFunction(), uniqueName ) );
    EXPECT_EQ( m_uniqueName, uniqueName );
}

TEST_F( ReportFullIntersectionNameTest, parseUniqueName_not_valid_name )
{
    llvm::StringRef uniqueName;

    EXPECT_FALSE( ReportFullIntersection::parseUniqueName( createFunction( "some_other_function" ), uniqueName ) );
    EXPECT_EQ( llvm::StringRef{}, uniqueName );
}

class IsPotentialIntersectionNameTest : public IntrinsicNameTest
{
  public:
    IsPotentialIntersectionNameTest()
        : IntrinsicNameTest( "optixi_isPotentialIntersection" )
    {
    }
};

TEST_F( IsPotentialIntersectionNameTest, class_of_call )
{
    EXPECT_TRUE( IsPotentialIntersection::classof( createCall() ) );
}

TEST_F( IsPotentialIntersectionNameTest, isIntrinsic )
{
    EXPECT_TRUE( IsPotentialIntersection::isIntrinsic( createFunction() ) );
}
