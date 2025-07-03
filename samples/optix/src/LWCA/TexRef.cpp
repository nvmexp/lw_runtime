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

#include <LWCA/TexRef.h>

#include <LWCA/Array.h>
#include <LWCA/ErrorCheck.h>
#include <LWCA/Function.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;


TexRef::TexRef()
    : m_texRef( nullptr )
{
}

TexRef::TexRef( LWtexref ref )
    : m_texRef( ref )
{
}

LWtexref TexRef::get()
{
    return m_texRef;
}

const LWtexref TexRef::get() const
{
    return m_texRef;
}

LWdeviceptr TexRef::getAddress( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWdeviceptr addr = 0;
    CHECK( lwdaDriver().LwTexRefGetAddress( &addr, m_texRef ) );
    return addr;
}

LWaddress_mode TexRef::getAddressMode( int dim, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWaddress_mode mode;
    CHECK( lwdaDriver().LwTexRefGetAddressMode( &mode, m_texRef, dim ) );
    return mode;
}

Array TexRef::getArray( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWarray array = nullptr;
    CHECK( lwdaDriver().LwTexRefGetArray( &array, m_texRef ) );
    return Array( array );
}

LWfilter_mode TexRef::getFilterMode( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWfilter_mode mode;
    CHECK( lwdaDriver().LwTexRefGetFilterMode( &mode, m_texRef ) );
    return mode;
}

unsigned int TexRef::getFlags( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    unsigned int flags = 0;
    CHECK( lwdaDriver().LwTexRefGetFlags( &flags, m_texRef ) );
    return flags;
}

void TexRef::getFormat( LWarray_format& format, int& numChannels, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefGetFormat( &format, &numChannels, m_texRef ) );
}

int TexRef::getMaxAnisotropy( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    int aniso = 0;
    CHECK( lwdaDriver().LwTexRefGetMaxAnisotropy( &aniso, m_texRef ) );
    return aniso;
}

LWfilter_mode TexRef::getMipmapFilterMode( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWfilter_mode mode;
    CHECK( lwdaDriver().LwTexRefGetMipmapFilterMode( &mode, m_texRef ) );
    return mode;
}

float TexRef::getMipmapLevelBias( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    float bias;
    CHECK( lwdaDriver().LwTexRefGetMipmapLevelBias( &bias, m_texRef ) );
    return bias;
}

void TexRef::getMipmapLevelClamp( float& minMipmapLevelClamp, float& maxMipmapLevelClamp, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefGetMipmapLevelClamp( &minMipmapLevelClamp, &maxMipmapLevelClamp, m_texRef ) );
}

MipmappedArray TexRef::getMipmappedArray( LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    LWmipmappedArray array;
    CHECK( lwdaDriver().LwTexRefGetMipmappedArray( &array, m_texRef ) );
    return MipmappedArray( array );
}

size_t TexRef::setAddress( LWdeviceptr dptr, size_t bytes, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    size_t byteOffset;
    CHECK( lwdaDriver().LwTexRefSetAddress( &byteOffset, m_texRef, dptr, bytes ) );
    return byteOffset;
}

void TexRef::setAddress2D( const LWDA_ARRAY_DESCRIPTOR& desc, LWdeviceptr dptr, size_t Pitch, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetAddress2D( m_texRef, &desc, dptr, Pitch ) );
}

void TexRef::setAddressMode( int dim, LWaddress_mode am, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetAddressMode( m_texRef, dim, am ) );
}

void TexRef::setArray( const Array& array, unsigned int Flags, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetArray( m_texRef, array.get(), Flags ) );
}

void TexRef::setFilterMode( LWfilter_mode fm, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetFilterMode( m_texRef, fm ) );
}

void TexRef::setFlags( unsigned int Flags, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetFlags( m_texRef, Flags ) );
}

void TexRef::setFormat( LWarray_format fmt, int numPackedComponents, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetFormat( m_texRef, fmt, numPackedComponents ) );
}

void TexRef::setMaxAnisotropy( unsigned int maxAniso, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetMaxAnisotropy( m_texRef, maxAniso ) );
}

void TexRef::setMipmapFilterMode( LWfilter_mode fm, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetMipmapFilterMode( m_texRef, fm ) );
}

void TexRef::setMipmapLevelBias( float bias, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetMipmapLevelBias( m_texRef, bias ) );
}

void TexRef::setMipmapLevelClamp( float minMipmapLevelClamp, float maxMipmapLevelClamp, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetMipmapLevelClamp( m_texRef, minMipmapLevelClamp, maxMipmapLevelClamp ) );
}

void TexRef::setMipmappedArray( const MipmappedArray& mipmappedArray, unsigned int Flags, LWresult* returnResult )
{
    RT_ASSERT( m_texRef != nullptr );
    CHECK( lwdaDriver().LwTexRefSetMipmappedArray( m_texRef, mipmappedArray.get(), Flags ) );
}
