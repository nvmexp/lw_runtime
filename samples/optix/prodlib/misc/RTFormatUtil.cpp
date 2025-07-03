// Copyright LWPU Corporation 2010
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <prodlib/misc/RTFormatUtil.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>

#include <o6/optix.h>

namespace prodlib {


bool isSupportedTextureFormat( RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF4:
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT4:
        case RT_FORMAT_BYTE:
        case RT_FORMAT_BYTE2:
        case RT_FORMAT_BYTE4:
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE4:
        case RT_FORMAT_SHORT:
        case RT_FORMAT_SHORT2:
        case RT_FORMAT_SHORT4:
        case RT_FORMAT_UNSIGNED_SHORT:
        case RT_FORMAT_UNSIGNED_SHORT2:
        case RT_FORMAT_UNSIGNED_SHORT4:
        case RT_FORMAT_INT:
        case RT_FORMAT_INT2:
        case RT_FORMAT_INT4:
        case RT_FORMAT_UNSIGNED_INT:
        case RT_FORMAT_UNSIGNED_INT2:
        case RT_FORMAT_UNSIGNED_INT4:
        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return true;
        default:
            return false;
    }
}

unsigned int getElementSize( RTformat fmt )
{
    switch( fmt )
    {
        case RT_FORMAT_FLOAT:
            return sizeof( float );
        case RT_FORMAT_FLOAT2:
            return sizeof( float ) * 2;
        case RT_FORMAT_FLOAT3:
            return sizeof( float ) * 3;
        case RT_FORMAT_FLOAT4:
            return sizeof( float ) * 4;
        case RT_FORMAT_BYTE:
            return sizeof( char );
        case RT_FORMAT_BYTE2:
            return sizeof( char ) * 2;
        case RT_FORMAT_BYTE3:
            return sizeof( char ) * 3;
        case RT_FORMAT_BYTE4:
            return sizeof( char ) * 4;
        case RT_FORMAT_UNSIGNED_BYTE:
            return sizeof( unsigned char );
        case RT_FORMAT_UNSIGNED_BYTE2:
            return sizeof( unsigned char ) * 2;
        case RT_FORMAT_UNSIGNED_BYTE3:
            return sizeof( unsigned char ) * 3;
        case RT_FORMAT_UNSIGNED_BYTE4:
            return sizeof( unsigned char ) * 4;
        case RT_FORMAT_SHORT:
            return sizeof( short );
        case RT_FORMAT_SHORT2:
            return sizeof( short ) * 2;
        case RT_FORMAT_SHORT3:
            return sizeof( short ) * 3;
        case RT_FORMAT_SHORT4:
            return sizeof( short ) * 4;
        case RT_FORMAT_UNSIGNED_SHORT:
            return sizeof( unsigned short );
        case RT_FORMAT_UNSIGNED_SHORT2:
            return sizeof( unsigned short ) * 2;
        case RT_FORMAT_UNSIGNED_SHORT3:
            return sizeof( unsigned short ) * 3;
        case RT_FORMAT_UNSIGNED_SHORT4:
            return sizeof( unsigned short ) * 4;
        case RT_FORMAT_INT:
            return sizeof( int );
        case RT_FORMAT_INT2:
            return sizeof( int ) * 2;
        case RT_FORMAT_INT3:
            return sizeof( int ) * 3;
        case RT_FORMAT_INT4:
            return sizeof( int ) * 4;
        case RT_FORMAT_UNSIGNED_INT:
            return sizeof( unsigned int );
        case RT_FORMAT_UNSIGNED_INT2:
            return sizeof( unsigned int ) * 2;
        case RT_FORMAT_UNSIGNED_INT3:
            return sizeof( unsigned int ) * 3;
        case RT_FORMAT_UNSIGNED_INT4:
            return sizeof( unsigned int ) * 4;
        case RT_FORMAT_LONG_LONG:
            return sizeof( long long );
        case RT_FORMAT_LONG_LONG2:
            return sizeof( long long ) * 2;
        case RT_FORMAT_LONG_LONG3:
            return sizeof( long long ) * 3;
        case RT_FORMAT_LONG_LONG4:
            return sizeof( long long ) * 4;
        case RT_FORMAT_UNSIGNED_LONG_LONG:
            return sizeof( unsigned long long );
        case RT_FORMAT_UNSIGNED_LONG_LONG2:
            return sizeof( unsigned long long ) * 2;
        case RT_FORMAT_UNSIGNED_LONG_LONG3:
            return sizeof( unsigned long long ) * 3;
        case RT_FORMAT_UNSIGNED_LONG_LONG4:
            return sizeof( unsigned long long ) * 4;
        case RT_FORMAT_BUFFER_ID:
            return sizeof( unsigned int );
        case RT_FORMAT_PROGRAM_ID:
            return sizeof( int );
        case RT_FORMAT_USER:
            return 0;
        case RT_FORMAT_UNKNOWN:
            return 0;
        case RT_FORMAT_HALF:
            return sizeof( short );
        case RT_FORMAT_HALF2:
            return sizeof( short ) * 2;
        case RT_FORMAT_HALF3:
            return sizeof( short ) * 3;
        case RT_FORMAT_HALF4:
            return sizeof( short ) * 4;

        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
            return sizeof( unsigned ) * 2;

        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return sizeof( unsigned ) * 4;

        default:
            throw IlwalidValue( RT_EXCEPTION_INFO, "Unexpected buffer format: ", fmt );
    }
    return 0;
}

int getNumElements( RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_HALF:
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_SHORT:
        case RT_FORMAT_UNSIGNED_SHORT:
        case RT_FORMAT_INT:
        case RT_FORMAT_UNSIGNED_INT:
        case RT_FORMAT_LONG_LONG:
        case RT_FORMAT_UNSIGNED_LONG_LONG:
            return 1;

        case RT_FORMAT_HALF2:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_SHORT2:
        case RT_FORMAT_UNSIGNED_SHORT2:
        case RT_FORMAT_INT2:
        case RT_FORMAT_UNSIGNED_INT2:
        case RT_FORMAT_LONG_LONG2:
        case RT_FORMAT_UNSIGNED_LONG_LONG2:
            return 2;

        case RT_FORMAT_HALF3:
        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_SHORT3:
        case RT_FORMAT_UNSIGNED_SHORT3:
        case RT_FORMAT_INT3:
        case RT_FORMAT_UNSIGNED_INT3:
        case RT_FORMAT_LONG_LONG3:
        case RT_FORMAT_UNSIGNED_LONG_LONG3:
            return 3;

        case RT_FORMAT_HALF4:
        case RT_FORMAT_FLOAT4:
        case RT_FORMAT_BYTE4:
        case RT_FORMAT_UNSIGNED_BYTE4:
        case RT_FORMAT_SHORT4:
        case RT_FORMAT_UNSIGNED_SHORT4:
        case RT_FORMAT_INT4:
        case RT_FORMAT_UNSIGNED_INT4:
        case RT_FORMAT_LONG_LONG4:
        case RT_FORMAT_UNSIGNED_LONG_LONG4:
            return 4;

        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
            return 2;

        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return 4;

        case RT_FORMAT_USER:
        case RT_FORMAT_BUFFER_ID:
        case RT_FORMAT_PROGRAM_ID:
        case RT_FORMAT_UNKNOWN:
            throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format for LWCA array: ", format );
    }
    throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown buffer format: ", format );
}

bool isVector( RTformat format )
{
    switch( format )
    {
        // Prevent exceptions.
        case RT_FORMAT_USER:
        case RT_FORMAT_BUFFER_ID:
        case RT_FORMAT_PROGRAM_ID:
        case RT_FORMAT_UNKNOWN:
            return false;
        default:
            return getNumElements( format ) > 1;
    }
}

std::string toString( RTformat val, size_t userSize )
{
    switch( val )
    {
        case RT_FORMAT_UNKNOWN:
            return "unknown";
        case RT_FORMAT_HALF:
            return "half";
        case RT_FORMAT_HALF2:
            return "half2";
        case RT_FORMAT_HALF3:
            return "half3";
        case RT_FORMAT_HALF4:
            return "half4";
        case RT_FORMAT_FLOAT:
            return "float";
        case RT_FORMAT_FLOAT2:
            return "float2";
        case RT_FORMAT_FLOAT3:
            return "float3";
        case RT_FORMAT_FLOAT4:
            return "float4";
        case RT_FORMAT_BYTE:
            return "char";
        case RT_FORMAT_BYTE2:
            return "char2";
        case RT_FORMAT_BYTE3:
            return "char3";
        case RT_FORMAT_BYTE4:
            return "char4";
        case RT_FORMAT_UNSIGNED_BYTE:
            return "uchar";
        case RT_FORMAT_UNSIGNED_BYTE2:
            return "uchar2";
        case RT_FORMAT_UNSIGNED_BYTE3:
            return "uchar3";
        case RT_FORMAT_UNSIGNED_BYTE4:
            return "uchar4";
        case RT_FORMAT_SHORT:
            return "short";
        case RT_FORMAT_SHORT2:
            return "short2";
        case RT_FORMAT_SHORT3:
            return "short3";
        case RT_FORMAT_SHORT4:
            return "short4";
        case RT_FORMAT_UNSIGNED_SHORT:
            return "ushort";
        case RT_FORMAT_UNSIGNED_SHORT2:
            return "ushort2";
        case RT_FORMAT_UNSIGNED_SHORT3:
            return "ushort3";
        case RT_FORMAT_UNSIGNED_SHORT4:
            return "ushort4";
        case RT_FORMAT_INT:
            return "int";
        case RT_FORMAT_INT2:
            return "int2";
        case RT_FORMAT_INT3:
            return "int3";
        case RT_FORMAT_INT4:
            return "int4";
        case RT_FORMAT_UNSIGNED_INT:
            return "uint";
        case RT_FORMAT_UNSIGNED_INT2:
            return "uint2";
        case RT_FORMAT_UNSIGNED_INT3:
            return "uint3";
        case RT_FORMAT_UNSIGNED_INT4:
            return "uint4";
        case RT_FORMAT_LONG_LONG:
            return "longlong";
        case RT_FORMAT_LONG_LONG2:
            return "longlong2";
        case RT_FORMAT_LONG_LONG3:
            return "longlong3";
        case RT_FORMAT_LONG_LONG4:
            return "longlong4";
        case RT_FORMAT_UNSIGNED_LONG_LONG:
            return "ulonglong";
        case RT_FORMAT_UNSIGNED_LONG_LONG2:
            return "ulonglong2";
        case RT_FORMAT_UNSIGNED_LONG_LONG3:
            return "ulonglong3";
        case RT_FORMAT_UNSIGNED_LONG_LONG4:
            return "ulonglong4";
        case RT_FORMAT_UNSIGNED_BC1:
            return "ubc1";
        case RT_FORMAT_UNSIGNED_BC2:
            return "ubc2";
        case RT_FORMAT_UNSIGNED_BC3:
            return "ubc3";
        case RT_FORMAT_UNSIGNED_BC4:
            return "ubc4";
        case RT_FORMAT_BC4:
            return "bc4";
        case RT_FORMAT_UNSIGNED_BC5:
            return "ubc5";
        case RT_FORMAT_BC5:
            return "bc5";
        case RT_FORMAT_UNSIGNED_BC6H:
            return "ubc6h";
        case RT_FORMAT_BC6H:
            return "bc6h";
        case RT_FORMAT_UNSIGNED_BC7:
            return "ubc7";
        case RT_FORMAT_USER:
        {
            std::ostringstream out;
            out << "u" << userSize << "b";
            return out.str();
        }
        case RT_FORMAT_BUFFER_ID:
            return "bufferid";
        case RT_FORMAT_PROGRAM_ID:
            return "programid";
        default:
            RT_ASSERT( !!!"Unknown enumerant specified to format2string" );
    }
    return "";
}

bool isCompressed( const RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return true;

        default:
            return false;
    }
}


}  // end namespace prodlib
