
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

#include <srcTests.h>

#include <Memory/tests/test_AllTextureModes.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/misc/RTFormatUtil.h>

#include <private/optix_6_enum_printers.h>

#include <optix_world.h>
#include <optixpp_namespace.h>

#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

using namespace optix;
using namespace prodlib;
using namespace testing;


bool operator==( FormatAndReadMode lhs, FormatAndReadMode rhs )
{
    return lhs.inputFmt == rhs.inputFmt && lhs.readMode == rhs.readMode;
}

std::ostream& operator<<( std::ostream& stream, FormatAndReadMode* value )
{
    return stream << '(' << value->inputFmt << ", " << value->readMode << ')';
}


template <typename T>
T valueA();
template <>
float valueA()
{
    return 0.9f;
}
template <>
char valueA()
{
    return 115;
}
template <>
unsigned char valueA()
{
    return 230;
}
template <>
short valueA()
{
    return 29490;
}
template <>
unsigned short valueA()
{
    return 58981;
}
template <>
int valueA()
{
    return 1932735282;
}
template <>
unsigned int valueA()
{
    return 3865470565;
}

template <typename T>
T valueB();
template <>
float valueB()
{
    return -0.1f;
}
template <>
char valueB()
{
    return -115;
}
template <>
unsigned char valueB()
{
    return 1;
}
template <>
short valueB()
{
    return -29490;
}
template <>
unsigned short valueB()
{
    return 1;
}
template <>
int valueB()
{
    return -1932735282;
}
template <>
unsigned int valueB()
{
    return 1;
}


template <typename T>
void fillBufferType( Buffer& buffer, const unsigned int numElements )
{
    T a       = valueA<T>();
    T b       = valueB<T>();
    T c       = a / 2 + b;
    T d       = c + a / 3;
    T e       = c + a / 5;
    T abcd[5] = {a, b, c, d, e};

    const int levels = buffer->getMipLevelCount();
    for( int level = 0; level < levels; level++ )
    {
        RTsize levelWidth, levelHeight, levelDepth;
        buffer->getMipLevelSize( level, levelWidth, levelHeight, levelDepth );
        T*  val   = (T*)buffer->map( level );
        int count = numElements * int( levelWidth * levelHeight * levelDepth );
        for( int i = 0; i < count; i++ )
        {
            val[i] = abcd[i % 5];
        }
        buffer->unmap( level );
    }
}

static unsigned short float2Half( float f )
{
    unsigned int   i = *(unsigned int*)( &f );
    unsigned short h;
    h = ( i >> 16 ) & 0x8000;
    if( ( i & 0x7f800000 ) == 0x7f800000 )
    {
        if( ( i & 0x7fffffff ) == 0x7f800000 )
        {
            h |= 0x7c00;  // Onfinity
        }
        else
        {
            h = 0x7fff;  // NaN
        }
    }
    else if( ( i & 0x7f800000 ) >= 0x33000000 )
    {
        int shift = (int)( ( i >> 23 ) & 0xff ) - 127;
        if( shift > 15 )
        {
            h |= 0x7c00;  // Infinity
        }
        else
        {
            i = ( i & 0x007fffff ) | 0x00800000;  // Mantissa
            if( shift < -14 )
            {
                // Denormal
                h |= i >> ( -1 - shift );
                i = i << ( 32 - ( -1 - shift ) );
            }
            else
            {
                // Normal
                h |= i >> ( 24 - 11 );
                i = i << ( 32 - ( 24 - 11 ) );
                h = h + ( ( 14 + shift ) << 10 );
            }
            // Round to nearest of even
            if( ( i > 0x80000000 ) || ( ( i == 0x80000000 ) && ( h & 1 ) ) )
            {
                h++;
            }
        }
    }
    return h;
}

void fillBufferHalfFloat( Buffer& buffer, const unsigned int numElements )
{
    float a       = valueA<float>();
    float b       = valueB<float>();
    float c       = a / 2 + b;
    float d       = c + a / 3;
    float e       = c + a / 5;
    float abcd[5] = {a, b, c, d, e};

    const int levels = buffer->getMipLevelCount();
    for( int level = 0; level < levels; level++ )
    {
        RTsize levelWidth, levelHeight, levelDepth;
        buffer->getMipLevelSize( level, levelWidth, levelHeight, levelDepth );
        unsigned short* val   = (unsigned short*)buffer->map( level );
        int             count = numElements * int( levelWidth * levelHeight * levelDepth );
        for( int i = 0; i < count; i++ )
        {
            val[i] = float2Half( abcd[i % 5] );
        }
        buffer->unmap( level );
    }
}

void fillBuffer( Buffer& buffer, RTformat fmt )
{
    const unsigned int numElements = getNumElements( fmt );
    switch( fmt )
    {
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF4:
            fillBufferHalfFloat( buffer, numElements );
            break;
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT4:
            fillBufferType<float>( buffer, numElements );
            break;
        case RT_FORMAT_BYTE:
        case RT_FORMAT_BYTE2:
        case RT_FORMAT_BYTE4:
            fillBufferType<char>( buffer, numElements );
            break;
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE4:
            fillBufferType<unsigned char>( buffer, numElements );
            break;
        case RT_FORMAT_SHORT:
        case RT_FORMAT_SHORT2:
        case RT_FORMAT_SHORT4:
            fillBufferType<short>( buffer, numElements );
            break;
        case RT_FORMAT_UNSIGNED_SHORT:
        case RT_FORMAT_UNSIGNED_SHORT2:
        case RT_FORMAT_UNSIGNED_SHORT4:
            fillBufferType<unsigned short>( buffer, numElements );
            break;
        case RT_FORMAT_INT:
        case RT_FORMAT_INT2:
        case RT_FORMAT_INT4:
            fillBufferType<int>( buffer, numElements );
            break;
        case RT_FORMAT_UNSIGNED_INT:
        case RT_FORMAT_UNSIGNED_INT2:
        case RT_FORMAT_UNSIGNED_INT4:
            fillBufferType<unsigned int>( buffer, numElements );
            break;
        default:
            ADD_FAILURE();
    }
}


std::string TexModesToStr( TexModes tex_opt )
{
    const char* indexStrs[]  = {"norm", "indx"};
    const char* wrapStrs[]   = {"repeat ", "clamp_e", "mirror ", "clamp_b"};
    const char* filterStrs[] = {"near ", "linear"};

    return std::string( indexStrs[tex_opt.index] ) + " " + wrapStrs[tex_opt.wrap] + " " + filterStrs[tex_opt.filter];
}


void dumpOutBuffer( std::ofstream& outFile, bool verbose, Buffer& buffer, const RTformat inputFmt, RTtexturereadmode readMode, float4 coords )
{
    float* data = (float*)buffer->map();
    for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
    {
        for( unsigned int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
        {

            for( unsigned int c = 0; c < 4; c++ )
            {
                float r = 0;
                if( readMode == RT_TEXTURE_READ_ELEMENT_TYPE || readMode == RT_TEXTURE_READ_ELEMENT_TYPE_SRGB || lookupIdx == TestTexture_size )
                    r = ( (float)( *(int*)( data ) ) );
                else
                    r = *data;
                data++;
                outFile << r << " ";
            }

            if( verbose )
            {
                outFile << toString( inputFmt, 0 ) << " " << READ_MODE_STRINGS[readMode] << " "
                        << LOOKUP_KIND_STRINGS[lookupIdx] << " " << TexModesToStr( tex_modes[m] ) << " "
                        << " coords=(" << coords.x << ", " << coords.y << ", " << coords.z << ", " << coords.w << ")";
            }

            outFile << "\n";
        }
    }
    buffer->unmap();
}

static void createTexture( Context& context, TextureSampler& sampler, const TexModes opts )
{
    sampler = context->createTextureSampler();
    sampler->setMaxAnisotropy( 16.0f );
    sampler->setMipLevelClamp( 0, 100.0f );
    sampler->setWrapMode( 0, (RTwrapmode)opts.wrap );
    sampler->setWrapMode( 1, (RTwrapmode)opts.wrap );
    sampler->setWrapMode( 2, (RTwrapmode)opts.wrap );
    sampler->setFilteringModes( (RTfiltermode)opts.filter, (RTfiltermode)opts.filter, (RTfiltermode)opts.filter );
    sampler->setIndexingMode( (RTtextureindexmode)opts.index );
}

class TestAllTextureModes : public Test
{
  public:
    Context      m_context;
    Buffer       m_outputBuffer;
    unsigned int m_outWidth  = 1024;
    unsigned int m_outHeight = 1;
    Buffer       m_buffer[NUMBER_OF_LOOKUP_KINDS][NUMBER_OF_FORMATS];
    std::map<std::pair<RTformat, int>, Buffer*> m_formatToBuffer;
    TextureSampler      m_sampler[NUMBER_OF_LOOKUP_KINDS][NUMBER_OF_TEX_MODES];
    std::string         m_ptxPath      = ptxPath( "test_Memory", "allTextureModes.lw" );
    std::string         m_outFileName  = "res.txt";
    std::string         m_goldFileName = dataPath() + "/test_Memory/allTextureModes.gold.txt";
    std::vector<float4> m_testCoords;
    bool                m_verbose = true;

    void createBuffer( Buffer& buffer, int lookupKind, RTformat fmt )
    {
        int d     = LOOKUP_KIND_DIM[lookupKind];
        int flags = LOOKUP_KIND_FLAGS[lookupKind];
        ASSERT_TRUE( 1 <= d && d <= 7 );
        if( d == 1 && flags == 0 )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT, fmt, 2u );
        else if( d == 2 && flags == 0 )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT, fmt, 2u, 2u );
        else if( d == 3 && flags == 0 )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT, fmt, 2u, 2u, 2u );
        else if( d == 2 && flags == RT_BUFFER_LAYERED )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LAYERED, fmt, 2u, 1u, 2u );
        else if( d == 3 && flags == RT_BUFFER_LAYERED )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LAYERED, fmt, 2u, 2u, 2u );
        else if( d == 3 && flags == RT_BUFFER_LWBEMAP )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LWBEMAP, fmt, 2u, 2u, 6u );
        else if( d == 3 && flags == ( RT_BUFFER_LWBEMAP | RT_BUFFER_LAYERED ) )
            buffer = m_context->createBuffer( RT_BUFFER_INPUT | RT_BUFFER_LWBEMAP | RT_BUFFER_LAYERED, fmt, 2u, 2u, 12u );

        buffer->setMipLevelCount( 2 );
        fillBuffer( buffer, fmt );
    }

    void SetUp() override
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );

        // set up output buffer
        m_outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_outWidth, m_outHeight );
        m_context["out_float4"]->set( m_outputBuffer );

        m_context["coords"]->setFloat( 0, 0, 0, 0 );

        for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
        {
            for( int f = 0; f < NUMBER_OF_FORMATS; f++ )
            {
                const RTformat inputFmt = FORMATS[f];
                createBuffer( m_buffer[lookupIdx][f], lookupIdx, inputFmt );
                m_formatToBuffer.insert( std::make_pair( std::make_pair( inputFmt, lookupIdx ), &m_buffer[lookupIdx][f] ) );
            }
        }

        for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
        {
            for( int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
            {
                createTexture( m_context, m_sampler[lookupIdx][m], tex_modes[m] );
            }
        }

        volatile float a           = 0.0f;
        volatile float b           = 0.0f;
        const float    THE_BIG     = 3.402823466e+38f - 1.0f;
        const float    THE_NAN     = a / b;
        const float    THE_INF     = 1.0f / b;
        const float    THE_NEG_INF = -1.0f / b;

        // clang-format off
        const float4 COORDS[] = {
            {      0.33f,      0.33f,       0.66f,        1.4f },
            {  65534.67f,      0.33f,       0.66f,        1.4f },
            {  65532.67f,      0.33f,       0.66f,        1.4f },
            {    THE_BIG,      0.33f,       0.66f,        1.4f },
            {    THE_NAN,      0.33f,       0.66f,        1.4f },
            {    THE_INF,      0.33f,       0.66f,        1.4f },
            {THE_NEG_INF,      0.33f,       0.66f,        1.4f }
#if 0
            {      0.33f,      0.33f,       0.66f,        1.4f },
            {      0.33f,  65534.67f,       0.66f,        1.4f },
            {      0.33f,  65532.67f,       0.66f,        1.4f },
            {      0.33f,    THE_BIG,       0.66f,        1.4f },
            {      0.33f,    THE_NAN,       0.66f,        1.4f },
            {      0.33f,    THE_INF,       0.66f,        1.4f },
            {      0.33f,THE_NEG_INF,       0.66f,        1.4f },

            {      0.33f,      0.66f,       0.33f,        1.4f },
            {      0.33f,      0.66f,   65534.67f,        1.4f },
            {      0.33f,      0.66f,   65532.67f,        1.4f },
            {      0.33f,      0.66f,     THE_BIG,        1.4f },
            {      0.33f,      0.66f,     THE_NAN,        1.4f },
            {      0.33f,      0.66f,     THE_INF,        1.4f },
            {      0.33f,      0.66f, THE_NEG_INF,        1.4f },

            {      0.33f,      0.66f,        1.4f,       0.33f },
            {      0.33f,      0.66f,        1.4f,   65534.67f },
            {      0.33f,      0.66f,        1.4f,   65532.67f },
            {      0.33f,      0.66f,        1.4f,     THE_BIG },
            {      0.33f,      0.66f,        1.4f,     THE_NAN },
            {      0.33f,      0.66f,        1.4f,     THE_INF },
            {      0.33f,      0.66f,        1.4f, THE_NEG_INF }
#endif
        };
        // clang-format on
        m_testCoords.insert( m_testCoords.begin(), std::begin( COORDS ), std::end( COORDS ) );
    }

    void TearDown() override { m_context->destroy(); }
};


const int NUMBER_OF_UNSUPPORTED_TEX_MODES                        = 4;
TexModes  unsupported_tex_modes[NUMBER_OF_UNSUPPORTED_TEX_MODES] = {
    {0, 3, 1},
    {1, 0, 1},
    {1, 2, 0},
    {1, 3, 0},
};

// -----------------------------------------------------------------------------
TEST( TestAllTextureModes, UnsupportedTextureModesByLWDA )
{
    Context context = Context::create();
    context->setRayTypeCount( 1 );
    context->setEntryPointCount( 1 );
    Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 2u );
    for( int m = 0; m < NUMBER_OF_UNSUPPORTED_TEX_MODES; m++ )
    {
        TextureSampler sampler;
        createTexture( context, sampler, unsupported_tex_modes[m] );
        sampler->setBuffer( buffer );
        ASSERT_THAT( rtTextureSamplerValidate( sampler->get() ), Eq( RT_ERROR_ILWALID_VALUE ) );
    }
    context->destroy();
}

// -----------------------------------------------------------------------------
TEST_F( TestAllTextureModes, DISABLED_BoundforRead )
{
    // setup ray generation program
    Program ray_gen_program = m_context->createProgramFromPTXFile( m_ptxPath, "bound_texture" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
    {
        for( int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
        {
            m_sampler[lookupIdx][m]->setBuffer( m_buffer[0][0] );  // workaround, TODO figure out if set sampler without buffer is okay
            std::stringstream st;
            st << "tex_" << LOOKUP_KIND_STRINGS[lookupIdx] << "_mode" << m;
            m_context[st.str()]->setTextureSampler( m_sampler[lookupIdx][m] );
        }
    }

    std::ofstream outFile( m_outFileName );
    std::ifstream goldFile( m_goldFileName );
    EXPECT_EQ( outFile.is_open(), true );
    EXPECT_EQ( goldFile.is_open(), true );

    for( unsigned int frm = 0; frm < NUMBER_OF_FORMATS_AND_READ_MODES; frm++ )
    {
        const RTformat          inputFmt = formatsAndReadmodes[frm].inputFmt;
        const RTtexturereadmode readMode = formatsAndReadmodes[frm].readMode;
        for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
        {
            for( int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
            {
                TextureSampler& sampler = m_sampler[lookupIdx][m];
                sampler->setBuffer( *m_formatToBuffer[std::make_pair( inputFmt, lookupIdx )] );
                sampler->setReadMode( readMode );
            }
        }

        for( float4 coords : m_testCoords )
        {
            m_context["coords"]->setFloat( coords );

            m_context->launch( 0, m_outWidth, m_outHeight );

            dumpOutBuffer( outFile, m_verbose, m_outputBuffer, inputFmt, readMode, coords );
        }
    }
}


struct WithFormatAndReadMode : TestAllTextureModes, WithParamInterface<FormatAndReadMode*>
{
    void SetUp() override;
    void readGoldResults();
    void setupOutputFile();
    void checkGoldResults( float4 coords );

    FormatAndReadMode m_param = *GetParam();
    unsigned int      m_resultIndex;
    std::ofstream     m_outputFile;

    struct ResultLine
    {
        float result[4];
    };
    static std::vector<ResultLine> m_goldResults;
};

std::vector<WithFormatAndReadMode::ResultLine> WithFormatAndReadMode::m_goldResults;

void WithFormatAndReadMode::SetUp()
{
    TestAllTextureModes::SetUp();
    const auto pos = algorithm::find( formatsAndReadmodes, m_param );
    ASSERT_NE( pos, std::end( formatsAndReadmodes ) );
    const size_t paramIndex = std::distance( std::begin( formatsAndReadmodes ), pos );
    m_resultIndex           = paramIndex * NUMBER_OF_LOOKUP_KINDS * NUMBER_OF_TEX_MODES * m_testCoords.size();

    readGoldResults();

    // setup ray generation program
    Program ray_gen_program = m_context->createProgramFromPTXFile( m_ptxPath, "bindless_texture" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    Buffer tex_ids = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, NUMBER_OF_LOOKUP_KINDS * NUMBER_OF_TEX_MODES );
    m_context["tex_ids"]->set( tex_ids );
    int* ids = (int*)tex_ids->map();
    for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
    {
        for( int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
        {
            TextureSampler& sampler                  = m_sampler[lookupIdx][m];
            ids[lookupIdx * NUMBER_OF_TEX_MODES + m] = sampler->getId();
        }
    }
    tex_ids->unmap();

    setupOutputFile();
}

void WithFormatAndReadMode::readGoldResults()
{
    if( m_goldResults.empty() )
    {
        std::ifstream goldFile( m_goldFileName );
        while( goldFile )
        {
            ResultLine resultLine;
            goldFile >> resultLine.result[0] >> resultLine.result[1] >> resultLine.result[2] >> resultLine.result[3];
            m_goldResults.push_back( resultLine );
            goldFile.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
        }
    }
}

void WithFormatAndReadMode::setupOutputFile()
{
    m_outputFile.open( m_outFileName, std::ios_base::out );
    EXPECT_TRUE( m_outputFile.is_open() );
}

void WithFormatAndReadMode::checkGoldResults( const float4 coords )
{
    const float* data = static_cast<float*>( m_outputBuffer->map() );

    for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
    {
        for( unsigned int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
        {
            const ResultLine& results = m_goldResults[m_resultIndex];
            for( unsigned int c = 0; c < 4; c++ )
            {
                float r;
                if( m_param.readMode == RT_TEXTURE_READ_ELEMENT_TYPE
                    || m_param.readMode == RT_TEXTURE_READ_ELEMENT_TYPE_SRGB || lookupIdx == TestTexture_size )
                    r = *reinterpret_cast<const int*>( data );
                else
                    r = *data;
                data++;

                const float eps = std::max( fabsf( results.result[c] ) * 0.01f, 0.01f );
                EXPECT_NEAR( results.result[c], r, eps )
                    << "CONTEXT: " << LOOKUP_KIND_STRINGS[lookupIdx] << ' ' << TexModesToStr( tex_modes[m] )
                    << " coords=(" << coords.x << ", " << coords.y << ", " << coords.z << ", " << coords.w << ')';
            }
            ++m_resultIndex;
        }
    }

    m_outputBuffer->unmap();
}

TEST_P( WithFormatAndReadMode, OutputForTextureCoords )
{
    for( int lookupIdx = 0; lookupIdx < NUMBER_OF_LOOKUP_KINDS; lookupIdx++ )
    {
        for( int m = 0; m < NUMBER_OF_TEX_MODES; m++ )
        {
            TextureSampler& sampler = m_sampler[lookupIdx][m];
            sampler->setBuffer( *m_formatToBuffer[std::make_pair( m_param.inputFmt, lookupIdx )] );
            sampler->setReadMode( m_param.readMode );
        }
    }

    for( float4 coords : m_testCoords )
    {
        m_context["coords"]->setFloat( coords );
        m_context->launch( 0, m_outWidth, m_outHeight );
        dumpOutBuffer( m_outputFile, m_verbose, m_outputBuffer, m_param.inputFmt, m_param.readMode, coords );
        checkGoldResults( coords );
    }
}

INSTANTIATE_TEST_SUITE_P( BindlessForRead,
                          WithFormatAndReadMode,
                          Range( std::begin( formatsAndReadmodes ), std::end( formatsAndReadmodes ) ) );
