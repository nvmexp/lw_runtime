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

#pragma once


/*
 * Simple runtime available for use in clang-based runtime code.
 * Keep this using simple types if possible.
 */

namespace lwca {
float minf( float, float );
float maxf( float, float );
float floorf( float );
int   float2int_rz( float );

enum lwdaTextureAddressMode
{
    lwdaAddressModeWrap   = 0,
    lwdaAddressModeClamp  = 1,
    lwdaAddressModeMirror = 2,
    lwdaAddressModeBorder = 3
};
enum lwdaTextureFilterMode
{
    lwdaFilterModePoint  = 0,
    lwdaFilterModeLinear = 1
};
}  // end namespace lwca

namespace cort {
typedef unsigned short     uint16;
typedef unsigned int       uint;
typedef unsigned long long uint64;
struct float4;

float sqrt( float );
float nan();

struct uint2
{
    inline uint2() {}
    uint2( uint x, uint y )
        : x( x )
        , y( y )
    {
    }
    uint2( const uint2& copy )
        : x( copy.x )
        , y( copy.y )
    {
    }
    uint2& operator=( const uint2& copy )
    {
        x = copy.x;
        y = copy.y;
        return *this;
    }
    uint2 operator+( const uint2& b ) const { return uint2( x + b.x, y + b.y ); }
    bool operator==( const uint2& b ) const { return x == b.x && y == b.y; }
    ~uint2(){};
    uint x, y;
};

struct uint3
{
    inline uint3() {}
    uint3( uint x, uint y, uint z )
        : x( x )
        , y( y )
        , z( z )
    {
    }
    uint3( const uint3& copy )
        : x( copy.x )
        , y( copy.y )
        , z( copy.z )
    {
    }
    uint3& operator=( const uint3& copy )
    {
        x = copy.x;
        y = copy.y;
        z = copy.z;
        return *this;
    }
    uint3 operator+( const uint3& b ) const { return uint3( x + b.x, y + b.y, z + b.z ); }
    bool operator==( const uint3& b ) const { return x == b.x && y == b.y && z == b.z; }
    ~uint3(){};
    uint x, y, z;
};

struct uint4
{
    inline uint4() {}
    uint4( int x, int y, int z, int w )
        : x( x )
        , y( y )
        , z( z )
        , w( w )
    {
    }
    uint4( uint3 xyz, int w )
        : x( xyz.x )
        , y( xyz.y )
        , z( xyz.z )
        , w( w )
    {
    }
    uint4( const uint4& copy )
        : x( copy.x )
        , y( copy.y )
        , z( copy.z )
        , w( copy.w )
    {
    }
    uint4& operator=( const uint4& copy )
    {
        x = copy.x;
        y = copy.y;
        z = copy.z;
        w = copy.w;
        return *this;
    }
    uint4 operator+( const uint4& b ) const { return uint4( x + b.x, y + b.y, z + b.z, w + b.w ); }
    ~uint4(){};
    int x, y, z, w;
};

struct size3
{
    inline size3() {}
    size3( uint64 x, uint64 y, uint64 z )
        : x( x )
        , y( y )
        , z( z )
    {
    }
    size3( const size3& copy )
        : x( copy.x )
        , y( copy.y )
        , z( copy.z )
    {
    }
    size3& operator=( const size3& copy )
    {
        x = copy.x;
        y = copy.y;
        z = copy.z;
        return *this;
    }
    size3 operator+( const size3& b ) const { return size3( x + b.x, y + b.y, z + b.z ); }
    ~size3(){};
    uint64 x, y, z;
};

struct float2
{
    float x, y;
};

struct float3
{
    inline float3() {}
    float3( float x, float y, float z )
        : x( x )
        , y( y )
        , z( z )
    {
    }
    float3( const float3& copy )
        : x( copy.x )
        , y( copy.y )
        , z( copy.z )
    {
    }
    float3( float4 xyxw );
    float3  operator-() { return float3( -x, -y, -z ); }
    float3& operator=( const float3& copy )
    {
        x = copy.x;
        y = copy.y;
        z = copy.z;
        return *this;
    }
    float3 operator+( const float3& b ) const { return float3( x + b.x, y + b.y, z + b.z ); }
    float3 operator-( const float3& b ) const { return float3( x - b.x, y - b.y, z - b.z ); }
    float3 operator*( const float3& b ) const { return float3( x * b.x, y * b.y, z * b.z ); }
    float3& operator*=( float b )
    {
        x *= b;
        y *= b;
        z *= b;
        return *this;
    }
    float length() { return sqrt( x * x + y * y + z * z ); }
    ~float3(){};
    float         x, y, z;
    static float3 min( const float3& a, const float3& b )
    {
        return float3( lwca::minf( a.x, b.x ), lwca::minf( a.y, b.y ), lwca::minf( a.z, b.z ) );
    }
    static float3 max( const float3& a, const float3& b )
    {
        return float3( lwca::maxf( a.x, b.x ), lwca::maxf( a.y, b.y ), lwca::maxf( a.z, b.z ) );
    }
};

struct float4
{
    inline float4() {}
    float4( float x, float y, float z, float w )
        : x( x )
        , y( y )
        , z( z )
        , w( w )
    {
    }
    float4( float3 xyz, float w )
        : x( xyz.x )
        , y( xyz.y )
        , z( xyz.z )
        , w( w )
    {
    }
    float4( const float4& copy )
        : x( copy.x )
        , y( copy.y )
        , z( copy.z )
        , w( copy.w )
    {
    }
    float4& operator=( const float4& copy )
    {
        x = copy.x;
        y = copy.y;
        z = copy.z;
        w = copy.w;
        return *this;
    }
    float4 operator+( const float4& b ) const { return float4( x + b.x, y + b.y, z + b.z, w + b.w ); }
    float4 operator-( const float4& b ) const { return float4( x - b.x, y - b.y, z - b.z, w - b.w ); }
    float4 operator*( const float s ) const { return float4( x * s, y * s, z * s, w * s ); }
    ~float4(){};
    float x, y, z, w;
};

inline float3::float3( float4 xyzw )
    : x( xyzw.x )
    , y( xyzw.y )
    , z( xyzw.z )
{
}

inline float3 cross( const float3& u, const float3& v )
{
    return float3( u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x );
}

inline float dot( const float3& u, const float3& v )
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

inline float3 operator*( float a, const float3& b )
{
    return float3( a * b.x, a * b.y, a * b.z );
}

struct Matrix4x4
{
    float4 operator*( float4 b ) const
    {
        float x = matrix[0][0] * b.x + matrix[0][1] * b.y + matrix[0][2] * b.z + matrix[0][3] * b.w;
        float y = matrix[1][0] * b.x + matrix[1][1] * b.y + matrix[1][2] * b.z + matrix[1][3] * b.w;
        float z = matrix[2][0] * b.x + matrix[2][1] * b.y + matrix[2][2] * b.z + matrix[2][3] * b.w;
        float w = matrix[3][0] * b.x + matrix[3][1] * b.y + matrix[3][2] * b.z + matrix[3][3] * b.w;
        return float4( x, y, z, w );
    }
    float matrix[4][4];
};

struct OptixRay
{
    float3       origin;
    float3       direction;
    unsigned int ray_type;
    float        tmin;
    float        tmax;
};


struct GeneralBB;

struct Aabb
{
    float3 min, max;

    Aabb() {}

    Aabb( const float3& min, const float3& max )
        : min( min )
        , max( max )
    {
    }

    void set( const float3& min, const float3& max )
    {
        this->min = min;
        this->max = max;
    }
    void set( const GeneralBB& genbb )
    {
        ilwalidate();
        include( genbb );
    }
    void ilwalidate()
    {
        float nanf = cort::nan();

        min = float3( nanf, nanf, nanf );
        max = float3( nanf, nanf, nanf );
    }
    void include( const Aabb& other )
    {
        min = float3::min( min, other.min );
        max = float3::max( max, other.max );
    }
    void include( const float3& p )
    {
        min = float3::min( min, p );
        max = float3::max( max, p );
    }
    void include( const GeneralBB& genbb );

    bool valid() const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }
};

struct GeneralBB
{
    float3 anchor, v0, v1, v2;
    bool   valid;
    void set( const Aabb& aabb )
    {
        valid       = aabb.valid();
        anchor      = aabb.min;
        float3 diag = aabb.max - aabb.min;
        v0          = float3( diag.x, 0, 0 );
        v1          = float3( 0, diag.y, 0 );
        v2          = float3( 0, 0, diag.z );
    }

    void ilwalidate() { valid = false; }
};

inline void Aabb::include( const GeneralBB& genbb )
{
    // Only include genbb if it is valid.  Don't mess with *this if it isn't.
    if( !genbb.valid )
        return;

    include( genbb.anchor + 0.f * genbb.v0 + 0.f * genbb.v1 + 0.f * genbb.v2 );
    include( genbb.anchor + 0.f * genbb.v0 + 0.f * genbb.v1 + 1.f * genbb.v2 );
    include( genbb.anchor + 0.f * genbb.v0 + 1.f * genbb.v1 + 0.f * genbb.v2 );
    include( genbb.anchor + 0.f * genbb.v0 + 1.f * genbb.v1 + 1.f * genbb.v2 );
    include( genbb.anchor + 1.f * genbb.v0 + 0.f * genbb.v1 + 0.f * genbb.v2 );
    include( genbb.anchor + 1.f * genbb.v0 + 0.f * genbb.v1 + 1.f * genbb.v2 );
    include( genbb.anchor + 1.f * genbb.v0 + 1.f * genbb.v1 + 0.f * genbb.v2 );
    include( genbb.anchor + 1.f * genbb.v0 + 1.f * genbb.v1 + 1.f * genbb.v2 );
}


// Parameters used by the AABB iterator program.
struct AabbRequest
{
    bool         isGroup;
    unsigned int recordOffset;  // Group or GeometryInstance
    unsigned int buildMotionSteps;
    unsigned int geometryMotionSteps;
    bool         computeUnion;

    Aabb*  aabbOutputPointer;
    uint2* motionAabbRequests;
    AabbRequest() = default;
    AabbRequest( bool         isGroup,
                 unsigned int recordOffset,
                 unsigned int buildMotionSteps,
                 unsigned int geometryMotionSteps,
                 bool         computeUnion,
                 Aabb*        aabbOutputPointer,
                 uint2*       motionAabbRequests )
        : isGroup( isGroup )
        , recordOffset( recordOffset )
        , buildMotionSteps( buildMotionSteps )
        , geometryMotionSteps( geometryMotionSteps )
        , computeUnion( computeUnion )
        , aabbOutputPointer( aabbOutputPointer )
        , motionAabbRequests( motionAabbRequests )
    {
    }
};

struct SrtTransform
{
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
};

/*
   * Upper limits
   */

static const unsigned int   MAXATTRIBUTE_TOTALSIZE = 256;
static const unsigned int   MAXPAYLOADSIZE         = 512;
static const unsigned int   MAXTRANSFORMDEPTH      = 16;
static const unsigned short ILWALIDOFFSET          = 0xffff;

/*
   * Types for IDs and offsets
   */
typedef unsigned int       ObjectRecordOffset;
typedef ObjectRecordOffset AbstractGroupHandle;
typedef ObjectRecordOffset AccelerationHandle;
typedef ObjectRecordOffset GeometryHandle;
typedef ObjectRecordOffset GeometryTrianglesHandle;
typedef ObjectRecordOffset MotionGeometryTrianglesHandle;
typedef ObjectRecordOffset GeometryInstanceHandle;
typedef ObjectRecordOffset GlobalScopeHandle;
typedef ObjectRecordOffset GraphNodeHandle;
typedef ObjectRecordOffset LexicalScopeHandle;
typedef ObjectRecordOffset MaterialHandle;
typedef ObjectRecordOffset MotionAccelerationHandle;
typedef ObjectRecordOffset ProgramHandle;
typedef ObjectRecordOffset SelectorHandle;
typedef ObjectRecordOffset TransformHandle;
typedef unsigned int       DynamicVariableTableOffset;

typedef int BufferId;
typedef int TextureSamplerId;
typedef int ProgramId;
typedef int TraversableId;

/*
   * Table types, some of which are device-specific
   */
struct ProgramHeader
{
    struct DeviceIndependent
    {
        ObjectRecordOffset programOffset;
    } di;
    struct DeviceDependent
    {
        unsigned int canonicalProgramID;
    } dd;
};
struct Buffer
{
    struct DeviceIndependent
    {
        size3 size;      // in elements
        uint3 pageSize;  // in elements
    } di;
    struct DeviceDependent
    {
        char* data;
        int   texUnit;
    } dd;

    enum
    {
        UseDataAsPointer = -3  // indicates to use the pointer, otherwise use the data pointer as a texture offset.
    };
};

struct TraversableHeader
{
    uint64 traversable;
};

// Strive to keep this struct smaller than 16 bytes, so that we can pack 4 SBT records into a cache line.
// (Assuming that the SBTRecordHeader is 16 bytes large).
union SBTRecordData
{
    // Name all the types to make linking with LLVM easier
    struct ProgramDataT
    {
        ProgramHandle programOffset;
    } ProgramData;

    struct GeometryInstanceDataT
    {
        GeometryInstanceHandle giOffset;
        MaterialHandle         materialOffset;
        int                    skip;
    } GeometryInstanceData;
};

// log2(componentSze) = (x&3)
// log2(vectorSize) = (x>>4)
// log2(eltSize) = (x&3)+(x>>4)
enum InternalTexFormat
{
    TEX_FORMAT_UNSIGNED_BYTE1  = 0,
    TEX_FORMAT_UNSIGNED_SHORT1 = 1,
    TEX_FORMAT_UNSIGNED_INT1   = 2,
    TEX_FORMAT_BYTE1           = 4,
    TEX_FORMAT_SHORT1          = 5,
    TEX_FORMAT_INT1            = 6,
    TEX_FORMAT_FLOAT1          = 10,

    TEX_FORMAT_UNSIGNED_BYTE2  = 0 + 16,
    TEX_FORMAT_UNSIGNED_SHORT2 = 1 + 16,
    TEX_FORMAT_UNSIGNED_INT2   = 2 + 16,
    TEX_FORMAT_BYTE2           = 4 + 16,
    TEX_FORMAT_SHORT2          = 5 + 16,
    TEX_FORMAT_INT2            = 6 + 16,
    TEX_FORMAT_FLOAT2          = 10 + 16,

    TEX_FORMAT_UNSIGNED_BYTE4  = 0 + 32,
    TEX_FORMAT_UNSIGNED_SHORT4 = 1 + 32,
    TEX_FORMAT_UNSIGNED_INT4   = 2 + 32,
    TEX_FORMAT_BYTE4           = 4 + 32,
    TEX_FORMAT_SHORT4          = 5 + 32,
    TEX_FORMAT_INT4            = 6 + 32,
    TEX_FORMAT_FLOAT4          = 10 + 32
};

enum InternalTexWrapmode
{
    TEX_WRAP_REPEAT,
    TEX_WRAP_CLAMP_TO_EDGE,
    TEX_WRAP_MIRROR,
    TEX_WRAP_CLAMP_TO_BORDER
};

#define MAX_ANISOTROPY 16
#define ILW_ANISOTROPY ( 1.0f / MAX_ANISOTROPY )

struct TextureSampler
{
    // There is only one for now in order to make this a fixed sized object for making a
    // table out of it on the device.  If we ever want to support multiple buffers
    // attached to the same sampler (e.g. mip maps or texture arrays) we should have
    // buffers point to another buffer with all the relevant information about the arrays
    // of buffers.

    // Device independent members
    // TODO: when these bitfield members are placed in a nested struct, clang crashes
    // during compilation of the device runtime.  So keep these device independent
    // members at the top of this struct while host code uses TextureSamplerHost that
    // contains a nested structure for d.i. members.  The memory layout of these two
    // structs must be maintained.

    // When modifying this struct, be sure to update the following code for consistency:
    // - TableManager::writeTextureHeader
    // - LayoutPrinter::printTextureHeader
    // - struct.cort::TextureSampler (in C14nRuntime.ll)
    // - Megakernel_getTextureHeaderFromConst (in RTXRuntime.cpp and MegakernelRuntime.cpp)

    unsigned width;
    unsigned height;
    unsigned depth;
    unsigned mipLevels;
    unsigned mipTailFirstLevel;
    float    ilwAnisotropy;  // only needed for demand textures

    unsigned format : 6;  // (float, uint, int, byte, ubyte, ushort, short)x(1, 2, 4)
    unsigned wrapMode0 : 2;
    unsigned wrapMode1 : 2;
    unsigned wrapMode2 : 2;
    unsigned normCoord : 1;   // normalized coords
    unsigned filterMode : 1;  // nearest / linear
    unsigned normRet : 1;     // normalized return
    unsigned isDemandLoad : 1;

    // Demand texture fields (valid only when isDemandLoad=1)
    unsigned tileWidth : 12;
    unsigned tileHeight : 12;
    unsigned tileGutterWidth : 4;
    unsigned isInitialized : 1;
    unsigned isSquarePowerOfTwo : 1;
    unsigned mipmapFilterMode : 1;
    unsigned padding : 17;
    unsigned padding2;

    struct DeviceDependent
    {
        uint64        texref;  // ID (sm_20), texobject (sm_30+), -1 (Software) or -2 (invalid)
        char*         swptr;   // For software interpolation.  Demand texture has startPage/numPages in low/high words.
        unsigned char minMipLevel;
        unsigned char padding[7];
    } dd;

    enum
    {
        UseSoftwarePointer = -1,
        IlwalidSampler     = -2
    };
};
struct TextureSamplerHost
{
    struct DeviceIndependent
    {
        unsigned width;
        unsigned height;
        unsigned depth;
        unsigned mipLevels;
        unsigned mipTailFirstLevel;
        float    ilwAnisotropy;  // only needed for demand textures

        unsigned format : 6;  // (float, uint, int, byte, ubyte, ushort, short)x(1, 2, 4)
        unsigned wrapMode0 : 2;
        unsigned wrapMode1 : 2;
        unsigned wrapMode2 : 2;
        unsigned normCoord : 1;   // normalized coords
        unsigned filterMode : 1;  // nearest / linear
        unsigned normRet : 1;     // normalized return
        unsigned isDemandLoad : 1;

        // Demand texture fields (valid only when isDemandLoad=1)
        unsigned tileWidth : 12;
        unsigned tileHeight : 12;
        unsigned tileGutterWidth : 4;
        unsigned isInitialized : 1;
        unsigned isSquarePowerOfTwo : 1;
        unsigned mipmapFilterMode : 1;
        unsigned padding : 17;
        unsigned padding2;
    } di;
    using DeviceDependent = TextureSampler::DeviceDependent;
    DeviceDependent dd;
};

static_assert( sizeof( TextureSamplerHost ) == sizeof( TextureSampler ),
               "Host and device TextureSampler representations differ in size" );

// Packing and unpacking colwentions for TextureSampler::DeviceDependent::swptr.
// (Using a union didn't work, apparently due to getting the type to match the one in C14nRuntime.ll)
inline void setDemandTextureDeviceHeader( TextureSampler::DeviceDependent* dd,
                                          unsigned int                     firstVirtualPage,
                                          unsigned int                     numPages,
                                          unsigned int                     minMipLevel )
{
    dd->swptr = reinterpret_cast<char*>( ( static_cast<unsigned long long>( numPages ) << 32 ) | firstVirtualPage );
    dd->minMipLevel = minMipLevel;
}

inline unsigned int getDemandTextureFirstPage( const TextureSampler::DeviceDependent& dd )
{
    // The low word of TextureSampler::DeviceDependent.swptr is the first page id.
    return static_cast<unsigned int>( reinterpret_cast<unsigned long long>( dd.swptr ) & 0xFFFFFFFFU );
}

inline unsigned int getDemandTextureNumPages( const TextureSampler::DeviceDependent& dd )
{
    // The upper word of TextureSampler::DeviceDependent.swptr is the number of pages.
    return static_cast<unsigned int>( reinterpret_cast<unsigned long long>( dd.swptr ) >> 32 );
}

//
// Base types
//

struct LexicalScopeRecord
{
    // The offset inside the cort::Global::dynamicVariableTable.
    DynamicVariableTableOffset dynamicVariableTable = static_cast<DynamicVariableTableOffset>( -1 );
};

struct ProgramRecord : public LexicalScopeRecord
{
    ProgramId programID;
};

struct GlobalScopeRecord : LexicalScopeRecord
{
    // Note that even though this is size of one, we can actually allocate as many of
    // these as necessary to get the right number of them.
    //
    // The size of the array is based
    // on the maximum of the number of entry points and ray type.  The number of raygen
    // and exception programs will match the number of entry points, and the number of
    // miss programs will match the number of ray types.
    //
    // programs[3].raygen will have the fourth entry point program.
    //
    // The reason the miss programs are mixed with the raygen and exception programs is
    // that this trick of putting variable length items into a struct only works for the
    // last member of the struct.  Mixing the programs together could waste a bit of
    // space, but it makes the variable number of programs work here.
    struct
    {
        ProgramHandle raygen;
        ProgramHandle exception;
        ProgramHandle miss;
    } programs[1];
};

struct GraphNodeRecord : public LexicalScopeRecord
{
    ProgramHandle traverse;
    ProgramHandle bounds;

    // The index of the traversable (acceleration structure, selector
    // or transform) used by rtcore.
    TraversableId traversableId;
};

enum InternalMotionBorderMode
{
    MOTIONBORDERMODE_CLAMP,
    MOTIONBORDERMODE_VANISH
};

enum InternalMotionKeyType
{
    MOTIONKEYTYPE_NONE = 0,
    MOTIONKEYTYPE_MATRIX_FLOAT12,
    MOTIONKEYTYPE_SRT_FLOAT16
};

enum InternalMotionDataOffset
{
    MDOFFSET_KEY_TYPE          = 0,
    MDOFFSET_BEGIN_BORDER_MODE = 1,
    MDOFFSET_END_BORDER_MODE   = 2,
    MDOFFSET_TIME_BEGIN        = 3,
    MDOFFSET_TIME_END          = 4,
    MDOFFSET_NUM_KEYS          = 5,
    MDOFFSET_KEYS              = 6
};

struct TransformRecord : public GraphNodeRecord
{  // Final (no classes derive from this)
    GraphNodeHandle child;
    Matrix4x4       matrix;
    Matrix4x4       ilwerse_matrix;

    // Motion blur state
    BufferId motionData;
};

struct AbstractGroupRecord : public GraphNodeRecord
{  // Final
    AccelerationHandle accel;
    BufferId           children;
};

struct AccelerationRecord : public LexicalScopeRecord
{
};

struct MotionAccelerationRecord : public AccelerationRecord
{  // Final
    float        timeBegin;
    float        timeEnd;
    unsigned int motionSteps;
    unsigned int motionStride;
};

struct SelectorRecord : public GraphNodeRecord
{
    BufferId children;
};

struct GeometryRecord;
struct MaterialRecord;
struct GeometryInstanceRecord : LexicalScopeRecord
{
    GeometryHandle geometry;
    unsigned int   numMaterials;
    // Note that even though this is size of one, we can actually allocate as many of
    // these as necessary to get the right number of them.
    MaterialHandle materials[1];
};

struct GeometryRecord : public LexicalScopeRecord
{
    unsigned int  indexOffset;
    ProgramHandle intersectOrAttribute;
    ProgramHandle aabb;
    unsigned int  attributeKind;
};

struct GeometryTrianglesRecord : public GeometryRecord
{
    long long          vertexBufferOffset;
    unsigned long long vertexBufferStride;
    int                vertexBufferID;
    int                indexBufferID;
    long long          indexBufferOffset;
    unsigned long long indexBufferStride;
};

struct MotionGeometryTrianglesRecord : public GeometryTrianglesRecord
{
    unsigned long long vertexBufferMotionStride;
    int                motionNumIntervals;
    float              timeBegin;
    float              timeEnd;
    int                motionBorderModeBegin;
    int                motionBorderModeEnd;
};

struct MaterialRecord : public LexicalScopeRecord
{
    // Note that even though this is size of one, we can actually allocate as many of
    // these as necessary to get the right number of them.
    struct
    {
        ProgramHandle closestHit;
        ProgramHandle anyHit;
    } programs[1];
};

// While the following utilities regarding the dynamic variable table are just implementation detail,
// they need to be shared between LexicalScope, CommonRuntime and RtxRuntime nevertheless.
// Note that the variable table is implemented as a binary tree, hence we talk about leaf entries.

// use highest-order bit as a marker for leaf entries in the variable table
static const unsigned short VARIABLE_TABLE_LEAF_MASK = 0x8000;

// Is the given variable token id representing a leaf in the dynamic variable table?
inline bool isVariableTableLeaf( const unsigned short* token )
{
    return ( *token & VARIABLE_TABLE_LEAF_MASK ) != 0;
}

// Mark the given variable token id as a leaf in the dynamic variable table.
inline void markAsVariableTableLeaf( unsigned short* token )
{
    *token |= VARIABLE_TABLE_LEAF_MASK;
}

// Return the real variable token id of the given value by removing a special leaf marker from it.
inline unsigned short getUnmarkedVariableTokenId( const unsigned short* token )
{
    return ( *token ) & ~VARIABLE_TABLE_LEAF_MASK;
}

static const int PROFILE_FULL_KERNEL_TIMER   = -1;
static const int PROFILE_NUM_RESERVED_TIMERS = 1;

static const int PROFILE_TRACE_COUNTER         = -1;
static const int PROFILE_NUM_RESERVED_COUNTERS = 1;

static const int PROFILE_NUM_RESERVED_EVENTS = 0;
}  // end namespace cort
