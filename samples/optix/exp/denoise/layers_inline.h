//
// Copyright 2020 LWPU Corporation. All rights reserved.
//
// Auto-encoder layer helper classes

#pragma once

namespace optix_exp {

#define pq_m1 0.1593017578125f
#define pq_m2 78.84375f
#define pq_c1 0.8359375f
#define pq_c2 18.8515625f
#define pq_c3 18.6875f
#define pq_C 10000.f

static __inline__ __device__ float pq_encode( float x )
{
    float L  = x / pq_C;
    float Lm = __powf( L, pq_m1 );
    float N  = ( pq_c1 + pq_c2 * Lm ) / ( 1.f + pq_c3 * Lm );
    return __powf( N, pq_m2 );
}

static __inline__ __device__ float pq_decode( float x )
{
    float Np = __powf( x, 1.f / pq_m2 );
    float L  = Np - pq_c1;
    if( L < 0.f )
        L = 0.f;
    L     = L / ( pq_c2 - pq_c3 * Np );
    L     = __powf( L, 1.f / pq_m1 );
    return L * pq_C;
}

static __inline__ __device__ float __clamp( float x, float xmin, float xmax = 1e20 )
{
    if( !isfinite( x ) )
        return 0.f;
    if( x > xmax )
        return xmax;
    return x >= xmin ? x : 0.f;
}

static __inline__ __device__ float __clamp0( float x, float xmax = 1e20 )
{
    if( !isfinite( x ) )
        return 0.f;
    if( x > xmax )
        return xmax;
    return x > 0.f ? x : 0.f;
}

static __inline__ __device__ float __clamp01( float x )
{
    return __saturatef( x );
}

static inline __host__ int M8( int x )
{
    return ( ( x + 7 ) / 8 ) * 8;
}

static inline __host__ int M8tc( int x, bool tc )
{
    return tc ? M8( x ) : x;
}

static __inline__ __device__ int addrNHWC( int channels, int height, int width, int n, int c, int y, int x )
{
    return n * height * width * channels + y * width * channels + x * channels + c;
}

static inline __device__ __host__ unsigned int getNumChannels( const OptixImage2D& image )
{
    switch( image.format )
    {
        case OPTIX_PIXEL_FORMAT_HALF2:
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            return 2;
        case OPTIX_PIXEL_FORMAT_HALF3:
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            return 3;
        case OPTIX_PIXEL_FORMAT_HALF4:
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            return 4;
        case OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER:
            return image.pixelStrideInBytes / sizeof( short );
    }
    return 0;
}

struct floatRdAccess
{
    inline floatRdAccess() : psb(0), hf(false) { image = {}; }
    inline floatRdAccess( const OptixImage2D& im )
        : image( im )
        , psb( im.pixelStrideInBytes )
        , hf( image.format == OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER || image.format == OPTIX_PIXEL_FORMAT_HALF2 || image.format == OPTIX_PIXEL_FORMAT_HALF3 || image.format == OPTIX_PIXEL_FORMAT_HALF4 )
    {
        if( im.pixelStrideInBytes == 0 )
        {
            unsigned int dsize = hf ? sizeof( __half ) : sizeof( float );
            psb                = getNumChannels( im ) * dsize;
        }
    }
    inline __device__ float operator()( int x, int y, int c ) const
    {
        if( hf )
            return float( *(const __half*)( image.data + (size_t)y * image.rowStrideInBytes + (size_t)x * psb + c * sizeof( __half ) ) );
        else
            return float( *(const float*)( image.data + (size_t)y * image.rowStrideInBytes + (size_t)x * psb + c * sizeof( float ) ) );
    }
    OptixImage2D image;
    unsigned int psb;
    bool         hf;
};

struct floatWrAccess
{
    inline floatWrAccess( const OptixImage2D& im )
        : image( im )
        , psb( im.pixelStrideInBytes )
        , hf( image.format == OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER || image.format == OPTIX_PIXEL_FORMAT_HALF2 || image.format == OPTIX_PIXEL_FORMAT_HALF3 || image.format == OPTIX_PIXEL_FORMAT_HALF4 )
    {
        if( im.pixelStrideInBytes == 0 )
        {
            unsigned int dsize = hf ? sizeof( __half ) : sizeof( float );
            psb                = getNumChannels( im ) * dsize;
        }
    }
    inline __device__ void write( int x, int y, int c, float value )
    {
        if( hf )
            *(__half*)( image.data + (size_t)y * image.rowStrideInBytes + (size_t)x * psb + c * sizeof( __half ) ) = value;
        else
            *(float*)( image.data + (size_t)y * image.rowStrideInBytes + (size_t)x * psb + c * sizeof( float ) ) = value;
    }
    OptixImage2D image;
    unsigned int psb;
    bool         hf;
};

#if( __LWDA_ARCH__ >= 530 || !defined( __LWDA_ARCH__ ) )
static inline __device__ int hmax2( int a, int b ) {
    int c;
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela, selb;\n" \
        "\n" \
        "\t set.ge.f16x2.f16x2 sela, %1, %2;\n" \
        "\t set.gt.f16x2.f16x2 selb, %2, %1;\n" \
        "\n" \
        "\t mul.f16x2 %0, sela, %1;\n" \
        "\t fma.rn.f16x2 %0, selb, %2, %0;\n" \
        "}\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

static inline __device__ int4 hmax8( int4 a, int4 b ) {
    int4 c;
    c.x = hmax2( a.x, b.x );
    c.y = hmax2( a.y, b.y );
    c.z = hmax2( a.z, b.z );
    c.w = hmax2( a.w, b.w );
    return c;
}

static inline __device__ int4 hadd8(int4 a, int4 b) {
    int4 c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a.x), "r"(b.x));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a.y), "r"(b.y));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a.z), "r"(b.z));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a.w), "r"(b.w));
    return c;
}

static inline __device__ int4 hsub8(int4 a, int4 b) {
    int4 c;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a.x), "r"(b.x));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a.y), "r"(b.y));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a.z), "r"(b.z));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a.w), "r"(b.w));
    return c;
}

static inline __device__ int4 hmul8(int a, int4 b) {
    int4 c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.z) : "r"(a), "r"(b.z));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.w) : "r"(a), "r"(b.w));
    return c;
}

static inline __device__ int float2_to_half2(float a, float b) {
    int c;
    asm volatile( \
        "{\n" \
        "    .reg .f16 lo, hi;\n" \
        "    cvt.rn.f16.f32 lo, %1;\n" \
        "    cvt.rn.f16.f32 hi, %2;\n" \
        "    mov.b32 %0, {lo, hi};\n" \
        "}\n" : "=r"(c) : "f"(a), "f"(b));
    return c;
}
#endif

template< typename T >
static inline __device__ T clamp( T x, T lb, T ub )
{
    return x < lb ? lb : (x > ub ? ub : x);
}

#if( __LWDA_ARCH__ >= 530 || !defined( __LWDA_ARCH__ ) )
static inline __device__ int4 hlerp8( uint32_t a, const int4 &x, uint32_t b, const int4 &y )
{
    int4 z;
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.x) : "r"(a), "r"(x.x));
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.y) : "r"(a), "r"(x.y));
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.z) : "r"(a), "r"(x.z));
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.w) : "r"(a), "r"(x.w));

    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.x) : "r"(b), "r"(y.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.y) : "r"(b), "r"(y.y));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.z) : "r"(b), "r"(y.z));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.w) : "r"(b), "r"(y.w));
    return z;
}
#endif

#if( __LWDA_ARCH__ >= 530 || !defined( __LWDA_ARCH__ ) )
static inline __device__ int2 hlerp4( uint32_t a, const int2 &x, uint32_t b, const int2 &y )
{
    int2 z;
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.x) : "r"(a), "r"(x.x));
    asm volatile("mul.f16x2 %0, %1, %2;"        : "=r"(z.y) : "r"(a), "r"(x.y));

    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.x) : "r"(b), "r"(y.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.y) : "r"(b), "r"(y.y));
    return z;
}
#endif

#if( __LWDA_ARCH__ >= 530 || !defined( __LWDA_ARCH__ ) )
static inline __device__ int2 hadd4(int2 a, int2 b)
{
    int2 c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a.x), "r"(b.x));
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a.y), "r"(b.y));
return c;
}

static inline __device__ int2 hmul4(int a, int2 b)
{
    int2 c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.x) : "r"(a), "r"(b.x));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c.y) : "r"(a), "r"(b.y));
    return c;
}

static inline __device__ int2 fma4(int2 a, int2 b, int c)
{
    int2 z;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(z.x) : "r"(c), "r"(a.x));
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(z.y) : "r"(c), "r"(a.y));

    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.x) : "r"(c), "r"(b.x));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(z.y) : "r"(c), "r"(b.y));
    return z;
}
#endif

};
