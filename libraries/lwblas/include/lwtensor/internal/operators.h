/**
* @file
* @brief This file contains all host and device operators.
*/

#pragma once

/* Bug 200478707 fixed. */
#include <cmath>
/* Fix Window's compilation issue. */
#include <algorithm>
#include <lwda_fp16.h>
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
#include <lwda_bf16.h>
#endif
#include <lwda_runtime.h>
#include <lwtensor/internal/operatorsPLC3.h>

#define INLINE_OPERATOR_PREFIX __inline__ __host__ __device__
#define OPERATOR_PREFIX __host__ __device__ static

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <cfloat>
namespace LWTENSOR_NAMESPACE
{

typedef struct  {
    unsigned short x;
} my_bfloat16_raw;

// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from lwComplex
// ----------------------------------------------------------------------------
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet( lwComplex );
/** Upcasting to lwDoublecomplex. */
template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet( const lwComplex x ) { return lwComplexFloatToDouble( x ); }


// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from lwDoubleComplex
// ----------------------------------------------------------------------------
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet( lwDoubleComplex );
/** Identity transformation. */
template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet( lwDoubleComplex x ) { return x; }
/** Upcasting to lwDoublecomplex. */
template<>
INLINE_OPERATOR_PREFIX
lwComplex lwGet( const lwDoubleComplex x ) { return lwComplexDoubleToFloat( x ); }

template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(signed char x) { return (make_lwDoubleComplex(static_cast<double>(x), 0.0)); }

template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(int x) { return (make_lwDoubleComplex(static_cast<double>(x), 0.0)); }


template <>
INLINE_OPERATOR_PREFIX
lwComplex lwGet<lwComplex>(const unsigned x) { return (make_lwComplex(float(x), 0.0F)); }

template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(const unsigned x) { return (make_lwDoubleComplex(static_cast<double>(x), 0.0)); }


template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(float x) { return (make_lwDoubleComplex(double(x), 0.0)); }


template <>
INLINE_OPERATOR_PREFIX
lwComplex lwGet<lwComplex>(half x) { return (make_lwComplex(lwGet<float>(x), 0.0F)); }

template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(half x) { return (make_lwDoubleComplex(static_cast<double>(lwGet<float>(x)), 0.0)); }

template <>
INLINE_OPERATOR_PREFIX
lwComplex lwGet<lwComplex>(double x) { return (make_lwComplex(float(x), 0.0F)); }

template <>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwGet<lwDoubleComplex>(double x) { return (make_lwDoubleComplex(double(x), 0.0)); }

template<>
INLINE_OPERATOR_PREFIX
bool lwIsEqual( const lwDoubleComplex lhs, const lwDoubleComplex rhs ) { return ( lhs.x == rhs.x ) && ( lhs.y == rhs.y ); }

template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwMul( const lwDoubleComplex a, const lwDoubleComplex b )
{
    return lwCmul( a, b );
}
template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwAdd( lwDoubleComplex a, lwDoubleComplex b )
{
    return lwCadd( a, b );
}
template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwSub( const lwDoubleComplex a, const lwDoubleComplex b )
{
    return lwCsub( a, b );
}
template<>
INLINE_OPERATOR_PREFIX
double lwSquare2Norm( lwComplex x ) { return static_cast<double>((x.x * x.x) + (x.y * x.y)); }
template<>
INLINE_OPERATOR_PREFIX
double lwSquare2Norm( const lwDoubleComplex x ) { return (double)((x.x * x.x) + (x.y * x.y)); }

/** This is not standard but for function complete.  */
template<>
INLINE_OPERATOR_PREFIX
lwComplex lwMax( lwComplex a, lwComplex b )
{
    return (lwSquare2Norm( a ) > lwSquare2Norm( b )) ? a : b;
}
/** This is not standard but for function complete.  */
template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwMax( lwDoubleComplex a, lwDoubleComplex b )
{
    return lwSquare2Norm( a ) > lwSquare2Norm( b ) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( lwComplex a ) { return std::sqrt(lwSquare2Norm( a )); }
template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( const lwDoubleComplex a ) { return std::sqrt(lwSquare2Norm( a )); }

template<typename typeA, typename typeB, typename typeCompute, lwtensorOperator_t op, bool isRealTimesComplex = false>
struct Operator
{
   OPERATOR_PREFIX
   typeCompute execute(typeA a, typeB b) = delete;
};

template<typename typeA, typename typeB, typename typeCompute, bool isRealTimesComplex>
struct Operator<typeA, typeB, typeCompute, lwtensorOperator_t::LWTENSOR_OP_MUL, isRealTimesComplex>
{
   OPERATOR_PREFIX
   typeCompute execute( typeA a, typeB b )
   {
      return lwMul( lwGet<typeCompute>( a ), lwGet<typeCompute>( b ) );
   }
};

template<bool isRealTimesComplex>
struct Operator<lwComplex, lwComplex, lwComplex, lwtensorOperator_t::LWTENSOR_OP_MUL, isRealTimesComplex>
{
   OPERATOR_PREFIX
   lwComplex execute( lwComplex a, lwComplex b )
   {
       if( isRealTimesComplex )
       {
           return make_lwFloatComplex  ((lwCrealf(a) * lwCrealf(b)),
                                        (lwCrealf(a) * lwCimagf(b)));
       }else{
           return lwMul( a, b );
       }
   }
};

template<bool isRealTimesComplex>
struct Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, lwtensorOperator_t::LWTENSOR_OP_MUL, isRealTimesComplex>
{
   OPERATOR_PREFIX
   lwDoubleComplex execute( lwDoubleComplex a, lwDoubleComplex b )
   {
       if( isRealTimesComplex )
       {
           return make_lwDoubleComplex  ((lwCreal(a) * lwCreal(b)),
                   (lwCreal(a) * lwCimag(b)));
       }else{
           return lwMul( a, b );
       }
   }
};

template<typename typeA, typename typeB, typename typeCompute, bool isRealTimesComplex>
struct Operator<typeA, typeB, typeCompute, lwtensorOperator_t::LWTENSOR_OP_ADD, isRealTimesComplex>
{
   OPERATOR_PREFIX
   typeCompute  execute(typeA a, typeB b)
   {
      return lwAdd( lwGet<typeCompute>( a ), lwGet<typeCompute>( b ) );
   }
};

template<typename typeA, typename typeB, typename typeCompute, bool isRealTimesComplex>
struct Operator<typeA, typeB, typeCompute, lwtensorOperator_t::LWTENSOR_OP_MAX, isRealTimesComplex>
{
   OPERATOR_PREFIX
   typeCompute  execute(typeA a, typeB b)
   {
      return lwMax( lwGet<typeCompute>( a ), lwGet<typeCompute>( b ) );
   }
};

template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwtensorUnaryOp( const lwDoubleComplex x, const lwtensorOperator_t op, const ElementwiseParameters::ActivationContext *ctx)
{
    switch(op)
    {
        case lwtensorOperator_t::LWTENSOR_OP_IDENTITY:
            return x;
        default: //case lwtensorOperator_t::LWTENSOR_OP_CONJ:
            return lwConj( x );
    }
}

template<>
INLINE_OPERATOR_PREFIX
lwDoubleComplex lwtensorBinaryOp( const lwDoubleComplex x, const lwDoubleComplex y, const lwtensorOperator_t op, const ElementwiseParameters::ActivationContext *ctx,
        const lwtensorOperator_t unaryOp)
{
   switch(op)
   {
      case lwtensorOperator_t::LWTENSOR_OP_ADD:
         return lwCadd( x, y );
      default: //case lwtensorOperator_t::LWTENSOR_OP_MUL:
         return lwCmul( x, y );
   }
}



static __inline__ __device__ half loadVolatile(const half* addr) {
    const volatile __half_raw* chr = (reinterpret_cast<const volatile __half_raw *>(addr) );
    __half_raw hr;
    hr.x = chr[0].x;
    return half( hr );
}
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
static __inline__ __device__ BFloat16 loadVolatile(const BFloat16* addr) {
    const volatile my_bfloat16_raw* chr = (reinterpret_cast<const volatile my_bfloat16_raw*>(addr) );
    my_bfloat16_raw hr;
    hr.x = chr[0].x;
    return *reinterpret_cast<BFloat16*>(&hr);
}
#endif
static __inline__ __device__ int loadVolatile(const int* addr)
{
    int val;
    volatile int* tmp = (volatile int*) addr;

    val = tmp[0];
    return val;
}
static __inline__ __device__ float loadVolatile(const float* addr)
{
    float val;
    volatile float* tmp = (volatile float*) addr;

    val = tmp[0];
    return val;
}
static __inline__ __device__ lwComplex loadVolatile(const lwComplex* addr)
{
    union {
        const lwComplex* lwC;
        volatile const unsigned long long* ull;
    } tmp1 = {addr};
    union {
        unsigned long long ull;
        lwComplex lwC;
    } tmp2 = {*tmp1.ull};

    return tmp2.lwC;
}
static __inline__ __device__ double loadVolatile(const double* addr)
{
    double val;
    volatile double* tmp = (volatile double*) addr;

    val = tmp[0];
    return val;
}
static __inline__ __device__ lwDoubleComplex loadVolatile(const lwDoubleComplex* addr)
{
    lwDoubleComplex val;
    volatile double* tmp = (volatile double*) addr;

    val.x = tmp[0];
    val.y = tmp[1];
    return val;
}
/* functions used to handling volatile stores */
static __inline__ __device__ void storeVolatile(int* addr, int val)
{
    volatile int* tmp = (volatile int*) addr;
    tmp[0] = val;
}

// Largest positive FP16 value, corresponds to 6.5504e+04
static __inline__ __device__ __host__ half
hmax() {
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    __half_raw hr;
    hr.x = 0x7bffU;
    return reinterpret_cast<half&>(hr);
}

// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
static __inline__ __device__ __host__ half
hmin() {
    // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
    __half_raw hr;
    hr.x = 0x0400U;
    return reinterpret_cast<half&>(hr);
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
// Largest positive FP16 value, corresponds to
static __inline__ __device__ __host__ BFloat16
bf16max() {
    // Exponent all ones except LSB, mantissa is all ones (7bits) (0x8f)
    my_bfloat16_raw hr;
    hr.x = 0x7f7fU;
    return reinterpret_cast<BFloat16&>(hr);
}

// Smallest positive (normalized) BF16 value, corresponds to 
static __inline__ __device__ __host__ BFloat16
bf16min() {
    // Exponent is 0x01 (8 bits), mantissa is all zeros (7 bits)
    my_bfloat16_raw hr;
    hr.x = 0x0080U;
    return reinterpret_cast<BFloat16&>(hr);
}
#endif


template <typename T_ELEM>
__inline__ __device__ __host__ T_ELEM
lwMaxOfType();
template <>
__inline__ __device__ __host__ half
lwMaxOfType<half>() {
    return hmax();
}
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
__inline__ __device__ __host__ BFloat16
lwMaxOfType<BFloat16>() {
    return bf16max();
}
#endif
template <>
__inline__ __device__ __host__ float
lwMaxOfType<float>() {
    return FLT_MAX;
}
template <>
__inline__ __device__ __host__ double
lwMaxOfType<double>() {
    return DBL_MAX;
}
template <>
__inline__ __device__ __host__ lwComplex
lwMaxOfType<lwComplex>() {
    return lwGet<lwComplex>(FLT_MAX);
}
template <>
__inline__ __device__ __host__ lwDoubleComplex
lwMaxOfType<lwDoubleComplex>() {
    return lwGet<lwDoubleComplex>(DBL_MAX);
}

/**
 * Returns the lowest, representable value for the given element type (for the most part
 * this should correspond to a negative vaule).
 * For floating point numbers this is equal to the negative maximal value.
 */
template <typename T_ELEM>
__inline__ __device__ __host__ T_ELEM
lwLowestOfType();
template <>
__inline__ __device__ __host__ half
lwLowestOfType<half>() {
    return lwNeg(hmax());
}
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
__inline__ __device__ __host__ BFloat16
lwLowestOfType<BFloat16>() {
    return lwNeg(bf16max());
}
#endif
template <>
__inline__ __device__ __host__ float
lwLowestOfType<float>() {
    return -FLT_MAX;
}
template <>
__inline__ __device__ __host__ double
lwLowestOfType<double>() {
    return -DBL_MAX;
}
template <>
__inline__ __device__ __host__ lwComplex
lwLowestOfType<lwComplex>() {
    return lwGet<lwComplex>(-FLT_MAX);
}
template <>
__inline__ __device__ __host__ lwDoubleComplex
lwLowestOfType<lwDoubleComplex>() {
    return lwGet<lwDoubleComplex>(-DBL_MAX);
}
}

#pragma GCC diagnostic pop
