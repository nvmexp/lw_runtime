#ifndef LWTENSOR_OPERATORSPLC3_H
#define LWTENSOR_OPERATORSPLC3_H

/**
* @file
* @brief This file contains all host and device operators.
*/

#include <cmath>
#include <algorithm>
#include <lwda_fp16.h>
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
#include <lwda_bf16.h>
#endif
#include <lwda_runtime.h>
#include <lwComplex.h>

#include <lwtensor/internal/types.h>
#include<lwtensor/internal/defines.h>

#define INLINE_OPERATOR_PREFIX __inline__ __host__ __device__
#define OPERATOR_PREFIX __host__ __device__ static

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace LWTENSOR_NAMESPACE
{
// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from int8_t
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from int8_t (int8_t) to output datatype.
 * \param[in] x input value in int8_t
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet(const int8_t x);

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(const int8_t x) { return (x); }

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const int8_t x) { return static_cast<uint8_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
int lwGet<int>(const int8_t x) { return static_cast<int>(x); }

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const int8_t x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __float2half_rn(static_cast<float>(x));
#else
    const uint16_t zero = 0;
    const void* ptr = &zero;
    return *(static_cast<const half*>( ptr ));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const int8_t x)
{
  return BFloat16(static_cast<float>(x));
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const int8_t x) { return static_cast<float>(x); }

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const int8_t x) { return static_cast<uint32_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const int8_t x) { return static_cast<double>(x); }


// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from int
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from int to output datatype.
 * \param[in] x input value in int
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet(const int32_t);

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(const int32_t x)
{
    int32_t intResult;
#if defined(__LWDA_ARCH__)
    asm("cvt.sat.s8.s32 %0, %1;" : "=r"(intResult) : "r"(x));
#else
    if (x < (SCHAR_MIN))
    {
        intResult = static_cast<int>(SCHAR_MIN);
    }
    else if (x > (SCHAR_MAX))
    {
        intResult = static_cast<int>(SCHAR_MAX);
    }
    else
    {
        intResult = static_cast<int>(x);
    }
#endif
    return static_cast<int8_t>(intResult);
}

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const int32_t x)
{
    uint32_t intResult;
#if defined(__LWDA_ARCH__)
    asm("cvt.sat.u8.s32 %0, %1;" : "=r"(intResult) : "r"(x));
#else
    if (x < 0)
    {
        intResult = static_cast<uint32_t>(0);
    }
    else if (x > (UCHAR_MAX))
    {
        intResult = static_cast<uint32_t>(UCHAR_MAX);
    }
    else
    {
        intResult = static_cast<uint32_t>(x);
    }
#endif
    return static_cast<uint8_t>(intResult);
}

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const int32_t x) { return static_cast<uint32_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
int32_t lwGet<int32_t>(const int32_t x) { return x; }

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const int32_t x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __float2half_rn(static_cast<float>(x));
#else
    const uint16_t zero = 0;
    const void* ptr = &zero;
    return *(static_cast<const half*>(ptr));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const int32_t x)
{
  return BFloat16(static_cast<float>(x));
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const int32_t x) { return static_cast<float>(x); }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const int32_t x) { return static_cast<double>(x); }


// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from uint32_t
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from uint32_t to output datatype.
 * \param[in] x input value in uint32_t
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet(const uint32_t x);

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const uint32_t x) { return static_cast<uint32_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const uint32_t x) { return static_cast<uint8_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(const uint32_t x) { return static_cast<int8_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
int lwGet<int>(const uint32_t x) { return static_cast<int>(x); }

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const uint32_t x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __float2half_rn(static_cast<float>(x));
#else
    const uint16_t zero = 0;
    const void* ptr = &zero;
    return *(static_cast<const half*>( ptr ));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const uint32_t x)
{
  return BFloat16(static_cast<float>(x));
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const uint32_t x) { return static_cast<float>(x); }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const uint32_t x) { return static_cast<double>(x); }


// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from float
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from float to output datatype.
 * \param[in] x input value in float
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
auto lwGet(const float) -> T_ELEM;

template <>
INLINE_OPERATOR_PREFIX
int lwGet<int>(const float x) { return static_cast<int>(x); }

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(float x)
{
  // Necessary buffer for colwersion
  int intResult;
  // Compiles into F2I.S32.F32 & I2I.S8.S32.SAT
#if defined(__LWDA_ARCH__)
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(intResult) : "f"(x));
#else
  /* TRT compliance (round-to-nearest-even). */
  x = std::max(x, static_cast<float>(SCHAR_MIN));
  x = std::min(x, static_cast<float>(SCHAR_MAX));
  if (std::abs(x - static_cast<int>(x) ) == 0.5)
  {
      intResult = lwGet<int>(std::round( x / 2.f ) * 2.f);
  }
  else
  {
      intResult = lwGet<int>(std::round( x ));
  }
#endif
  // Return casted result (determined result after colwersions)
  return static_cast<int8_t>(intResult);
}

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const float x) { return static_cast<uint32_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const float x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __float2half_rn(float(x));
#else
    const uint16_t zero = 0;
    const void* ptr = &zero;
    return *(static_cast<const half*>( ptr ));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const float x)
{
  return BFloat16(x);
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const float x) { return x; }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const float x) { return static_cast<double>(x); }

// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from half
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from half to output datatype.
 * \param[in] x input value in half
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet(const half);

template <>
INLINE_OPERATOR_PREFIX
int lwGet<int>(const half x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return static_cast<int>(__half2float(x));
#else
    return static_cast<int>(0);
#endif
}

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const half x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return static_cast<uint32_t>(__half2float(x));
#else
    return static_cast<uint32_t>(0);
#endif
}

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const half x) { return x; }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const half x)
{
  return BFloat16(static_cast<float>(x));
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const half x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __half2float(x);
#else
    return static_cast<float>(0);
#endif
}

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(const half x) { return lwGet<int8_t>(lwGet<float>(x)); }

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const half x) { return lwGet<float>(x); }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const half x) { return static_cast<double>(lwGet<float>(x)); }

// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from double
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from double to output datatype.
 * \param[in] x input value in double
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
auto lwGet(const double) -> T_ELEM;

template <>
INLINE_OPERATOR_PREFIX
int8_t lwGet<int8_t>(const double x)
{
    // Necessary buffer for colwersion
    int intResult;
    // F2I & I2I
#if defined(__LWDA_ARCH__)
    asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(intResult) : "d"(x));
#else
    if( x < (SCHAR_MIN) )
    {
        intResult = static_cast<int>(SCHAR_MIN) ;
    }
    else if( x > (SCHAR_MAX))
    {
        intResult = static_cast<int>(SCHAR_MAX);
    }
    else
    {
        intResult = static_cast<int>(x);
    }
#endif
    // Return casted result (determined result after colwersions)
    return static_cast<int8_t>(intResult);
}

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const double x)
{
    // Necessary buffer for colwersion
    uint32_t intResult;

    // F2I & I2I
#if defined(__LWDA_ARCH__)
    asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(intResult) : "d"(x));
#else
    if( x < 0U )
    {
        intResult = 0U;
    }
    else if( x > (UCHAR_MAX) )
    {
        intResult = (UCHAR_MAX);
    }
    else
    {
        intResult = x;
    }
#endif
    // Return casted result (determined result after colwersions)
    return static_cast<uint8_t>(intResult);
}

template <>
INLINE_OPERATOR_PREFIX
uint8_t lwGet<uint8_t>(const float x) { return static_cast<uint8_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
int lwGet<int>(const double x) { return static_cast<int>(x); }

template <>
INLINE_OPERATOR_PREFIX
uint32_t lwGet<uint32_t>(const double x) { return static_cast<uint32_t>(x); }

template <>
INLINE_OPERATOR_PREFIX
half lwGet<half>(const double x)
{
#ifndef LWDART_VERSION
#error LWDART_VERSION Undefined!
#elif ( LWDART_VERSION >= 9200 ) ||  defined(__LWDA_ARCH__)
    return __float2half_rn(static_cast<float>(x));
#else
    const uint16_t zero = 0;
    const void* ptr = &zero;
    return *(static_cast<const half*>( ptr ));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const double x)
{
  return BFloat16(x);
}
#endif

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const double x) { return static_cast<float>(x); }

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const double x) { return x; }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
auto lwGet(const BFloat16) -> T_ELEM;

template <>
INLINE_OPERATOR_PREFIX
BFloat16 lwGet<BFloat16>(const BFloat16 x)
{
  return x;
}

template <>
INLINE_OPERATOR_PREFIX
float lwGet<float>(const BFloat16 x)
{
  return static_cast<float>(x);
}

template <>
INLINE_OPERATOR_PREFIX
double lwGet<double>(const BFloat16 x)
{
  return static_cast<double>(x);
}
#endif

// ----------------------------------------------------------------------------
// Functions to initialize T_ELEM from lwComplex
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Colwert a value from lwComplex to output datatype.
 * \param[in] x input value in lwComplex
 * \param[in] T_ELEM the output datatype
 * \return value in type T_ELEM
 * \behavior blocking, reentrant, and thread safe
 */
template <typename T_ELEM>
INLINE_OPERATOR_PREFIX
T_ELEM lwGet(const lwComplex x);

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwGet(const lwComplex x) { return x; }

template<>
INLINE_OPERATOR_PREFIX
int8_t lwGet(const lwComplex x) { return lwGet<int8_t>(x.x); }

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwGet(const int8_t x) { return make_lwComplex(lwGet<float>(x), 0.0f); }

template<>
INLINE_OPERATOR_PREFIX
float lwGet(const lwComplex x) { return x.x; }

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwGet(const float x) { return make_lwComplex(x, 0.0f); }

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwGet(const int32_t x) { return make_lwComplex(lwGet<float>(x), 0.0f); }


// ----------------------------------------------------------------------------
// lwSqrt
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief compute the square root of input
 * \param[in] x input value in typeCompute
 * \param[in] typeCompute the input and output type
 * \return the square root of x
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwSqrt(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>( sqrtf( lwGet<float>( x ) ) );
}

template<>
INLINE_OPERATOR_PREFIX
double lwSqrt(const double x)
{
    return sqrt(x);
}

template<>
INLINE_OPERATOR_PREFIX
float lwSqrt(const float x)
{
    return sqrtf( x );
}

// ----------------------------------------------------------------------------
// lwSub
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute a - b
 * \param[in] a input value in typeCompute
 * \param[in] b input value in typeCompute
 * \param[in] typeCompute the input and output type
 * \return a - b in typeCompute
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwSub(const typeCompute a, const typeCompute b);

template<>
INLINE_OPERATOR_PREFIX
half lwSub(const half a, const half b);

template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwAdd(const typeCompute a, const typeCompute b);

template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwMul(const typeCompute a, const typeCompute b);

template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwDiv(const typeCompute a, const typeCompute b);

template <>
INLINE_OPERATOR_PREFIX half lwDiv(const half a, const half b);

template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwMin(const typeCompute a, const typeCompute b);

template <typename typeCompute>
INLINE_OPERATOR_PREFIX typeCompute lwMax(const typeCompute a, const typeCompute b);

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Substract lhs from rhs in typeCompute.
 * \param[in] lhs the first value
 * \param[in] rhs the second value
 * \param[in] typeCompute the input value datatype
 * \return the substraction of two values
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwSub(const typeCompute a, const typeCompute b ) -> typeCompute
{
    // Need to handle uint32_t value specially?
    return a - b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwSub(const half a, const half b)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
   return __hsub(a, b);
#else
   return lwGet<half>(lwGet<float>(a) - lwGet<float>(b));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwSub(const BFloat16 a, const BFloat16 b)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
   return __hsub(a, b);
#else
   return lwGet<BFloat16>(lwGet<float>(a) - lwGet<float>(b));
#endif
}
#endif

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwSub(const lwComplex a, const lwComplex b )
{
    return make_lwComplex(a.x - b.x, a.y - b.y);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the division of two values.
 * \param[in] a the first value
 * \param[in] b the second value
 * \param[in] typeCompute the input value datatype
 * \return the division of two values
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwDiv(const typeCompute a, const typeCompute b) -> typeCompute
{
    return a / b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwDiv(const half a, const half b )
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return __hdiv(a, b);
#else
    return lwGet<half>(lwDiv(lwGet<float>(a), lwGet<float>(b)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwDiv(const BFloat16 a, const BFloat16 b )
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return __hdiv(a, b);
#else
    return lwGet<BFloat16>(lwDiv(lwGet<float>(a), lwGet<float>(b)));
#endif
}
#endif




// ----------------------------------------------------------------------------
// lwScalef
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Scale a value in typeCompute.
 * \param[in] x the input value
 * \param[in] typeCompute the compute type
 * \param[in] scale the factor in float
 * \return the scaled value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwScalef(const typeCompute x, const float scale) -> typeCompute
{
    return lwMul(x, lwGet<typeCompute>(scale));
}

// ----------------------------------------------------------------------------
// lwRcp
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the reciprocal of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the reciprocal of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwRcp(const typeCompute x) -> typeCompute
{
    return lwDiv(lwGet<typeCompute>(1.F), x);
}

template <>
INLINE_OPERATOR_PREFIX half lwRcp(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hrcp(x);
#else
    return lwDiv(lwGet<half>(1.F), x);
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwRcp(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hrcp(x);
#else
    return lwDiv(lwGet<BFloat16>(1.F), x);
#endif
}
#endif


// ----------------------------------------------------------------------------
// lwExp
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the natural exponent of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the natural exponent of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwExp(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(exp(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwExp(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hexp(x);
#else
    return lwGet<half>(expf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwExp(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hexp(x);
#else
    return lwGet<BFloat16>(expf(lwGet<float>(x)));
#endif
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwExp(const float x)
{
    return expf(x);
}

// ----------------------------------------------------------------------------
// lwLn
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the natural logarithm of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the natural logarithm of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwLn(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(log(x));
}

template <>
INLINE_OPERATOR_PREFIX float lwLn(const float x)
{
    return logf(x);
}

template <>
INLINE_OPERATOR_PREFIX half lwLn(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hlog(x);
#else
    return lwGet<half>(logf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwLn(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hlog(x);
#else
    return lwGet<BFloat16>(logf(lwGet<float>(x)));
#endif
}
#endif

// ----------------------------------------------------------------------------
// lwSin
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the sine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the sine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwSin(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(sin(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwSin(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hsin(x);
#else
    return lwGet<half>(sinf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwSin(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hsin(x);
#else
    return lwGet<BFloat16>(sinf(lwGet<float>(x)));
#endif
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwSin(const float x)
{
    return sinf(x);
}


// ----------------------------------------------------------------------------
// lwCos
// ----------------------------------------------------------------------------

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the cosine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the cosine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwCos(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(cos(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwCos(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hcos(x);
#else
    return lwGet<half>(cosf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwCos(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hcos(x);
#else
    return lwGet<BFloat16>(cosf(lwGet<float>(x)));
#endif
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwCos(const float x)
{
    return cosf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the tangent of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the tangent of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwTan(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(tan(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwTan(const half x)
{
    return lwGet<half>(tanf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwTan(const BFloat16 x)
{
    return lwGet<BFloat16>(tanf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwTan(const float x)
{
    return tanf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the hyperbolic sine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the hyperbolic sine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwSinh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(sinh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwSinh(const half x)
{
    return lwGet<half>(sinhf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwSinh(const BFloat16 x)
{
    return lwGet<BFloat16>(sinhf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwSinh(const float x)
{
    return sinhf(x);
}


/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the hyperbolic cosine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the hyperbolic cosine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwCosh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(cosh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwCosh(const half x)
{
    return lwGet<half>(coshf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwCosh(const BFloat16 x)
{
    return lwGet<BFloat16>(coshf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwCosh(const float x)
{
    return coshf(x);
}


/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the hyperbolic tangent of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the hyperbolic tangent of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwTanh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(tanh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwTanh(const half x)
{
    return lwGet<half>(tanhf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwTanh(const BFloat16 x)
{
    return lwGet<BFloat16>(tanhf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwTanh(const float x)
{
    return tanhf(x);
}


/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse sine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse sine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAsin(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(asin(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAsin(const half x)
{
    return lwGet<half>(asinf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAsin(const BFloat16 x)
{
    return lwGet<BFloat16>(asinf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwAsin(const float x)
{
    return asinf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse cosine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse cosine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAcos(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(acos(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAcos(const half x)
{
    return lwGet<half>(acosf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAcos(const BFloat16 x)
{
    return lwGet<BFloat16>(acosf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwAcos(const float x)
{
    return acosf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse tangent of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse tangent of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAtan(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(atan(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAtan(const half x)
{
    return lwGet<half>(atanf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAtan(const BFloat16 x)
{
    return lwGet<BFloat16>(atanf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwAtan(const float x)
{
    return atanf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse hyperbolic sine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse hyperbolic sine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAsinh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(asinh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAsinh(const half x)
{
    return lwGet<half>(asinhf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAsinh(const BFloat16 x)
{
    return lwGet<BFloat16>(asinhf(lwGet<float>(x)));
}
#endif


template <>
INLINE_OPERATOR_PREFIX float lwAsinh(const float x)
{
    return asinhf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse hyperbolic cosine of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse hyperbolic cosine of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAcosh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(acosh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAcosh(const half x)
{
    return lwGet<half>(acoshf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAcosh(const BFloat16 x)
{
    return lwGet<BFloat16>(acoshf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwAcosh(const float x)
{
    return acoshf(x);
}


/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the ilwerse hyperbolic tangent of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the ilwerse hyperbolic tangent of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwAtanh(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(atanh(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwAtanh(const half x)
{
    return lwGet<half>(atanhf(lwGet<float>(x)));
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwAtanh(const BFloat16 x)
{
    return lwGet<BFloat16>(atanhf(lwGet<float>(x)));
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwAtanh(const float x)
{
    return atanhf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the greatest integer less than or equal to a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the greatest integer less than or equal to the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwFloor(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(floor(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwFloor(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hfloor(x);
#else
    return lwGet<half>(floorf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwFloor(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hfloor(x);
#else
    return lwGet<BFloat16>(floorf(lwGet<float>(x)));
#endif
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwFloor(const float x)
{
    return floorf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the least integer greater than or equal to a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the least integer greater than or equal to the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwCeil(const typeCompute x) -> typeCompute
{
    return lwGet<typeCompute>(ceil(x));
}

template <>
INLINE_OPERATOR_PREFIX half lwCeil(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return hceil(x);
#else
    return lwGet<half>(ceilf(lwGet<float>(x)));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template <>
INLINE_OPERATOR_PREFIX BFloat16 lwCeil(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return hceil(x);
#else
    return lwGet<BFloat16>(ceilf(lwGet<float>(x)));
#endif
}
#endif

template <>
INLINE_OPERATOR_PREFIX float lwCeil(const float x)
{
    return ceilf(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the sigmoid function of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the sigmoid of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwSigmoid(const typeCompute x) -> typeCompute
{
    // y = 0.5f*(tanh(0.5*x)+1);
    return lwScalef(lwAdd(lwTanh(lwScalef(x, 0.5F)), lwGet<typeCompute>(1.F)), 0.5F);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the exponential linear unit (elu) of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \param[in] alpha coefficient in float
 * \return the exponential linear unit of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwElu(const typeCompute x, const float alpha) -> typeCompute
{
    // y=x for x > 0, alpha*(exp(x)–1) for x <= 0
    return (lwGet<float>(x) > 0.F) ? x : lwScalef(lwSub(lwExp(x), lwGet<typeCompute>(1.F)), alpha);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the leaky relu of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \param[in] k coefficient in float
 * \return the leaky relu of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwLeakyRelu(const typeCompute x, const float k) -> typeCompute
{
    // y=x for x >= 0, k*x for x < 0
    return (lwGet<float>(x) > 0.F) ? x : lwScalef(x, k);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the rectified linear unit (relu) of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the rectified linear unit of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwRelu(const typeCompute x) -> typeCompute
{
    return (lwGet<float>(x) < 0.F) ? lwGet<typeCompute>(0.F) : x;
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the scaled elu of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \param[in] eluAlpha the coefficient for elu in float
 * \param[in] outScale the factor for scaling the elu in float
 * \return the scaled elu of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwSelu(const typeCompute x, const float eluAlpha, const float outScale) -> typeCompute
{
    return lwScalef(lwElu(x, eluAlpha), outScale);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the opposite of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \return the opposite of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwNeg(const typeCompute x) -> typeCompute
{
    return lwMul(lwGet<typeCompute>(-1), x);
}

INLINE_OPERATOR_PREFIX
half lwNeg(const half x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return __hneg(x);
#else
    return lwSub(lwGet<half>(0.F), x);
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
INLINE_OPERATOR_PREFIX
BFloat16 lwNeg(const BFloat16 x)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return __hneg(x);
#else
    return lwSub(lwGet<BFloat16>(0.F), x);
#endif
}
#endif

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the absolute value of a signed value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype, signed.
 * \return the absolute value of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX
// half (maybe also some other types) would not pass is_signed, use !is_unsigned instead.
auto lwAbs(const typeCompute x) -> typename std::enable_if<!std::is_unsigned<typeCompute>::value, typeCompute>::type
{
    return (lwGet<float>(x) >= 0.F) ? x : lwMul(lwGet<typeCompute>(-1), x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the absolute value of an uint32_t value, i.e. itself.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype, uint32_t.
 * \return the absolute value of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwAbs(const typeCompute x) -> typename std::enable_if<std::is_unsigned<typeCompute>::value, typeCompute>::type
{
    return x;
}

INLINE_OPERATOR_PREFIX
half lwAbs(const half x)
{
    // operator- half is a device function, call lwSub instead.
    return (lwGet<float>(x) >= 0.F) ? x : lwNeg(x);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the softsign function of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype.
 * \return the softsign function of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwSoftSign(const typeCompute x) -> typeCompute
{
    // y=x/(1+abs(x))
    return lwDiv(x, lwAdd(lwAbs(x), lwGet<typeCompute>(1.F)));
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the softplus function of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype.
 * \param[in] inScale the scale for input in float.
 * \param[in] outScale the scale for output in float.
 * \param[in] approximateThreshold the threshold in float for using x as an approximate.
 * \return the softplus function of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwSoftPlus(const typeCompute x, const float inScale, const float outScale, const float approximateThreshold) -> typeCompute
{
    typeCompute y = lwScalef(x, inScale);
    // x as approximate when >=approximateThreshold
    if (lwGet<float>(y) < approximateThreshold)
    {
        // y=ln(1+exp(x))
        y = lwLn(lwAdd(lwExp(y), lwGet<typeCompute>(1.F)));
    }
    return lwScalef(y, outScale);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Clip a value with lower and upper bounds.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype.
 * \param[in] lower the lower bound in float.
 * \param[in] upper the upper bound in float.
 * \return the clipped value of the input value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto
lwClip(const typeCompute x, const float lower, const float upper) -> typeCompute
{
    return lwMax(lwMin(x, lwGet<typeCompute>(upper)), lwGet<typeCompute>(lower));
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the hard sigmoid of a value.
 * A hard sigmoid is a 3-part piecewise linear approximation (output 0, line with slope rising from 0 to 1, output 1)
 * of a sigmoid function for better computation speed.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \param[in] slope the slope for the hard sigmoid in float.
 * \return the hard sigmoid of the input value
 * \behavior blocking, reentrant, and thread safe
 */

template <typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwHardSigmoid(const typeCompute x, const float slope, const float shift) -> typeCompute
{
    // y=clip((x*slope)+shift), 0, 1)
    // shift being 0.5 keeps y=0.5 for x=0;
    return lwGet<typeCompute>(lwClip(lwAdd(lwScalef(x, slope), lwGet<typeCompute>(shift)), 0.F, 1.F));
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the scaled tanh of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype
 * \param[in] inScale the scale for input in float.
 * \param[in] outScale the scale for output in float.
 * \return the scaled tanh of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwScaledTanh(const typeCompute x, const float inScale, const float outScale) -> typeCompute
{
    return lwScalef(lwTanh(lwScalef(x, inScale)), outScale);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the thresholded relu of a value.
 * \param[in] x the value
 * \param[in] typeCompute the input value datatype.
 * \param[in] threshold the threshold for relu in float.
 * \return the thresholded relu of the value
 * \behavior blocking, reentrant, and thread safe
 */
template <typename typeCompute>
INLINE_OPERATOR_PREFIX auto lwThresholdedRelu(const typeCompute x, const float threshold) -> typeCompute
{
    // y=x for x>theta, y=0 otherwise
    return (lwGet<float>(x) < threshold) ? lwGet<typeCompute>(0.F) : x;
}

// GELU is not required.

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Check if two values in typeCompute are the same.
 * \param[in] lhs the first value
 * \param[in] rhs the second value
 * \param[in] typeCompute the input value datatype
 * \return whether the two values are the same
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
bool lwIsEqual(const typeCompute lhs, const typeCompute rhs) { return lhs == rhs; }

template<>
INLINE_OPERATOR_PREFIX
bool lwIsEqual(const half lhs, const half rhs) { return lwIsEqual(lwGet<float>(lhs), lwGet<float>(rhs)); }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
bool lwIsEqual(const BFloat16 lhs, const BFloat16 rhs) {
  #ifdef __LWDA_ARCH__
  return lhs == rhs;  // operator== is only available in device code
  #else
  return lwIsEqual(lwGet<float>(lhs), lwGet<float>(rhs));
  #endif
}
#endif

template<>
INLINE_OPERATOR_PREFIX
bool lwIsEqual(const lwComplex lhs, const lwComplex rhs)
{
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the multiplication of two values.
 * \param[in] lhs the first value
 * \param[in] rhs the second value
 * \param[in] typeCompute the input value datatype
 * \return the multiplication of two values
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwMul(const typeCompute a, const typeCompute b) -> typeCompute
{
    return a * b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwMul(const half a, const half b )
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
   return __hmul( a, b );
#else
   return lwGet<half>(lwGet<float>(a) * lwGet<float>(b));
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwMul(const BFloat16 a, const BFloat16 b )
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
   return __hmul( a, b );
#else
   return lwGet<BFloat16>(lwGet<float>(a) * lwGet<float>(b));
#endif
}
#endif

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwMul(const lwComplex a, const lwComplex b)
{
    return lwCmulf(a, b);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Add two values in typeCompute.
 * \param[in] lhs the first value
 * \param[in] rhs the second value
 * \param[in] typeCompute the input value datatype
 * \return the sum of two values
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwAdd(const typeCompute a, const typeCompute b ) -> typeCompute
{
    return a + b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwAdd(const half a, const half b)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
   return __hadd( a, b );
#else
   return lwGet<half>( lwGet<float>( a ) + lwGet<float>( b ) );
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwAdd(const BFloat16 a, const BFloat16 b)
{
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
   return __hadd( a, b );
#else
   return lwGet<BFloat16>( lwGet<float>( a ) + lwGet<float>( b ) );
#endif
}
#endif

template<>
INLINE_OPERATOR_PREFIX
lwComplex lwAdd(const lwComplex a, const lwComplex b)
{
    return make_lwComplex(a.x + b.x, a.y + b.y);
}

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Compute the square of input in typeCompute and output in double
 * \param[in] x input in typeCompute
 * \param[in] typeCompute the input value datatype
 * \return the square of input in double
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
double lwSquare2Norm(const typeCompute x) { return (double)(x * x); }

template<>
INLINE_OPERATOR_PREFIX
double lwSquare2Norm(const half x) { return lwMul( lwGet<double>( x ), lwGet<double>( x ) ); }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
double lwSquare2Norm(const BFloat16 x) { return lwMul( lwGet<double>( x ), lwGet<double>( x ) ); }
#endif

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Return the maximum value between a and b
 * \param[in] a input in typeCompute
 * \param[in] b input in typeCompute
 * \param[in] typeCompute the input value datatype
 * \return the maximum value between a and b
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto lwMax( const typeCompute a, const typeCompute b ) -> typeCompute
{
    return (a > b) ? a : b;
}

INLINE_OPERATOR_PREFIX
bool myIsNan(double a)
{
    return a != a;
}

INLINE_OPERATOR_PREFIX
bool myIsNan(float a)
{
    return a != a;
}

template<>
INLINE_OPERATOR_PREFIX
float lwMax( const float a, const float b )
{
    if (myIsNan(a)) { return a; }
    if (myIsNan(b)) { return b; }
    return (a > b) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
double lwMax( const double a, const double b )
{
    if (myIsNan(a)) { return a; }
    if (myIsNan(b)) { return b; }
    return (a > b) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwMax( const half a, const half b )
{
    if (myIsNan(lwGet<float>(a))) { return a; }
    if (myIsNan(lwGet<float>(b))) { return b; }
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return __hge( a, b ) ? a : b;
#else
    return lwGet<half>( lwMax( lwGet<float>( a ), lwGet<float>( b ) ) );
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwMax( const BFloat16 a, const BFloat16 b )
{
    if (myIsNan(lwGet<float>(a))) { return a; }
    if (myIsNan(lwGet<float>(b))) { return b; }
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return __hge( a, b ) ? a : b;
#else
    return lwGet<BFloat16>( lwMax( lwGet<float>( a ), lwGet<float>( b ) ) );
#endif
}
#endif

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Return the minimum value between a and b
 * \param[in] a input in typeCompute
 * \param[in] b input in typeCompute
 * \param[in] typeCompute the input value datatype
 * \return the minimum value between a and b
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
typeCompute lwMin( const typeCompute a, const typeCompute b )
{
    return (a < b) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
float lwMin( const float a, const float b )
{
    if (myIsNan(a)) { return a; }
    if (myIsNan(b)) { return b; }
    return (a < b) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
double lwMin( const double a, const double b )
{
    if (myIsNan(a)) { return a; }
    if (myIsNan(b)) { return b; }
    return (a < b) ? a : b;
}

template<>
INLINE_OPERATOR_PREFIX
half lwMin( const half a, const half b )
{
    if (myIsNan(lwGet<float>(a))) { return a; }
    if (myIsNan(lwGet<float>(b))) { return b; }
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 600
    return __hlt( a, b ) ? a : b;
#else
    return lwGet<half>( lwMin( lwGet<float>( a ), lwGet<float>( b ) ) );
#endif
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
BFloat16 lwMin( const BFloat16 a, const BFloat16 b )
{
    if (myIsNan(lwGet<float>(a))) { return a; }
    if (myIsNan(lwGet<float>(b))) { return b; }
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    return __hlt( a, b ) ? a : b;
#else
    return lwGet<BFloat16>( lwMin( lwGet<float>( a ), lwGet<float>( b ) ) );
#endif
}
#endif

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Return absolute value in double
 * \param[in] a input in typeCompute
 * \param[in] typeCompute the input value datatype
 * \return the absolute value in double
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
double lw2Norm( const typeCompute a ) { return (a >= lwGet<typeCompute>(0)) ? static_cast<double>(a) : -(static_cast<double>(a)) ; }
template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( const half a ) { return lw2Norm( lwGet<float>( a ) ); }
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( const BFloat16 a ) { return lw2Norm( lwGet<float>( a ) ); }
#endif
template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( const uint32_t a ) { return static_cast<double>(a); }
template<>
INLINE_OPERATOR_PREFIX
double lw2Norm( const uint8_t a ) { return static_cast<double>(a); }

/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Conduct an unary operation on a value.
 * \param[in] x the input value
 * \param[in] typeOut the input value datatype
 * \param[in] op the unary operation
 * \param[in] ctx the activation context
 * \return the result of the unary operation on the input value
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeOut>
INLINE_OPERATOR_PREFIX
auto lwtensorUnaryOp(
        const typeOut x, 
        const lwtensorOperator_t op, 
        const ElementwiseParameters::ActivationContext *ctx = nullptr) -> typename std::enable_if<!std::is_integral<typeOut>::value, typeOut>::type
{
    switch (op)
    {
        case lwtensorOperator_t::LWTENSOR_OP_IDENTITY: return x;
        case lwtensorOperator_t::LWTENSOR_OP_SQRT: return lwSqrt(x);
        case lwtensorOperator_t::LWTENSOR_OP_RCP: return lwRcp(x);
        case lwtensorOperator_t::LWTENSOR_OP_CONJ: return x;
        case lwtensorOperator_t::LWTENSOR_OP_RELU: return lwRelu(x);
        // case lwtensorOperator_t::LWTENSOR_OP_CLIP: return lwClip(x, ctx->clip.lower, ctx->clip.upper);
        // case lwtensorOperator_t::LWTENSOR_OP_THRESHOLDED_RELU: return lwThresholdedRelu(x, ctx->thresholdedRelu.threshold);
        case lwtensorOperator_t::LWTENSOR_OP_SIGMOID: return lwSigmoid(x);
        case lwtensorOperator_t::LWTENSOR_OP_TANH: return lwTanh(x);
        // case lwtensorOperator_t::LWTENSOR_OP_ELU: return lwElu(x, ctx->elu.alpha);
        // case lwtensorOperator_t::LWTENSOR_OP_LEAKY_RELU: return lwLeakyRelu(x, ctx->leakyRelu.k);
        // case lwtensorOperator_t::LWTENSOR_OP_SOFT_PLUS: return lwSoftPlus(x, ctx->softPlus.inScale, ctx->softPlus.outScale, ctx->softPlus.approximateThreshold);
        // case lwtensorOperator_t::LWTENSOR_OP_SOFT_SIGN: return lwSoftSign(x);
        // case lwtensorOperator_t::LWTENSOR_OP_SELU: return lwSelu(x, ctx->selu.eluAlpha, ctx->selu.outScale);
        // case lwtensorOperator_t::LWTENSOR_OP_HARD_SIGMOID: return lwHardSigmoid(x, ctx->hardSigmoid.slope, ctx->hardSigmoid.shift);
        // case lwtensorOperator_t::LWTENSOR_OP_SCALED_TANH: return lwScaledTanh(x, ctx->scaledTanh.inScale, ctx->scaledTanh.outScale);
                                      // New operators for TRT unary runner.
        case lwtensorOperator_t::LWTENSOR_OP_EXP: return lwExp(x);
        case lwtensorOperator_t::LWTENSOR_OP_LOG: return lwLn(x);
        case lwtensorOperator_t::LWTENSOR_OP_ABS: return lwAbs(x);
        case lwtensorOperator_t::LWTENSOR_OP_NEG: return lwNeg(x);
        case lwtensorOperator_t::LWTENSOR_OP_SIN: return lwSin(x);
        case lwtensorOperator_t::LWTENSOR_OP_COS: return lwCos(x);
        case lwtensorOperator_t::LWTENSOR_OP_TAN: return lwTan(x);
        case lwtensorOperator_t::LWTENSOR_OP_SINH: return lwSinh(x);
        case lwtensorOperator_t::LWTENSOR_OP_COSH: return lwCosh(x);
        case lwtensorOperator_t::LWTENSOR_OP_ASIN: return lwAsin(x);
        case lwtensorOperator_t::LWTENSOR_OP_ACOS: return lwAcos(x);
        case lwtensorOperator_t::LWTENSOR_OP_ATAN: return lwAtan(x);
        case lwtensorOperator_t::LWTENSOR_OP_ASINH: return lwAsinh(x);
        case lwtensorOperator_t::LWTENSOR_OP_ACOSH: return lwAcosh(x);
        case lwtensorOperator_t::LWTENSOR_OP_ATANH: return lwAtanh(x);
        case lwtensorOperator_t::LWTENSOR_OP_CEIL: return lwCeil(x);
        case lwtensorOperator_t::LWTENSOR_OP_FLOOR: return lwFloor(x);
        default: return x;
    }
}

template<typename typeOut>
INLINE_OPERATOR_PREFIX
auto lwtensorUnaryOp( 
        const typeOut x, 
        const lwtensorOperator_t op, 
        const ElementwiseParameters::ActivationContext *ctx = nullptr) -> typename std::enable_if<std::is_integral<typeOut>::value, typeOut>::type
{
    switch (op)
    {
        case lwtensorOperator_t::LWTENSOR_OP_IDENTITY: return x;
        case lwtensorOperator_t::LWTENSOR_OP_RELU: return lwRelu(x);
        // case lwtensorOperator_t::LWTENSOR_OP_CLIP: return lwClip(x, ctx->clip.lower, ctx->clip.upper);
        // case lwtensorOperator_t::LWTENSOR_OP_THRESHOLDED_RELU: return lwThresholdedRelu(x, ctx->thresholdedRelu.threshold);
        // case lwtensorOperator_t::LWTENSOR_OP_LEAKY_RELU: return lwLeakyRelu(x, ctx->leakyRelu.k);
        default: return x;
    }
}

template<>
INLINE_OPERATOR_PREFIX
lwComplex
lwtensorUnaryOp(
        const lwComplex x, 
        const lwtensorOperator_t op, 
        const ElementwiseParameters::ActivationContext *ctx)
{
    switch (op)
    {
        case lwtensorOperator_t::LWTENSOR_OP_CONJ:
            return lwConjf(x);
        default:
            return x;
    }
}


/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief First apply the input quantization, then apply the activation, and finally apply the output quantization.
 * \param[in] x the input value
 * \param[in] scalar quntization scalars
 * \param[in] typeOut the input value datatype
 * \param[in] op the unary operation
 * \param[in] ctx the activation context
 * \return the result of the unary operation on the input value
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeCompute>
INLINE_OPERATOR_PREFIX
auto activationWithQuantization(
        const typeCompute x,
        const typeCompute scalar,
        const ElementwiseParameters::ActivationContext *ctx,
        const lwtensorOperator_t op) -> typeCompute;

template<>
INLINE_OPERATOR_PREFIX
lwComplex activationWithQuantization(
        const lwComplex x,
        const lwComplex scalar,
        const ElementwiseParameters::ActivationContext *ctx,
        const lwtensorOperator_t op)
{
    return make_lwComplex(lwtensorUnaryOp(x.x * scalar.x, op, ctx) * scalar.y, 0.0f);
}




/**
 * \ingroup runtimeSupportFuncPLC3
 * \brief Conduct an binary operation on x and y
   \param[in] x the input value
   \param[in] y the input value
 * \param[in] typeOut the input value datatype
 * \param[in] op the binary operation
 * \param[in] ctx the activation context
 * \param[in] opUnary the unary operation that is optional to perform
 * \return the result of the unary operation on the input value
 * \behavior blocking, reentrant, and thread safe
 */
template<typename typeOut>
INLINE_OPERATOR_PREFIX
auto lwtensorBinaryOp(
        const typeOut x,
        const typeOut y,
        const lwtensorOperator_t op,
        const ElementwiseParameters::ActivationContext *ctx = nullptr,
        const lwtensorOperator_t unaryOp = lwtensorOperator_t::LWTENSOR_OP_IDENTITY) -> typeOut
{
   if (op == lwtensorOperator_t::LWTENSOR_OP_ADD)
   {
       return lwAdd( x, y );
   }
   else if (op == lwtensorOperator_t::LWTENSOR_OP_MUL)
   {
       return lwMul( x, y );
   }
   else if (op == lwtensorOperator_t::LWTENSOR_OP_MAX)
   {
       return lwMax( x, y );
   }
   else
   {
       return lwMin( x, y );
   }
}


template<>
INLINE_OPERATOR_PREFIX
lwComplex lwtensorBinaryOp(
        const lwComplex x,
        const lwComplex y,
        const lwtensorOperator_t op,
        const ElementwiseParameters::ActivationContext *ctx,
        const lwtensorOperator_t unaryOp)
{
   if (op == lwtensorOperator_t::LWTENSOR_OP_ADD)
   {
       return lwAdd( x, y );
   }
   else if (op == lwtensorOperator_t::LWTENSOR_OP_MUL)
   {
       return lwMul( x, y );
   }
   else// if (op == lwtensorOperator_t::LWTENSOR_OP_ACTIVATION_WITH_QUANTIZATION)
   {
       return activationWithQuantization(x, y, ctx, unaryOp);
   }
}


} /* end of namespace */
#pragma GCC diagnostic pop
#endif /* define LWTENSOR_OPERATORSPLC3_H */
