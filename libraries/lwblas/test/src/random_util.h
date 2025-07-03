#pragma once

#include <lwda_runtime.h>
#include <lwda_fp16.h>

#include <lwtensor/internal/defines.h>

namespace random_util
{

/// computes random numbers via:
/// x_i+1 = (a*x_i)mod m;
/// m must be prim
/// a must be a primative root modulo m
///
/// to prevent overflow we use the algorithm by Schrage
int32_t linearCongruentialGenerator(int32_t &seed)
{
    constexpr int32_t a = 16807;
    constexpr int32_t m = 2147483647;

    constexpr int32_t q = 127773;
    constexpr int32_t r = 2836;

    int32_t tmp1 = a*(seed % q);
    int32_t tmp2 = -r*(seed / q);
    seed = tmp1 + tmp2;
    if( seed < 0 )
        seed += m;

    return seed;
}

/** Generate random number in uniform [a, b] distribution. */
template<typename typeCompute>
typeCompute UniformRandomNumber( double a, double b, int32_t &seed )
{
   /** Generate uniform [ 0, 1 ] first. */
   auto u01 = (double)( linearCongruentialGenerator(seed) % 1000000 ) / 1000000;
   /** Scale and shift. */
   return ( b - a ) * u01 + a;
}

/** Specialization for half. */
template<>
half UniformRandomNumber( double a, double b, int32_t &seed )
{
    return LWTENSOR_NAMESPACE::lwGet<half>( UniformRandomNumber<double>( a, b, seed ) );
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
/** Specialization for half. */
template<>
LWTENSOR_NAMESPACE::BFloat16 UniformRandomNumber( double a, double b, int32_t &seed )
{
    return LWTENSOR_NAMESPACE::lwGet<LWTENSOR_NAMESPACE::BFloat16>( UniformRandomNumber<double>( a, b, seed ) );
}
#endif

/** Specialization for single complex. */
template<>
lwComplex UniformRandomNumber( double a, double b, int32_t &seed )
{
    lwComplex x;
    x.x = UniformRandomNumber<double>( a, b, seed );
    b = a + sqrt(b*b - x.x * x.x);
    x.y = UniformRandomNumber<double>( a, b, seed );
    return x;
}

/** Specialization for double complex. */
template<>
lwDoubleComplex UniformRandomNumber( double a, double b, int32_t &seed )
{
    lwDoubleComplex x;
    x.x = UniformRandomNumber<double>( a, b, seed );
    b = a + sqrt(b*b - x.x * x.x);
    x.y = UniformRandomNumber<double>( a, b, seed );
    return x;
}

}  // namespace random_util
