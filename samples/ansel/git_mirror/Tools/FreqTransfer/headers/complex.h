#pragma once

#ifdef __LWDACC__
typedef float2 fComplex;
#define LWDA_FUNC	__device__
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#define LWDA_FUNC
#endif

LWDA_FUNC
fComplex mul(double k, const fComplex & c)
{
    fComplex result;
    result.x = (float)(k * c.x);
    result.y = (float)(k * c.y);
    return result;
}

LWDA_FUNC
float getModulus(const fComplex & c)
{
    return sqrtf(c.x*c.x + c.y*c.y);
}

LWDA_FUNC
fComplex complexLerp(const fComplex & Ca, const fComplex & Cb, float coeff)
{
    fComplex C;
    C.x = Ca.x * (1.0f - coeff) + Cb.x * coeff;
    C.y = Ca.y * (1.0f - coeff) + Cb.y * coeff;
    return C;
}

#undef LWDA_FUNC
