/******************************************************************************
 * Copyright 2018 LWPU Corporation. All rights reserved.
 *****************************************************************************/

static __inline__ __device__ __half __saturate( __half x )
{
#if __LWDA_ARCH__ >= 530
    return x > hone() ? hone() : x;
#else
    return __saturate( float( x ) );
#endif
}

static __inline__ __device__ float __saturate( float x )
{
    return __saturatef( x );
}
