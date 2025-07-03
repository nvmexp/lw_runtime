/*
 * Copyright (c) 2005 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __FLOAT_UTIL_H__
#define __FLOAT_UTIL_H__

#include "lwogtypes.h"

typedef uint32_t LWu6e5;
typedef uint32_t LWu5e5;
typedef uint32_t LWs10e5;

uint32_t lwU6E5toUI32(LWu6e5 f);
float    lwU6E5toF32(LWu6e5 f);
LWu6e5   lwUI32toU6E5(uint32_t ui);
LWu6e5   lwF32toU6E5(float f);

uint32_t lwU5E5toUI32(LWu5e5 f);
float    lwU5E5toF32(LWu5e5 f);
LWu5e5   lwUI32toU5E5(uint32_t ui);
LWu5e5   lwF32toU5E5(float f);

uint32_t lwS10E5toUI32(LWs10e5 f);
float    lwS10E5toF32(LWs10e5 f);
LWs10e5  lwUI32toS10E5(uint32_t ui);
LWs10e5  lwF32toS10E5(float f);

uint32_t lwColwertFloat3ToR11fG11fB10f(const float rgb[3]);
void lwColwertR11fG11fB10fToFloat3(uint32_t v, float retval[3]);

uint32_t lwColwertFloat3ToRGB9E5(const float rgb[3]);
void lwColwertRGB9E5ToFloat3(uint32_t v, float retval[3]);

float srgbToLinear(float c);

#endif // __FLOAT_UTIL_H__
