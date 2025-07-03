/*
 * Copyright (c) 2005 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */


// utility routines for dealing with packed/small floating-point representations

#include "ogtest.h"

#include "float_util.h"

#include "floatu10.h"
#include "floatu11.h"
#include "float16.h"

#define R_MASK ((1<<11)-1)
#define G_MASK ((1<<11)-1)
#define B_MASK ((1<<10)-1)

#define R_SHIFT 0
#define G_SHIFT 11
#define B_SHIFT 22

uint32_t lwColwertFloat3ToR11fG11fB10f(const float rgb[3])
{
  uint32_t r = lwF32toU6E5(rgb[0]);
  uint32_t g = lwF32toU6E5(rgb[1]);
  uint32_t b = lwF32toU5E5(rgb[2]);
  uint32_t retval;

  assert(0 == (r & ~R_MASK));
  assert(0 == (g & ~G_MASK));
  assert(0 == (b & ~B_MASK));

  retval  = r << R_SHIFT;
  retval |= g << G_SHIFT;
  retval |= b << B_SHIFT;

  return retval;
}

void lwColwertR11fG11fB10fToFloat3(uint32_t v, float retval[3])
{
  uint32_t r = (v >> R_SHIFT) & R_MASK;
  uint32_t g = (v >> G_SHIFT) & G_MASK;
  uint32_t b = (v >> B_SHIFT) & B_MASK;

  retval[0] = lwU6E5toF32(r);
  retval[1] = lwU6E5toF32(g);
  retval[2] = lwU5E5toF32(b);
}

#define FP32_MANTISSA_BITS            23
#define FP32_EXPONENT_BITS            8

#define FP32_EXPONENT_MASK            ((1<<FP32_EXPONENT_BITS)-1)
#define FP32_MANTISSA_MASK            ((1<<FP32_MANTISSA_BITS)-1)

#define FP32_EXPONENT_BIAS            127

#define RGB9E5_EXPONENT_BITS          5
#define RGB9E5_MANTISSA_BITS          9

#define RGB9E5_EXPONENT_MASK          ((1<<RGB9E5_EXPONENT_BITS)-1)
#define RGB9E5_MANTISSA_MASK          ((1<<RGB9E5_MANTISSA_BITS)-1)

#define RGB9E5_EXP_BIAS              15
#define RGB9E5_MAX_VALID_BIASED_EXP  31
#define MAX_RGB9E5_EXP               (RGB9E5_MAX_VALID_BIASED_EXP - RGB9E5_EXP_BIAS)
#define RGB9E5_MANTISSA_VALUES       (1<<RGB9E5_MANTISSA_BITS)
#define MAX_RGB9E5_MANTISSA          (RGB9E5_MANTISSA_VALUES-1)
#define MAX_RGB9E5                   (((float)MAX_RGB9E5_MANTISSA)/RGB9E5_MANTISSA_VALUES * (1<<MAX_RGB9E5_EXP))
#define EPSILON_RGB9E5               ((1.0/RGB9E5_MANTISSA_VALUES) / (1<<RGB9E5_EXP_BIAS))

/* For quick float->int colwersions */
#define __GL_F0_DOT_0           12582912.0f

/* Fast colwersion for values between 0.0 and 65535.0 */
#define __GL_QUICK_FLOAT2UINT(fval, fTmp)                                   \
    ((fTmp = (fval) + __GL_F0_DOT_0), (float_as_uint32(fTmp) & 0xFFFF))

// This might be faster to re-implement as integer instructions.
static float ClampRange_for_rgb9e5(float x)
{
  if (x > 0.0) {
    if (x >= MAX_RGB9E5) {
      return MAX_RGB9E5;
    } else {
      return x;
    }
  } else {
    // NaN gets here too!
    return 0.0;
  }
}

static float MaxOf3(float x, float y, float z)
{
  if (x > y) {
    if (x > z) {
      return x;
    } else {
      return z;
    }
  } else {
    if (y > z) {
      return y;
    } else {
      return z;
    }
  }
}

int FloorLog2(float x)
{
  unsigned int f = float_as_uint32(x);
  unsigned int expbits = ((f >> FP32_MANTISSA_BITS) & FP32_EXPONENT_MASK);
  int exponent = expbits - FP32_EXPONENT_BIAS;
  const unsigned int top9manbits = RGB9E5_MANTISSA_MASK << (FP32_MANTISSA_BITS-RGB9E5_MANTISSA_BITS);
  unsigned int manbits = f & top9manbits;

  // But if within an rgb9e5 half mantissa LSB, use the higher exponent.
  if (manbits == top9manbits) {
    exponent++;
  }

  return exponent;
}

static int Max(int x, int y)
{
  if (x > y) {
    return x;
  } else {
    return y;
  }
}

#if 1  // Your can implement FASTPOW2 either with a table or function.

#define POW_TABLE_ENTRY(i) ((i - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS + FP32_EXPONENT_BIAS) << FP32_MANTISSA_BITS)
static const unsigned int powTable[1<<RGB9E5_EXPONENT_BITS] = {
  POW_TABLE_ENTRY(0),
  POW_TABLE_ENTRY(1),
  POW_TABLE_ENTRY(2),
  POW_TABLE_ENTRY(3),
  POW_TABLE_ENTRY(4),
  POW_TABLE_ENTRY(5),
  POW_TABLE_ENTRY(6),
  POW_TABLE_ENTRY(7),
  POW_TABLE_ENTRY(8),
  POW_TABLE_ENTRY(9),
  POW_TABLE_ENTRY(10),
  POW_TABLE_ENTRY(11),
  POW_TABLE_ENTRY(12),
  POW_TABLE_ENTRY(13),
  POW_TABLE_ENTRY(14),
  POW_TABLE_ENTRY(15),
  POW_TABLE_ENTRY(16),
  POW_TABLE_ENTRY(17),
  POW_TABLE_ENTRY(18),
  POW_TABLE_ENTRY(19),
  POW_TABLE_ENTRY(20),
  POW_TABLE_ENTRY(21),
  POW_TABLE_ENTRY(22),
  POW_TABLE_ENTRY(23),
  POW_TABLE_ENTRY(24),
  POW_TABLE_ENTRY(25),
  POW_TABLE_ENTRY(26),
  POW_TABLE_ENTRY(27),
  POW_TABLE_ENTRY(28),
  POW_TABLE_ENTRY(29),
  POW_TABLE_ENTRY(30),
  POW_TABLE_ENTRY(31)
};
#define FASTPOW2(i) uint32_as_float(powTable[i])

#else

static float fastpow2(int i)
{
  unsigned int v = ((i - RGB9E5_EXP_BIAS + FP32_EXPONENT_BIAS - RGB9E5_MANTISSA_BITS) << FP32_MANTISSA_BITS);

  return *((const float*)&v);
}
#define FASTPOW2(i) fastpow2(i)

#endif

uint32_t lwColwertFloat3ToRGB9E5(const float rgb[3])
{
  uint32_t retval;
  float maxrgb;
  int rm, gm, bm;
  float rc, gc, bc;
  int exp_shared;
  float divisor;
  float fTmp;

  rc = ClampRange_for_rgb9e5(rgb[0]);
  gc = ClampRange_for_rgb9e5(rgb[1]);
  bc = ClampRange_for_rgb9e5(rgb[2]);

  maxrgb = MaxOf3(rc, gc, bc);
  exp_shared = Max(-RGB9E5_EXP_BIAS-1, FloorLog2(maxrgb)) + 1 + RGB9E5_EXP_BIAS;
  assert(exp_shared <= RGB9E5_MAX_VALID_BIASED_EXP);
  assert(exp_shared >= 0);
  divisor = FASTPOW2(exp_shared);
  assert(divisor == (float) pow(2, exp_shared - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS));

  rm = __GL_QUICK_FLOAT2UINT(rc / divisor, fTmp);
  gm = __GL_QUICK_FLOAT2UINT(gc / divisor, fTmp);
  bm = __GL_QUICK_FLOAT2UINT(bc / divisor, fTmp);

  assert(rm <= MAX_RGB9E5_MANTISSA);
  assert(gm <= MAX_RGB9E5_MANTISSA);
  assert(bm <= MAX_RGB9E5_MANTISSA);
  assert(rm >= 0);
  assert(gm >= 0);
  assert(bm >= 0);

  retval  = rm;
  retval |= gm         << (  RGB9E5_MANTISSA_BITS);
  retval |= bm         << (2*RGB9E5_MANTISSA_BITS);
  retval |= exp_shared << (3*RGB9E5_MANTISSA_BITS);

  return retval;
}

void lwColwertRGB9E5ToFloat3(uint32_t v, float retval[3])
{
  int exp_shared = (v >> (3*RGB9E5_MANTISSA_BITS));
  float scale = FASTPOW2(exp_shared);

  assert(scale == (float) pow(2, exp_shared - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS));

  retval[0] = (RGB9E5_MANTISSA_MASK & (v                            )) * scale;
  retval[1] = (RGB9E5_MANTISSA_MASK & (v >> (  RGB9E5_MANTISSA_BITS))) * scale;
  retval[2] = (RGB9E5_MANTISSA_MASK & (v >> (2*RGB9E5_MANTISSA_BITS))) * scale;
}

float srgbToLinear(float c)
{
    if (c <= 0.004045f) {
        return c / 12.92f;
    }
    return pow((c + 0.055f) / 1.055f, 2.4f);
}
