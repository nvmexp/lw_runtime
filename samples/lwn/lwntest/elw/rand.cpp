/*
 * Copyright (c) 2009 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "cmdline.h"

static unsigned int seed = 1;
static unsigned int alternateSeed = 1;
static int (*randFunc)(void) = NULL;

void lwSetRandFunc(int (*f)(void))
{
    randFunc = f;
}

unsigned int lwGetSeed(void)
{
    if (randFunc == lwRand) {
        return seed;
    } else if (randFunc == lwAlternateRand) {
        return alternateSeed;
    }
    return 0;
}

int lwRand(void)
{
    seed = seed * 1103515245 + 12345;
    return seed >> 16;
}

int lwAlternateRand(void)
{
    alternateSeed = alternateSeed * 1106515245 + 13245;
    return alternateSeed >> 16;
}

void lwSRand(unsigned int n)
{
    if (randFunc == lwRand) {
        seed = n;
    } else if (randFunc == lwAlternateRand) {
        alternateSeed = n;
    } else {
        srand(0);
    }
}

int
lwRandNumber(void)
{
    return (*randFunc)();
}

/*****************************************************************
* lwBitRand() -
*****************************************************************/
unsigned int lwBitRand(int b)
{
    unsigned int mask;

    if (b == 32)
        mask = ~0;
    else mask = (1<<b)-1;

    mask &= ((*randFunc)()<<16) | (*randFunc)();

    return(mask);
}

/*****************************************************************
* lwIntRand() -
*****************************************************************/
int lwIntRand(int min,int max)
{
   int tmp,i;
   unsigned int x,delta;

   if (min > max) {
     tmp = min;
     min = max;
     max = tmp;
   }

   delta = max-min;

   i = 0;
   while (delta) {
     i++;
     delta >>= 1;
   }

   delta = max-min;

   do { x = lwBitRand(i); }
   while (x > delta);

   return(min+x);
}

/*****************************************************************
* lwFloatRand() -
*****************************************************************/
float lwFloatRand(float min,float max)
{
   unsigned int mant;
   float f;

   mant = (*randFunc)() << 8; 
   mant |= 0x3f800000;

   f = uint32_as_float(mant);
   f -= 1.0f;
   f *= max-min;
   f += min;

   return(f);
}

void
lwRandColor(float rgba[4])
{
    rgba[0] = lwFloatRand(0.0,1.0);
    rgba[1] = lwFloatRand(0.0,1.0);
    rgba[2] = lwFloatRand(0.0,1.0);
    rgba[3] = lwFloatRand(0.0,1.0);
}
