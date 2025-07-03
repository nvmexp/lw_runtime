/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "gl_pointers.h"

// Define function pointer variable

#define GLPROC(_type,_name) _type _name
#include "gl_pointers.inc"
#undef GLPROC

// Call wglGetProcAddress for each function

void InitGLPointers()
{
  #define GLPROC(_type,_name)   _name = (_type) wglGetProcAddress(#_name)
  #include "gl_pointers.inc"
  #undef GLPROC
}
