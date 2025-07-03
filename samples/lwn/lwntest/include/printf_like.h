/*
 * Copyright (c) 2007 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */


/* printf_like.h - support to declare printf-like functions for gcc format warnings */

#ifndef __printflike
# if __GNUC__ > 2 || __GNUC__ == 2 && __GNUC_MINOR__ >= 7
// fmtarg and firstvararg are start from one, 1==1st argument
#  define __printflike(fmtarg, firstvararg) \
          __attribute__((__format__ (__printf__, fmtarg, firstvararg)))
# else
#  define __printflike(fmtarg, firstvararg)
# endif
#endif

