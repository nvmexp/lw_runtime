/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : ptxDescriptorReader.h
 *
 *  Description              :
 *
 */

#ifndef __DESCRIPTOR_HANDLER_H
#define __DESCRIPTOR_HANDLER_H

#include "stdLocal.h"

#ifdef __cplusplus
extern "C" {
#endif

String obtainExtDescBuffer(cString extDescFileName, cString extDescAsString, int len);

#ifdef __cplusplus
}
#endif

#endif // __DESCRIPTOR_HANDLER_H
