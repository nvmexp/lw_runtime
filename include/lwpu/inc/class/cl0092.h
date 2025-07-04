/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef SDK_CL0092_H
#define SDK_CL0092_H

#include "lwtypes.h"

/*
 * This RgLineCallback class allows RM clients to register/unregister the RG line callback functions.
 *
 * Must be allocated with kernel access rights.
 *
 * Allocation params:
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the LW04_DISPLAY_COMMON parent device to which the
 *     operation should be directed.
 *   head
 *     This parameter specifies the head for which the callback is to be egistered/unregistered. This value must be
 *     less than the maximum number of heads supported by the GPU subdevice.
 *   rgLineNum
 *     This indicates the RG scanout line number on which the callback will be exelwted.
 *       1/ Client should set the proper RG line number based on mode in which the display head is running and
 *          subsequent possible modeset that may affect the line number.
 *       2/ Client is expected to clear/set the interrupts around modesets or power-transitions (like s3/hibernation).
 *       3/ Client should make sure that this param does not exceed the raster settings.
 *   pCallbkFn
 *     Pointer to callback function. Cannot be NULL.
 *   pCallbkParams
 *     Pointer to the ctrl call param struct.
 */

#define LW0092_RG_LINE_CALLBACK 0x0092

typedef void (*LW0092_REGISTER_RG_LINE_CALLBACK_FN)(LwU32 rgIntrLine, void *param1, LwBool bIsIrqlIsr);

typedef struct
{
    LwU32  subDeviceInstance;
    LwU32  head;
    LwU32  rgLineNum;

    LW0092_REGISTER_RG_LINE_CALLBACK_FN pCallbkFn;

    void  *pCallbkParams;
} LW0092_RG_LINE_CALLBACK_ALLOCATION_PARAMETERS;

#endif // SDK_CL0092_H
