/*
 * Copyright (c) 1993-2014, LWPU CORPORATION. All rights reserved.
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


#ifndef _cl917a_h_
#define _cl917a_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW917A_LWRSOR_CHANNEL_PIO                                               (0x0000917A)

typedef volatile struct {
    LwV32 Reserved00[0x2];
    LwV32 Free;                                                                 // 0x00000008 - 0x0000000B
    LwV32 Reserved01[0x1D];
    LwV32 Update;                                                               // 0x00000080 - 0x00000083
    LwV32 SetLwrsorHotSpotPointsOut[2];                                         // 0x00000084 - 0x0000008B
    LwV32 Reserved02[0x3DD];
} GK104DispLwrsorControlPio;

#define LW917A_FREE                                                             (0x00000008)
#define LW917A_FREE_COUNT                                                       5:0
#define LW917A_UPDATE                                                           (0x00000080)
#define LW917A_UPDATE_INTERLOCK_WITH_CORE                                       0:0
#define LW917A_UPDATE_INTERLOCK_WITH_CORE_DISABLE                               (0x00000000)
#define LW917A_UPDATE_INTERLOCK_WITH_CORE_ENABLE                                (0x00000001)
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT(b)                                (0x00000084 + (b)*0x00000004)
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT_X                                 15:0
#define LW917A_SET_LWRSOR_HOT_SPOT_POINTS_OUT_Y                                 31:16

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl917a_h

