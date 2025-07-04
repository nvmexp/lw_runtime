/*
 * Copyright (c) 1993-2020, LWPU CORPORATION. All rights reserved.
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

#ifndef _clC57b_h_
#define _clC57b_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LWC57B_WINDOW_IMM_CHANNEL_DMA                                           (0x0000C57B)

// dma opcode instructions
#define LWC57B_DMA                                                                     
#define LWC57B_DMA_OPCODE                                                        31:29 
#define LWC57B_DMA_OPCODE_METHOD                                            0x00000000 
#define LWC57B_DMA_OPCODE_JUMP                                              0x00000001 
#define LWC57B_DMA_OPCODE_NONINC_METHOD                                     0x00000002 
#define LWC57B_DMA_OPCODE_SET_SUBDEVICE_MASK                                0x00000003 
#define LWC57B_DMA_METHOD_COUNT                                                  27:18 
#define LWC57B_DMA_METHOD_OFFSET                                                  13:2 
#define LWC57B_DMA_DATA                                                           31:0 
#define LWC57B_DMA_DATA_NOP                                                 0x00000000 
#define LWC57B_DMA_JUMP_OFFSET                                                    11:2 
#define LWC57B_DMA_SET_SUBDEVICE_MASK_VALUE                                       11:0 

// class methods
#define LWC57B_PUT                                                              (0x00000000)
#define LWC57B_PUT_PTR                                                          9:0
#define LWC57B_GET                                                              (0x00000004)
#define LWC57B_GET_PTR                                                          9:0
#define LWC57B_UPDATE                                                           (0x00000200)
#define LWC57B_UPDATE_INTERLOCK_WITH_WINDOW                                     1:1
#define LWC57B_UPDATE_INTERLOCK_WITH_WINDOW_DISABLE                             (0x00000000)
#define LWC57B_UPDATE_INTERLOCK_WITH_WINDOW_ENABLE                              (0x00000001)
#define LWC57B_SET_POINT_OUT(b)                                                 (0x00000208 + (b)*0x00000004)
#define LWC57B_SET_POINT_OUT_X                                                  15:0
#define LWC57B_SET_POINT_OUT_Y                                                  31:16

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clC57b_h
