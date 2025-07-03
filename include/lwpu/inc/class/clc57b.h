/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/



#ifndef _clC57b_h_
#define _clC57b_h_


#ifdef __cplusplus
extern "C" {
#endif

#define LWC57B_WINDOW_IMM_CHANNEL_DMA                                           (0x0000C57B)

typedef volatile struct _clc57b_tag0 {
    LwV32 Put;                                                                  // 0x00000000 - 0x00000003
    LwV32 Get;                                                                  // 0x00000004 - 0x00000007
    LwV32 Reserved00[0x7E];
    LwV32 Update;                                                               // 0x00000200 - 0x00000203
    LwV32 Reserved01[0x1];
    LwV32 SetPointOut[2];                                                       // 0x00000208 - 0x0000020F
    LwV32 Reserved02[0x37C];
} LWC57BDispControlDma;


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
#define LWC57B_UPDATE_RELEASE_ELV                                               0:0
#define LWC57B_UPDATE_RELEASE_ELV_FALSE                                         (0x00000000)
#define LWC57B_UPDATE_RELEASE_ELV_TRUE                                          (0x00000001)
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
