/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// dkumar@lwpu.com - Mar 14, 2007
// disp_0201.h - display includes. This file is not intended to be used without
// a preceding #include "xxxx/dev_disp.h" since it refers to LW_PDISP_*
// 
//*****************************************************

#ifndef _DISP_DISP0201_H_
#define _DISP_DISP0201_H_


//cl917E has stereo overlay
#define LW917E_OVLY_SURFACE_PER_HEAD            2

//refer to cl907D/C/E for surface and iso ctx dma info
#define LW917D_CORE_SURFACE_PER_HEAD            1
#define LW917C_BASE_SURFACE_PER_HEAD            2

// CORE Channel State 
static const char * const dCoreState[] = {
    "DEALLOC",
    "DEALLOC_LIMBO",
    "LIMBO1",
    "LIMBO2",
    "EFIINIT",
    "EFI",
    "VBIOSINIT",
    "VBIOSOPER",
    "UNCONNECTED",
    "INITIALIZE",
    "IDLE",
    "WRTIDLE",
    "EMPTY",
    "FLUSHED",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

// BASE Channel State
static const char * const dBaseState[] = {
    "DEALLOC",
    "UNCONNECTED",
    "INITIALIZE",
    "IDLE",
    "WRTIDLE",
    "EMPTY",
    "FLUSHED",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

// OVERLAY Channel State
static const char * const dOvlyState[] = {
    "DEALLOC",
    "UNCONNECTED",
    "INITIALIZE",
    "IDLE",
    "WRTIDLE",
    "EMPTY",
    "FLUSHED",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

// OVERLAY IMMEDIATE Channel State
static const char * const dOvimState[] = {
    "DEALLOC",
    "IDLE",
    "BUSY",
};

// CURSOR Channel State
static const char * const dLwrsState[] = {
    "DEALLOC",
    "IDLE",
    "BUSY",
};

#define CHNST_SIZE(x) (sizeof(x) / sizeof(char *))

#define LW_DISP_CHAN_DESC(name,type,state,hn,cap) \
        { name, hn, LW_PDISP_CHNCTL_ ## type ## __SIZE_1, \
          DRF_EXTENT(LW_PDISP_CHNCTL_ ## type ## _STATE), \
          DRF_BASE(LW_PDISP_CHNCTL_ ## type ## _STATE), \
          CHNST_SIZE(state), LW_PDISP_CHNCTL_ ## type(hn), cap,\
          CHNTYPE_ ## type},
          ///CHNST_SIZE(state), 0x4500000 },

#define GET_BITS(num,h,l) (((num) >> l) & (0xffffffff >> (31 - h + l))) 

static const char * const * dChanStateWords_v02_01[] = {
    dCoreState,

    dBaseState,
    dBaseState,
    dBaseState,
    dBaseState,

    dOvlyState,
    dOvlyState,
    dOvlyState,
    dOvlyState,

    dOvimState,
    dOvimState,
    dOvimState,
    dOvimState,

    dLwrsState,
    dLwrsState,
    dLwrsState,
    dLwrsState
};

#define DCHN_GET_DESC_V02_01(x,y) (dChanStateWords_v02_01[x][y])


// It may be better to use LW_PDISP_507X_CHN_* macros, 
// but this file is very arch specific, so the following
// must be changed in a great deal..
static ChanDesc_t dispChanState_v02_01[] = {
    /////////////////  name , type,    #states, hd#   cap
    LW_DISP_CHAN_DESC("core", CORE, dCoreState,  0  , DISP_DEFLT|DISP_DEBUG|DISP_SPVSR|DISP_CORE)

    LW_DISP_CHAN_DESC("base", BASE, dBaseState,  0  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("base", BASE, dBaseState,  1  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("base", BASE, dBaseState,  2  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("base", BASE, dBaseState,  3  , DISP_DEFLT|DISP_DEBUG                     ) 

    LW_DISP_CHAN_DESC("ovly", OVLY, dOvlyState,  0  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("ovly", OVLY, dOvlyState,  1  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("ovly", OVLY, dOvlyState,  2  , DISP_DEFLT|DISP_DEBUG                     ) 
    LW_DISP_CHAN_DESC("ovly", OVLY, dOvlyState,  3  , DISP_DEFLT|DISP_DEBUG                     ) 

    LW_DISP_CHAN_DESC("ovim", OVIM, dOvimState,  0  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("ovim", OVIM, dOvimState,  1  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("ovim", OVIM, dOvimState,  2  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("ovim", OVIM, dOvimState,  3  , DISP_DEFLT                                ) 

    LW_DISP_CHAN_DESC("lwrs", LWRS, dLwrsState,  0  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("lwrs", LWRS, dLwrsState,  1  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("lwrs", LWRS, dLwrsState,  2  , DISP_DEFLT                                ) 
    LW_DISP_CHAN_DESC("lwrs", LWRS, dLwrsState,  3  , DISP_DEFLT                                ) 
};

#define DISP_PRINT_SC_NON_IDX_V02_01(r, chanNum)                                \
    assy = GPU_REG_RD32(LW_UDISP_DSI_CHN_ASSY_BASEADR(chanNum)  + r);           \
    arm  = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r);           \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%48s: 0x%08x | 0x%08x", commandString, assy, arm);                 \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");

#define DISP_PRINT_SC_SINGLE_IDX_V02_01(r,idx, chanNum)                         \
    assy = GPU_REG_RD32(LW_UDISP_DSI_CHN_ASSY_BASEADR(chanNum)  + r(idx));      \
    arm  = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r(idx));      \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%45s[%1u]: 0x%08x | 0x%08x", commandString, idx, assy, arm);       \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");

#define DISP_PRINT_SC_DOUBLE_IDX_V02_01(r, surf, offset, chanNum)               \
    assy = GPU_REG_RD32(LW_UDISP_DSI_CHN_ASSY_BASEADR(chanNum)  + r(surf, offset)); \
    arm  = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r(surf, offset)); \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%43s[%1u,%1u]: 0x%08x | 0x%08x", commandString, surf, offset, assy, arm); \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");

#define DISP_PRINT_SC_VAR_NON_IDX_V02_01(r, chanNum)                            \
    {                                                                           \
    LwU32 arm  = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r);     \
    dprintf("%50s: 0x%08x   |   0x%08x", #r, arm,                               \
                    LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r);               \
    dprintf("\n");                                                              \
    }                                                                           

#define DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(r,idx, chanNum)                     \
    {                                                                           \
    LwU32 arm  = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r(idx)); \
    dprintf("%47s[%1u]: 0x%08x   |   0x%08x", #r, idx, arm,                     \
                    LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + r(idx));          \
    dprintf("\n");                                                              \
    }   
    
#define PRINTLOCKPIN(lockPin, dataStruct1, dataStruct2, value) \
        { \
                            switch((dataStruct1+head)->lockPin) \
                            { \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_0: \
                                (dataStruct2+head)->value = "/Pin-0"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_1: \
                                (dataStruct2+head)->value = "/Pin-1"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_2: \
                                (dataStruct2+head)->value = "/Pin-2"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_3: \
                                (dataStruct2+head)->value = "/Pin-3"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_4: \
                                (dataStruct2+head)->value = "/Pin-4"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_5: \
                                (dataStruct2+head)->value = "/Pin-5"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_6: \
                                (dataStruct2+head)->value = "/Pin-6"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_7: \
                                (dataStruct2+head)->value = "/Pin-7"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_8: \
                                (dataStruct2+head)->value = "/Pin-8"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_9: \
                                (dataStruct2+head)->value = "/Pin-9"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_A: \
                                (dataStruct2+head)->value = "/Pin-A"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_B: \
                                (dataStruct2+head)->value = "/Pin-B"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_C: \
                                (dataStruct2+head)->value = "/Pin-C"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_D: \
                                (dataStruct2+head)->value = "/Pin-D"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_E: \
                                (dataStruct2+head)->value = "/Pin-E"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_PIN_LOCK_PIN_F: \
                                (dataStruct2+head)->value = "/Pin-F"; \
                                break; \
                            default: \
                                (dataStruct2+head)->value = "/Internal"; \
                                break; \
                            } \
        }   

#define PRINTLOCKMODE(lockMode, dataStruct1, dataStruct2, value) \
{ \
                            switch((dataStruct1+head)->lockMode) \
                            { \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_NO_LOCK: \
                                (dataStruct2+head)->value = "No-Lock"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_FRAME_LOCK: \
                                (dataStruct2+head)->value = "Frame-Lock"; \
                                break; \
                            case LW_PDISP_VGA_HEAD_SET_CONTROL_SLAVE_LOCK_MODE_RASTER_LOCK: \
                               (dataStruct2+head)->value = "Raster-Lock"; \
                                break; \
                            default: \
                                (dataStruct2+head)->value = "N/A"; \
                                break; \
                            } \
}

#endif
