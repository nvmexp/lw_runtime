/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _DISP_DISP0400_H_
#define _DISP_DISP0400_H_

#define LW_PDISP_FE_CHNCTL_CORE__SIZE_1        1
#define CHNST_SIZE(x) (sizeof(x) / sizeof(char *))

// CORE Channel State (LW_PDISP_FE_CHNSTATUS_CORE_STATE)
static const char * const dCoreState[] = {
    "INIT/DEALLOC",
    "DEALLOC_LIMBO",
    "VBIOS_INIT1",
    "VBIOS_INIT2",
    "VBIOS_OPERATION",
    "EFI_INIT1",
    "EFI_INIT2",
    "EFI_OPERATION",
    "UNCONNECTED",
    "INIT1",
    "INIT2",
    "IDLE",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

// WINDOW Channel State (LW_PDISP_FE_CHNSTATUS_WIN_STATE)
static const char * const dWinState[] = {
    "INIT/DEALLOC",
    "UNCONNECTED",
    "INIT1",
    "INIT2",
    "IDLE",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

// WINDOW immediate Channel State (LW_PDISP_FE_CHNSTATUS_WINIM_STATE)
static const char * const dWinimState[] = {
    "INIT/DEALLOC",
    "UNCONNECTED",
    "INIT1",
    "INIT2",
    "IDLE",
    "BUSY",
    "SHUTDOWN1",
    "SHUTDOWN2",
};

static const char * const * dChanStateWords_v04_00[] = {
    dCoreState,

    dWinState,
    dWinState,
    dWinState,
    dWinState,
    dWinState,
    dWinState,
    dWinState,
    dWinState,

    dWinimState,
    dWinimState,
    dWinimState,
    dWinimState,
    dWinimState,
    dWinimState,
    dWinimState,
    dWinimState
};

#define DCHN_GET_DESC_V04_00(x,y) (dChanStateWords_v04_00[x][y])

#define LW_DISP_CHAN_DESC(name,type,state,cap) \
        { name, 0, LW_PDISP_FE_CHNCTL_ ## type ## __SIZE_1, \
          DRF_SHIFT(LW_PDISP_FE_CHNSTATUS_ ## type ##  _STATE), \
          DRF_MASK(LW_PDISP_FE_CHNSTATUS_ ## type ## _STATE), \
          0, CHNST_SIZE(state), \
          LW_PDISP_SC_ ## type, \
          LW_PDISP_FE_CHNSTATUS_ ## type, cap, \
          LWDISPLAY_CHNTYPE_ ## type},

#define LW_DISP_CHAN_DESC_IDX(name,type,state,hn,cap) \
        { name, hn, LW_PDISP_FE_CHNCTL_ ## type ## __SIZE_1, \
          DRF_SHIFT(LW_PDISP_FE_CHNSTATUS_ ## type ## _STATE), \
          DRF_MASK(LW_PDISP_FE_CHNSTATUS_ ## type ## _STATE), \
          hn, CHNST_SIZE(state), \
          LW_PDISP_SC_ ## type(hn), \
          LW_PDISP_FE_CHNSTATUS_ ## type(hn), cap, \
          LWDISPLAY_CHNTYPE_ ## type},

#define DCHN_GET_CHNSTATUS_STATE_V04_00(_chnNum,_chnstatus) \
    (((_chnstatus) >> dispChanState_v04_00[_chnNum].shift) & \
     dispChanState_v04_00[_chnNum].mask)

static ChanDesc_t_Lwdisplay dispChanState_v04_00[] = {
    /////////////////  name , type,    #states,  cap
    LW_DISP_CHAN_DESC("core", CORE, dCoreState,  DISP_DEFLT|DISP_DEBUG|DISP_SPVSR|DISP_CORE)

    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  0  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  1  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  2  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  3  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  4  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  5  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  6  , DISP_DEFLT|DISP_DEBUG                     )
    LW_DISP_CHAN_DESC_IDX("win", WIN, dWinState,  7  , DISP_DEFLT|DISP_DEBUG                     )

    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  0  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  1  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  2  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  3  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  4  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  5  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  6  , DISP_DEFLT|DISP_DEBUG                          ) 
    LW_DISP_CHAN_DESC_IDX("winim", WINIM, dWinimState,  7  , DISP_DEFLT|DISP_DEBUG                          )
};

#define DISP_PRINT_SC_NON_IDX_V04_00(r, chanNum)                                \
    assy = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR(chanNum)  + r);            \
    arm  = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR(chanNum) + r);            \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%55s: 0x%08x | 0x%08x", commandString, assy, arm);                 \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");

#define DISP_PRINT_SC_SINGLE_IDX_COMP_V04_00(r, idx, chanNum)                   \
    assy = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR(chanNum)  + r(idx));       \
    arm  = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR(chanNum) + r(idx));       \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%52s[%1u]: 0x%08x | 0x%08x", commandString, idx, assy, arm);       \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");
                                                                        
#define DISP_PRINT_SC_DOUBLE_IDX_COMP_V04_00(r, surf, offset, chanNum)               \
    assy = GPU_REG_RD32(LW_UDISP_FE_CHN_ASSY_BASEADR(chanNum)  + r(surf, offset)); \
    arm  = GPU_REG_RD32(LW_UDISP_FE_CHN_ARMED_BASEADR(chanNum) + r(surf, offset)); \
    strcpy(commandString, #r);                                                  \
    for(i=0; i<5; i++)                                                          \
    {                                                                           \
        if(strlen(classString) == 0)                                            \
            continue;                                                           \
        commandString[i] = classString[i];                                      \
    }                                                                           \
    dprintf("%50s[%1u,%1u]: 0x%08x | 0x%08x", commandString, surf, offset, assy, arm); \
    if (assy != arm)                                                            \
    {                                                                           \
        dprintf(" |       Yes");                                                \
    }                                                                           \
    dprintf("\n");

#endif
