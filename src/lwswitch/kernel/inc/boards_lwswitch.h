/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _BOARDS_LWSWITCH_H_
#define _BOARDS_LWSWITCH_H_

//
// LWSwitch board IDs
//
#define LWSWITCH_BOARD_UNKNOWN              0x0

#define LWSWITCH_BOARD_LR10_4700_0000_PC0   0x01C5
#define LWSWITCH_BOARD_LR10_4612_0301_ES    0x01D1
#define LWSWITCH_BOARD_LR10_4700_0000_STA   0x01D2
#define LWSWITCH_BOARD_LR10_4612_0301_891   0x01E4
#define LWSWITCH_BOARD_LR10_4612_0301_890   0x01E5
#define LWSWITCH_BOARD_LR10_4612_0300_891   0x01E6
#define LWSWITCH_BOARD_LR10_4612_0300_890   0x01E7
#define LWSWITCH_BOARD_LR10_3517_0300_890   0x02F7
#define LWSWITCH_BOARD_LR10_3597_0000_891   0x0365

#define LWSWITCH_BOARD_UNKNOWN_NAME             "UNKNOWN"

#define LWSWITCH_BOARD_LR10_4700_0000_PC0_NAME  "LR10_4700_0000_PC0"
#define LWSWITCH_BOARD_LR10_4612_0301_ES_NAME   "LR10_4612_0301_ES"
#define LWSWITCH_BOARD_LR10_4700_0000_STA_NAME  "LR10_4700_0000_STA"
#define LWSWITCH_BOARD_LR10_4612_0301_891_NAME  "LR10_4612_0301_891"
#define LWSWITCH_BOARD_LR10_4612_0301_890_NAME  "LR10_4612_0301_890"
#define LWSWITCH_BOARD_LR10_4612_0300_891_NAME  "LR10_4612_0300_891"
#define LWSWITCH_BOARD_LR10_4612_0300_890_NAME  "LR10_4612_0300_890"
#define LWSWITCH_BOARD_LR10_3517_0300_890_NAME  "LR10_3517_0300_890"
#define LWSWITCH_BOARD_LR10_3597_0000_891_NAME  "LR10_3597_0000_891"

typedef struct {
    LwU16 boardId;
    const char* boardName;
} LWSWITCH_BOARD_ENTRY;

#endif // _BOARDS_LWSWITCH_H_
