/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl208f/ctrl208fbif.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
* LW208F_CTRL_CMD_BIF_PBI_WRITE_COMMAND
*
* Control command to send a write command to the Post Box Interface
*
* Parameters:
*
*cmdFuncId
*   this specifies the function that needs to be performed on pbi
*data
*   the data to be set in the data in register
* status 
*   this corresponds to pbi status register
* sysNotify
*   this corresponds to system notify event, i.e. whether system
*   needs to be notified of command completion
* drvNotify
*   this corresponds to driver notify event, i.e. whether driver
*   needs to be notified of command completion
*
* For the possible values of the above parameters refer rmpbicmdif.h
*/
#define LW208F_CTRL_CMD_BIF_PBI_WRITE_COMMAND (0x208f0701) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BIF_INTERFACE_ID << 8) | LW208F_CTRL_BIF_PBI_WRITE_COMMAND_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BIF_PBI_WRITE_COMMAND_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_BIF_PBI_WRITE_COMMAND_PARAMS {
    LwU8   cmdFuncId;
    LwU32  data;
    LwU8   status;
    LwBool sysNotify;
    LwBool drvNotify;
} LW208F_CTRL_BIF_PBI_WRITE_COMMAND_PARAMS;

/*
* LW208F_CTRL_CMD_BIF_CONFIG_REG_READ
*   This command is used to read any of the PBI registers in the config space
*
* Parameters:
*
* RegIndex
*   Defines the index of the PBI register
* data
*   Data that is read
*/
#define LW208F_CTRL_CMD_BIF_CONFIG_REG_READ (0x208f0702) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BIF_INTERFACE_ID << 8) | LW208F_CTRL_BIF_CONFIG_REG_READ_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BIF_CONFIG_REG_READ_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_BIF_CONFIG_REG_READ_PARAMS {
    LwU8  RegIndex;
    LwU32 data;
} LW208F_CTRL_BIF_CONFIG_REG_READ_PARAMS;

/*
* LW208F_CTRL_CMD_BIF_CONFIG_REG_WRITE
*   This command is used to write any of the PBI registers in the config space
*
* Parameters:
*
* RegIndex
*   Defines the index of the PBI register
* data
*   Data that is to be written
*/
#define LW208F_CTRL_CMD_BIF_CONFIG_REG_WRITE (0x208f0703) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BIF_INTERFACE_ID << 8) | LW208F_CTRL_BIF_CONFIG_REG_WRITE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BIF_CONFIG_REG_WRITE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_BIF_CONFIG_REG_WRITE_PARAMS {
    LwU8  RegIndex;
    LwU32 data;
} LW208F_CTRL_BIF_CONFIG_REG_WRITE_PARAMS;

/*
* LW208F_CTRL_CMD_BIF_INFO
*   This command is used to read a bif property
*
* Parameters:
*
* index
*   Defines the index of the property to read
* data
*   Data that is read 
*/
#define LW208F_CTRL_CMD_BIF_INFO (0x208f0704) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BIF_INTERFACE_ID << 8) | LW208F_CTRL_BIF_INFO_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BIF_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW208F_CTRL_BIF_INFO_PARAMS {
    LwU32 index;
    LwU32 data;
} LW208F_CTRL_BIF_INFO_PARAMS;

/* valid bif info index values */
#define LW208F_CTRL_BIF_INFO_INDEX_L0S_ENABLED (0x00000000)
#define LW208F_CTRL_BIF_INFO_INDEX_L1_ENABLED  (0x00000001)


/* _ctrl208fbif_h_ */
