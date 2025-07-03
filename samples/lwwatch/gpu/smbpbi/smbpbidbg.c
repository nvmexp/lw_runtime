
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  smbpbidbg.c
 * @brief WinDbg Extension for SMBus Post-Box Interface (SMBPBI).
 */

/* ------------------------ Includes ---------------------------------------- */

#include "smbpbi.h"
#include "oob/smbpbi_priv.h"

/* ------------------------ Typedefs ---------------------------------------- */

typedef void SMBPBI_CONTEXT_PRINTER(SMBPBI_CONTEXT *pContext);

/* ------------------------ Global Variables -------------------------------- */

SMBPBI_CONTEXT_PRINTER _smbpbiPrintContext_PCONTROL;

static const struct
{
    LwU8                    opcode;
    SMBPBI_CONTEXT_PRINTER *pPrinter;
} SmbpbiContextPrinters[] =
{
    {LW_MSGBOX_CMD_OPCODE_GPU_PCONTROL     , _smbpbiPrintContext_PCONTROL},
};

static const struct
{
    LwU8        opcode;
    const char *pName;
} SmbpbiOpcodeStrings[] =
{
    {LW_MSGBOX_CMD_OPCODE_NULL_CMD         , "NULL"            },
    {LW_MSGBOX_CMD_OPCODE_GET_CAP_DWORD    , "GET_CAP_DWORD"   },
    {LW_MSGBOX_CMD_OPCODE_GET_TEMP         , "GET_TEMP"        },
    {LW_MSGBOX_CMD_OPCODE_GET_EXT_TEMP     , "GET_EXT_TEMP"    },
    {LW_MSGBOX_CMD_OPCODE_GET_POWER        , "GET_POWER"       },
    {LW_MSGBOX_CMD_OPCODE_GET_SYS_ID_DATA  , "GET_SYS_ID_DATA" },
    {LW_MSGBOX_CMD_OPCODE_GET_ECC_V1       , "GET_ECC_V1"      },
    {LW_MSGBOX_CMD_OPCODE_GET_ECC_V2       , "GET_ECC_V2"      },
    {LW_MSGBOX_CMD_OPCODE_GPU_PCONTROL     , "GPU_PCONTROL"    },
    {LW_MSGBOX_CMD_OPCODE_GPU_SYSCONTROL   , "GPU_SYSCONTROL"  },
    {LW_MSGBOX_CMD_OPCODE_SET_MASTER_CAPS  , "SET_MASTER_CAPS" },
    {LW_MSGBOX_CMD_OPCODE_GPU_REQUEST_CPL  , "GPU_REQUEST_CPL" },
};

static const struct
{
    LwU8        status;
    const char *pName;
} SmbpbiStatusStrings[] =
{
    {LW_MSGBOX_CMD_STATUS_NULL             , "NULL"              },
    {LW_MSGBOX_CMD_STATUS_ERR_REQUEST      , "ERR_REQUEST"       },
    {LW_MSGBOX_CMD_STATUS_ERR_OPCODE       , "ERR_OPCODE"        },
    {LW_MSGBOX_CMD_STATUS_ERR_ARG1         , "ERR_ARG1"          },
    {LW_MSGBOX_CMD_STATUS_ERR_ARG2         , "ERR_ARG2"          },
    {LW_MSGBOX_CMD_STATUS_ERR_DATA         , "ERR_DATA"          },
    {LW_MSGBOX_CMD_STATUS_ERR_MISC         , "ERR_MISC"          },
    {LW_MSGBOX_CMD_STATUS_ERR_I2C_ACCESS   , "ERR_I2C_ACCESS"    },
    {LW_MSGBOX_CMD_STATUS_ERR_NOT_SUPPORTED, "ERR_NOT_SUPPORTED" },
    {LW_MSGBOX_CMD_STATUS_ERR_NOT_AVAILABLE, "ERR_NOT_AVAILABLE" },
    {LW_MSGBOX_CMD_STATUS_INACTIVE         , "INACTIVE"          },
    {LW_MSGBOX_CMD_STATUS_READY            , "READY"             },
    {LW_MSGBOX_CMD_STATUS_SUCCESS          , "SUCCESS"           },
};

/* ------------------------ Function Prototypes ----------------------------- */

static char                   *_smbpbiGetNextWord     (char **ppCmd);
static void                    _smbpbiExecClearContext(void);
static void                    _smbpbiExecCommand     (char  *pCmd);
static void                    _smbpbiExecDumpCaps    (void);
static void                    _smbpbiExecDumpStatus  (void);
static const char             *_smbpbiGetOpcodeString (LwU8 opcode);
static const char             *_smbpbiGetStatusString (LwU8 pbiStatus);
static SMBPBI_CONTEXT_PRINTER *_smbpbiGetContextPrinter(LwU8 opcode);
static void                    _smbpbiPrintUsage(void);

/* ------------------------ Public Functions -------------------------------- */

void 
smbpbiExec
(
    char *pCmd
)
{
    char *pCmdName;
    pCmdName = _smbpbiGetNextWord(&pCmd);
    if (strcmp(pCmdName, "status") == 0)
    {
        _smbpbiExecDumpStatus();
        //
        // The user may have requested status in order to obtain the results
        // from a previously issued posted command. When that oclwrs, the
        // interface needs released (mutex released) to allow new commands to
        // flow through.
        //
        smbpbiReleaseInterface();
    }
    else if (strcmp(pCmdName, "caps") == 0)
    {
        _smbpbiExecDumpCaps();
    }
    else if ((strcmp(pCmdName, "cmd")     == 0) ||
             (strcmp(pCmdName, "command") == 0))
    {
        _smbpbiExecCommand(pCmd);
    }
    else if (strcmp(pCmdName, "clear") == 0)
    {
        _smbpbiExecClearContext();
    }
    else if (strcmp(pCmdName, "help") == 0)
    {
        _smbpbiPrintUsage();
    }
    else
    {
        dprintf("lw: Unrecognized smbpbi command: %s\n", pCmdName);
        dprintf("lw:\n");
        _smbpbiPrintUsage();
    }
    return;
}

static void
_smbpbiPrintUsage(void)
{
    dprintf("lw: Usage:\n");
    dprintf("lw:\n");
    dprintf("lw: !smbpbi <function> [args]\n");
    dprintf("lw:\n");
    dprintf("lw:     Available functions:\n");
    dprintf("lw:        status      - Dump the SMBPBI status\n");
    dprintf("lw:        caps        - Dump the SMBPBI capability dwords\n");
    dprintf("lw:        cmd|command - Send a SMBPBI command to the PMU\n");
    dprintf("lw:        clear       - Clear all SMBPBI register state/context\n");
    dprintf("lw:        help        - Print this message\n");
    dprintf("lw:\n");
    dprintf("lw:     Function-Specific Usage:\n");
    dprintf("lw:          !smbpbi cmd <opcode> [arg1] [arg2] [data-in]\n");
}

/* ------------------------ Private Functions ------------------------------- */

static void 
_smbpbiExecClearContext(void)
{
    SMBPBI_CONTEXT context;
    LW_STATUS      status;
    LwU8           pbiStatus;

    status = pSmbpbi[indexGpu].smbpbiGetContext(&context);
    if (status == LW_OK)
    {
        pbiStatus = LW_MSGBOX_GET_CMD_STATUS(context.cmd);

        context.dataIn  = 0;
        context.dataOut = 0;
        context.cmd     =
            FLD_SET_DRF_NUM(_MSGBOX, _CMD, _STATUS, pbiStatus, 0);

        pSmbpbi[indexGpu].smbpbiSetContext(&context);
        dprintf("lw: Done\n");
        _smbpbiExecDumpStatus();
    }
}

static void
_smbpbiExecCommand
(
    char *pCmd
)
{
    SMBPBI_CONTEXT context = {0};
    LW_STATUS      status;
    LwU64          opcode = 0;
    LwU64          arg1   = 0;
    LwU64          arg2   = 0;
    LwU64          data   = 0;
    const char    *pOpcodeString;

    if (!GetExpressionEx(pCmd, &opcode, &pCmd))
    {
        dprintf("lw: Opcode not provided. Cannot submit request.\n");
        dprintf("lw:\n");
        dprintf("lw: Usage: smbpbi cmd opcode [arg1] [arg2] [data]\n");
        return;
    }

    if (GetExpressionEx(pCmd, &arg1, &pCmd))
    {
        if (GetExpressionEx(pCmd, &arg2, &pCmd))
        {
            GetExpressionEx(pCmd, &data, &pCmd);
        }
    }

    context.dataIn = (LwU32)data;
    context.cmd =
        DRF_NUM(_MSGBOX, _CMD, _OPCODE, (LwU32)opcode) |
        DRF_NUM(_MSGBOX, _CMD, _ARG1  , (LwU32)arg1)   |
        DRF_NUM(_MSGBOX, _CMD, _ARG2  , (LwU32)arg2)   |
        DRF_DEF(_MSGBOX, _CMD, _STATUS, _NULL)         |
        DRF_DEF(_MSGBOX, _CMD, _RSVD  , _INIT)         |
        DRF_DEF(_MSGBOX, _CMD, _INTR  , _PENDING);

    pOpcodeString =_smbpbiGetOpcodeString((LwU8)opcode);
    if (pOpcodeString == NULL)
    {
        pOpcodeString = "???";
    }

    dprintf("lw: Exelwting command (0x%08x):\n", context.cmd);
    dprintf("lw:\n");
    dprintf("lw: ----------------------------------------------------------\n");
    dprintf("lw: Inputs:\n");
    dprintf("lw:     opcode: 0x%x (%s)\n"      , (LwU32)opcode, pOpcodeString);
    dprintf("lw:       arg1: 0x%02x\n"         , (LwU8)arg1);
    dprintf("lw:       arg2: 0x%02x\n"         , (LwU8)arg2);
    dprintf("lw:       data: 0x%08x\n"         , (LwU32)data);
    dprintf("lw:\n");

    status = smbpbiExelwteCommand(&context);
    if (status == LW_ERR_MORE_PROCESSING_REQUIRED)
    {
        dprintf("lw: Command was issued as a posted command. You may use\n"
                "lw: 'smbpbi status' to check on the progress of the command.\n");
        dprintf("lw:\n");
        dprintf("lw: If this was not expected, it may have been forced based\n"
                "lw: on the command-type (opcode). Some commands require RM-\n"
                "lw: side processing which cannot complete while the debugger\n"
                "lw: is active. Please resume the debugger in order to allow\n"
                "lw: the RM to complete the request and break in later to\n"
                "lw: check on the status of the command.\n");
        return;
    }
    else
    if (status != LW_OK)
    {
        dprintf("lw: %s: Failed to submit SMBPBI command (status=%d)\n",
                __FUNCTION__, status);
        return;
    }
    dprintf("lw: ----------------------------------------------------------\n");
    dprintf("lw: Outputs:\n");
    _smbpbiExecDumpStatus();
}

static void
_smbpbiExecDumpCaps(void)
{
    LwU32 caps;
    LwU8  i;

    dprintf("lw: SMBPBI Capabilities\n");
    dprintf("lw:\n");

    for (i = 0; i < LW_MSGBOX_DATA_CAP_COUNT; i++)
    {
        caps = smbpbiGetCapabilities(i);
        dprintf("lw: [%d] 0x%08x\n", i, caps);
    }
    return;
}

static void
_smbpbiExecDumpStatus(void)
{
    SMBPBI_CONTEXT          context = {0};
    SMBPBI_CONTEXT_PRINTER *pPrinter;
    LW_STATUS               status;
    LwU8                    opcode;
    LwU8                    arg1;
    LwU8                    arg2;
    LwU8                    pbiStatus;
    const char             *pOpcodeString;
    const char             *pStatusString;

    status = pSmbpbi[indexGpu].smbpbiGetContext(&context);
    if (status != LW_OK)
    {
        dprintf("lw: Unexpected error %d. Cannot retrieve SMBPBI context.\n",
                status);
        return;
    }

    opcode    = LW_MSGBOX_GET_CMD_OPCODE(context.cmd);
    arg1      = LW_MSGBOX_GET_CMD_ARG1(context.cmd);
    arg2      = LW_MSGBOX_GET_CMD_ARG2(context.cmd);
    pbiStatus = LW_MSGBOX_GET_CMD_STATUS(context.cmd);

    pStatusString = _smbpbiGetStatusString(pbiStatus);
    if (pStatusString == NULL)
    {
        pStatusString = "???";
    }
    pOpcodeString =_smbpbiGetOpcodeString((LwU8)opcode);
    if (pOpcodeString == NULL)
    {
        pOpcodeString = "???";
    }

    dprintf("lw:   Command: 0x%08x\tOpcode: 0x%02x (%s)\n", context.cmd    , opcode, pOpcodeString);
    dprintf("lw:   Data-In: 0x%08x\t  Arg1: 0x%02x\n"     , context.dataIn , arg1);
    dprintf("lw:  Data-Out: 0x%08x\t  Arg2: 0x%02x\n"     , context.dataOut, arg2);
    dprintf("lw:                \t\tStatus: 0x%02x (%s)\n", pbiStatus      , pStatusString);

    pPrinter = _smbpbiGetContextPrinter(opcode);
    if (pPrinter != NULL)
    {
        dprintf("lw:\n");
        pPrinter(&context);
    }
    dprintf("lw:\n");
}

static SMBPBI_CONTEXT_PRINTER *
_smbpbiGetContextPrinter
(
    LwU8 opcode
)
{
    LwU8 i;
    for (i = 0; i < LW_ARRAY_ELEMENTS(SmbpbiContextPrinters); i++)
    {
        if (SmbpbiContextPrinters[i].opcode == opcode)
        {
            return SmbpbiContextPrinters[i].pPrinter;
        }
    }
    return NULL;
}

static const char *
_smbpbiGetOpcodeString
(
    LwU8 opcode
)
{
    LwU8 i;
    for (i = 0; i < LW_ARRAY_ELEMENTS(SmbpbiOpcodeStrings); i++)
    {
        if (SmbpbiOpcodeStrings[i].opcode == opcode)
        {
            return SmbpbiOpcodeStrings[i].pName;
        }
    }
    return NULL;
}

static const char *
_smbpbiGetStatusString
(
    LwU8 pbiStatus
)
{
    LwU8 i;
    for (i = 0; i < LW_ARRAY_ELEMENTS(SmbpbiStatusStrings); i++)
    {
        if (SmbpbiStatusStrings[i].status == pbiStatus)
        {
            return SmbpbiStatusStrings[i].pName;
        }
    }
    return NULL;
}

static char *
_smbpbiGetNextWord
(
    char **ppCmd
)
{
    char *pCmd  = *ppCmd;
    char *pWord = NULL;

    // strip-off leading whitespace
    while (*pCmd == ' ')
    {
        pCmd++;
    }
    pWord = pCmd;

    // command-name ends at first whitespace character or EOS
    while ((*pCmd != ' ') && (*pCmd != '\0'))
    {
        pCmd++;
    }

    if (*pCmd != '\0')
    {
        *pCmd  = '\0';
        *ppCmd = pCmd + 1;
    }
    else
    {
        *ppCmd = pCmd;
    }
    return pWord;
}

/* ------------------------ Context-Specific Print Functions ---------------- */

void
_smbpbiPrintContext_PCONTROL
(
    SMBPBI_CONTEXT *pContext
)
{
    LwU8 arg1;
    LwU8 arg2;
    LwU8 action;
    LwU8 target;
    LwU8 limit;
    LwU8 vPstateMax;
    LwU8 vPstateMin;

#if 0
    LwU8 vPstateLwrr;
#endif

    arg1   = LW_MSGBOX_GET_CMD_ARG1(pContext->cmd);
    arg2   = LW_MSGBOX_GET_CMD_ARG2(pContext->cmd);
    action = LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_GET_ACTION(arg1);
    target = LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_GET_TARGET(arg1);

    dprintf("lw: PCONTROL {action=");
    switch (action)
    {
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_INFO:
            dprintf("GET_INFO");
            break;
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_LIMIT:
            dprintf("GET_LIMIT");
            break;
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_SET_LIMIT:
            dprintf("SET_LIMIT");
            break;
#if 0
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_STATUS:
            dprintf("GET_STATUS");
            break;
#endif
        default:
            dprintf("???");
    }
    dprintf(",target=");
    switch (target)
    {
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE:
            dprintf("VPSTATE");
            break;
        default:
            dprintf("???");
    }
    dprintf("}\n");

    switch (action)
    {
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_INFO:
            if (target == LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE)
            {
                vPstateMax  = LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_GET_INFO_GET_MAX(pContext->dataIn);
                vPstateMin  = LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_GET_INFO_GET_MIN(pContext->dataIn);
                dprintf("lw:     MAX: 0x%02x (slowest)\n", vPstateMax);
                dprintf("lw:     MIN: 0x%02x (fastest)\n", vPstateMin);
            }
            break;

        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_LIMIT:
            if (target == LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE)
            {
                limit = LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_GET_LIMIT_GET_VALUE(
                           pContext->dataIn);
                dprintf("lw:    LIMIT=0x%02x\n", limit);
            }
            break;

        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_SET_LIMIT:
            if (target == LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE)
            {
                limit = LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_SET_LIMIT_GET_VALUE(
                           pContext->dataIn);
                if (limit != 
                        LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_SET_LIMIT_VALUE_CLEAR)
                    dprintf("lw:    LIMIT=0x%02x\n", limit);
                else
                    dprintf("lw:    LIMIT=0x%02x (CLEAR)\n", limit);
            }
            break;

#if 0
        case LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_STATUS:
            if (target == LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE);
            {
                vPstateLwrr = LW_MSGBOX_CMD_GPU_PCONTROL_DATA_VPSTATE_GET_STATUS_GET_LWRRENT(pContext->dataIn);
                dprintf("lw:    LWRR: 0x%02x\n"          , vPstateLwrr);
            }
            break;
#endif
    }
}

