
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
 * @file  smbpbi.c
 * @brief WinDbg Extension for SMBus Post-Box Interface (SMBPBI).
 */

/* ------------------------ Includes ---------------------------------------- */

#include "smbpbi.h"
#include "pmu.h"
#include "rmpmucmdif.h"

/* ------------------------ Function Prototypes ----------------------------- */
/* ------------------------ Global Variables -------------------------------- */

LwBool SmbpbiPostedCommandPending = LW_FALSE;
LwU32  SmbpbiMutexToken           = PMU_ILWALID_MUTEX_OWNER_ID;

/* ------------------------ Public Functions -------------------------------- */

LW_STATUS
smbpbiExelwteCommand
(
    SMBPBI_CONTEXT *pContext
)
{
    LwBool    bPosted = LW_FALSE;
    LwU8      opcode;
    LwU8      arg1;
    LwU32     token   = PMU_ILWALID_MUTEX_OWNER_ID;
    LwU32     cmd;
    LwU32     dataIn;
    LwU32     dataOut;
    LW_STATUS status  = LW_OK;

    cmd     = pContext->cmd;
    dataIn  = pContext->dataIn;
    dataOut = pContext->dataOut;
    opcode  = (LwU8)DRF_VAL(_MSGBOX, _CMD, _OPCODE, cmd);
    arg1    = LW_MSGBOX_GET_CMD_ARG1(cmd);

    if ((opcode == LW_MSGBOX_CMD_OPCODE_GPU_REQUEST_CPL) ||
        (opcode == LW_MSGBOX_CMD_OPCODE_SET_MASTER_CAPS) ||
        (opcode == LW_MSGBOX_CMD_OPCODE_GPU_SYSCONTROL)  ||
        (opcode == LW_MSGBOX_CMD_OPCODE_GPU_PCONTROL))
    {
        bPosted = LW_TRUE;
    }
    
#if 0
    if (opcode == LW_MSGBOX_CMD_OPCODE_GPU_PCONTROL)
    {
       LwU8 action = (LwU8)LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_GET_ACTION(arg1);
       LwU8 target = (LwU8)LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_GET_TARGET(arg1);

       if ((target != LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_TARGET_VPSTATE) ||
           (action != LW_MSGBOX_CMD_GPU_PCONTROL_ARG1_ACTION_GET_STATUS))
       {
           bPosted = LW_TRUE;
       }
    }
#endif

    status = pmuAcquireMutex(RM_PMU_MUTEX_ID_MSGBOX, 10, &token);
    if (status != LW_OK)
    {
        dprintf("lw: Failed to acquire SMBPBI mutex. Cannot complete "
                "operation. Please retry later.\n");
        return status;
    }

    // make sure the interface is up and running
    pSmbpbi[indexGpu].smbpbiGetContext(pContext);
    if (FLD_TEST_DRF(_MSGBOX, _CMD, _STATUS, _NULL, pContext->cmd))
    {
        dprintf("lw: SMBPBI interface is not ready (status=NULL).\n");
        status = LW_ERR_GENERIC;
        goto smbpbiExelwteCommand_exit;
    }

    pContext->cmd     = cmd;
    pContext->dataIn  = dataIn;
    pContext->dataOut = dataOut;

    pContext->cmd = FLD_SET_DRF(_MSGBOX, _CMD, _STATUS, _READY, pContext->cmd);
    while (FLD_TEST_DRF(_MSGBOX, _CMD, _STATUS, _READY, pContext->cmd))
    {
        // 
        // Ensure that writing the command will clear the STATUS field. On a
        // retry, it is also necessary to re-set the INTR_PENDING bit.
        //
        pContext->cmd =
            FLD_SET_DRF(_MSGBOX, _CMD, _STATUS, _NULL   , pContext->cmd);
        pContext->cmd =
            FLD_SET_DRF(_MSGBOX, _CMD, _INTR  , _PENDING, pContext->cmd);
        
        // issue the command
        pSmbpbi[indexGpu].smbpbiSetContext(pContext);

        // do not wait on the response for posted commands
        if (bPosted)
        {
            SmbpbiPostedCommandPending = LW_TRUE;
            SmbpbiMutexToken           = token;

            status = LW_ERR_MORE_PROCESSING_REQUIRED;
            goto smbpbiExelwteCommand_exit;
        }
        
        // wait for completion
        while (FLD_TEST_DRF(_MSGBOX, _CMD, _STATUS, _NULL, pContext->cmd))
        {
            pSmbpbi[indexGpu].smbpbiGetContext(pContext);
        }
    }

smbpbiExelwteCommand_exit:
    //
    // Do not release the mutex upon successful exelwtion of a posted command.
    // The mutex needs to be held until the user gets a chance to view the
    // results.
    //
    if (!SmbpbiPostedCommandPending)
    {
        pmuReleaseMutex(RM_PMU_MUTEX_ID_MSGBOX, &token);
    }
    return status;
}

LwU32
smbpbiGetCapabilities
(
    LwU8 dwordIdx
)
{
    SMBPBI_CONTEXT context = {0};
    LW_STATUS      status;

    context.cmd = LW_MSGBOX_CMD(_GET_CAP_DWORD, dwordIdx, 0);
    status = smbpbiExelwteCommand(&context);
    return (status == LW_OK) ?  context.dataIn : 0;
}

void
smbpbiReleaseInterface(void)
{
    if ((SmbpbiPostedCommandPending) && 
        (SmbpbiMutexToken != PMU_ILWALID_MUTEX_OWNER_ID))
    {
        (void)pmuReleaseMutex(RM_PMU_MUTEX_ID_MSGBOX, &SmbpbiMutexToken);
        SmbpbiPostedCommandPending = LW_FALSE;
    }
}

