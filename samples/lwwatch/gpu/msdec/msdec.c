/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "msdec.h"

dbg_msdec *pMsdecPrivRegs = NULL;
static char *eng[]={"VLD","PDEC","PPP"};

//-----------------------------------------------------
// msdecEng
//-----------------------------------------------------
char* msdecEng(LwU32 idec)
{
    if (idec < 3)
        return eng[idec];
    else
        return NULL;
}

//-----------------------------------------------------
// msdecPrintPriv
//-----------------------------------------------------
void msdecPrintPriv(unsigned int clmn,char *tag,int id)
{
    size_t len = strlen(tag);
    dprintf("lw: %s",tag);

    if((len>0)&&(len<(clmn+4)))
    {
        unsigned int i;
        for(i=0;i<clmn-len;i++)
        {
            dprintf(" ");
        }
    }
    dprintf("(0x%08X)  = 0x%08X\n",id,GPU_REG_RD32(id));
}

//-----------------------------------------------------
// msdecGetPriv
//-----------------------------------------------------
void msdecGetPriv(void * fmt,int id, LwU32 idec)
{
    dprintf("lw:\n");
    dprintf("lw: -- %s priv registers -- \n",eng[id]);
    dprintf("lw:\n");
    // Dump PC vals
    pMsdec[indexGpu].msdecGetPcInfo(idec);

}

//-----------------------------------------------------
// msdecTestMsdecState
// @param eng - which engines to test: 0-msvld, 1-mspdec,
// 2-msppp, 3 - all
//-----------------------------------------------------
LW_STATUS msdecTestMsdecState( LwU32 eng )
{
    LW_STATUS status = LW_OK;

    if ( (eng == 0) || (eng == 3) )
    {
        dprintf("\n\tlw: ******** MSVLD state test... ********\n");
        if (pMsdec[indexGpu].msdecTestMsvldState() == LW_ERR_GENERIC)
        {
            dprintf("lw: ******** MSVLD state test FAILED ********\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: ******** MSVLD state test succeeded ********\n");
        }
    }

    if ( (eng == 1) || (eng == 3))
    {
        dprintf("\n\tlw: ******** MSPDEC state test... ********\n");
        if (pMsdec[indexGpu].msdecTestMspdecState() == LW_ERR_GENERIC)
        {
            dprintf("lw: ******** MSPDEC state test FAILED ********\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: ******** MSPDEC state test succeeded ********\n");
        }
    }

    if ( (eng == 2) || (eng == 3))
    {
        dprintf("\n\tlw: ******** MSPPP state test... ********\n");
        if (pMsdec[indexGpu].msdecTestMspppState() == LW_ERR_GENERIC)
        {
            dprintf("lw: ******** MSPPP state test FAILED ********\n");
            status = LW_ERR_GENERIC;
        }
        else
        {
            dprintf("lw: ******** MSPPP state test succeeded ********\n");
        }
    }
    
    return status;
}
