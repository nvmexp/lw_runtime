
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbt124.c
//
//*****************************************************

//
// includes
//
#include "t12x/t124/hwproject.h"
#include "msdec.h"

#define N_FB_READS      3

/*!
 * @brief Gets the LTS per LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The LTS per LTC count.
 */
LwU32
fbGetLTSPerLTCCountLwW_T124( void )
{
    return LW_SCAL_LITTER_NUM_LTC_SLICES;
}

LW_STATUS fbTest_T124( void )
{
    LwU32 i,j;
    LwU32 prevValue = 0;
    LwU32 value = 0xdeadbeef;
    LwU32 testValue[N_FB_READS];
    LW_STATUS status = LW_OK;
    LW_STATUS retVal = LW_OK;
    LwU32 readStatus = TRUE;
    VMemSpace iVMemSpace;
    LwU64 addr = 0;
    LwU32 partitions = pFb[indexGpu].fbGetActiveFbpCount();

    // do R/W test on active partitions
    if (partitions == 0)
    {
        dprintf("lw: Error. Incorrect number of FB partitions\n");
        addUnitErr("\t Incorrect number of FB partitions: 0x%08x\n", partitions);
        return LW_ERR_GENERIC;
    }
    
    dprintf("lw: Partitions:0x%02x\n", partitions); 

    /* Take bar1 instance block ptr to perform fbread/fbwrite */
    status = vmemGet(&iVMemSpace, VMEM_TYPE_BAR1, NULL);
    if (status == LW_ERR_GENERIC)
    {
        dprintf("lw: Could not fetch vmemspace for #%d. \n", indexGpu);
        addUnitErr("\t Could not fetch vmemspace for #%d. \n", indexGpu);
        return LW_ERR_GENERIC;
    }

    dprintf("lw: Start bypassing L2 ...\n"); 
    pFb[indexGpu].fbL2BypassEnable(TRUE);

    for (i = 0; i < partitions; i++)
    {
        /* The 4096 bytes instance block has so far oclwpied with 167*4 
         * bytes so the remianing 857*4 bytes are unused. using the 
         * 1023rd DWORD for read/write purpose. 
         */
        addr = iVMemSpace.instBlock.instBlockAddr + 1023*4;
        dprintf("lw: Testing partition %u @ 0x%llx\n", i, addr);

        status = pFb[indexGpu].fbRead(addr, &prevValue, 4);

        if (status == LW_ERR_GENERIC)
        {
           retVal = LW_ERR_GENERIC;
           dprintf("lw: Partition %u FAILED read test\n", i);
           addUnitErr("\t Partition %u FAILED read test\n", i);
           continue;
        }

        //now do repeated reads at the same address
        
        pFb[indexGpu].fbWrite(addr, &value, 4);

        status = LW_OK;
        for (j=0;j<N_FB_READS;j++)
        {
            status &= pFb[indexGpu].fbRead(addr, &testValue[j], 4);
        }

        pFb[indexGpu].fbWrite(addr, &prevValue, 4);

        if (status == LW_ERR_GENERIC)
        {
           retVal = LW_ERR_GENERIC;
           dprintf("lw: Partition %u FAILED read test\n", i);
           addUnitErr("\t Partition %u FAILED read test\n", i);
           continue;
        }

        if (verboseLevel > 1)
        {
            dprintf("lw: Value expected: 0x%08x\n", value);
            dprintf("lw: Values read:    ");
            for (j=0;j<N_FB_READS;j++)
            {
                dprintf(" 0x%08x\t", testValue[j]);
            }
            dprintf("\n");
        }

        for (j=0;j<N_FB_READS;j++)
        {
            readStatus = readStatus && (testValue[j] == value);
        }

        if ((readStatus == TRUE) && (status == LW_OK))
        {
            dprintf("lw: Partition %u passed test\n", i);
            retVal = LW_OK;
        }
        else
        {
            dprintf("lw: Partition %u FAILED write test\n", i);
            addUnitErr("\t Partition %u FAILED read test\n", i);
            retVal = LW_ERR_GENERIC;
        }   
    }

    dprintf("lw: Stop bypassing L2 ...\n"); 
    pFb[indexGpu].fbL2BypassEnable(FALSE);

    return retVal;
}
