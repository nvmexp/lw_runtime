/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//*****************************************************
//
// Slavik Bryksin <sbryksin@lwpu.com>
// lwwatch extension
// gpuanalyzegt215.c
//
//*****************************************************


#include "gpuanalyze.h"

#include "chip.h"
#include "os.h"
#include "mmu.h"
#include "disp.h"
#include "fifo.h"
#include "pmu.h"
#include "vic.h"
#include "sig.h"
#include "fb.h"
#include "bus.h"
#include "gr.h"
#include "msdec.h"
#include "ce.h"
#include "chip.h"
#include "vmem.h"

#include <stdarg.h>

#include <stdio.h>


gpu_state gpuState;

static LW_STATUS initGpuState( void )
{
    LW_STATUS status = LW_OK;

    gpuState.busyGr = FALSE;
    gpuState.busyMsvld = FALSE;
    gpuState.busyMspdec = FALSE;
    gpuState.busyMsppp = FALSE;
    gpuState.busyCe0 = FALSE;
    gpuState.busyCe1 = FALSE;

    //allocate output buffers
    gpuState.output = (char*)malloc( 65536 );

    if ( gpuState.output == NULL )
    {
        dprintf("lw: ERROR: malloc failed in initGpuState\n");
        status = LW_ERR_GENERIC;
    }
    else {
        sprintf(gpuState.output, "\n ******** ERROR SUMMARY:\t (check above "
            "output for detailed info)  ******** \n");
    }

    gpuState.unitErr = (char*)malloc( 16384 );

    if ( gpuState.unitErr == NULL )
    {
        dprintf("lw: ERROR: malloc failed in initGpuState\n");
        status = LW_ERR_GENERIC;
    }
    else{
        gpuState.unitErr[0] = '\0';
    }

    return status;
}


static void destroyGpuState( void )
{
    if (gpuState.output != NULL)
    {
        free(gpuState.output);
        gpuState.output = NULL;
    }
    if (gpuState.unitErr != NULL)
    {
        free(gpuState.unitErr);
        gpuState.unitErr = NULL;
    }
}


static void printGpuState( void )
{
    if (gpuState.output != NULL)
    {
        dprintf("%s", gpuState.output);
    }
}

//-----------------------------------------------------
// addOutput
//
// Concatenate formatted string to output buffer,
// and concat buffer of per unit errors to output
//-----------------------------------------------------
static void addOutputErrors( const char *format, ... )
{
    char str[1024];
    va_list arglist;

    if ((gpuState.output != NULL) && (gpuState.unitErr != NULL))
    {
        //print fmt to string
        va_start(arglist, format);

        vsprintf(str, format, arglist);

        va_end(arglist);
        //concatenate it
        strcat(gpuState.output, str);

        //add unit errors to output
        strcat(gpuState.output, gpuState.unitErr);

        //clear unit errors
        gpuState.unitErr[0] = '\0';
    }
}



//-----------------------------------------------------
// addUnitErr
//
// Concatenate formatted string to buffer of per unit
// errors
//-----------------------------------------------------
void addUnitErr( const char *format, ...)
{
    char str[1024];
    va_list arglist;

    if (gpuState.unitErr != NULL)
    {
        //print fmt to string
        va_start(arglist, format);

        vsprintf(str, format, arglist);

        va_end(arglist);
        //concatenate it
        strcat(gpuState.unitErr, str);
    }
}


LW_STATUS gpuAnalyze( LwU32 grIdx )
{
    FILE    *fp;
    LW_STATUS ret = LW_OK;

    // Sigdump parameters are hardcoded here. The user should be able to change these.
    int regWriteOptimization = 1;
    int regWriteCheck = 0;
    int markerValuesCheck = 1;
    int verifySigdump = 0;
    int engineStatusVerbose = 0;
    int priCheckVerbose = 0;
    int multiSignalOptimization = 0;
    char *chipletKeyword = NULL;
    char *chipletNumKeyword = NULL;
    char *domainKeyword = NULL;
    char *domainNumKeyword = NULL;
    char *instanceNumKeyword = NULL;
    LW_STATUS    status = LW_OK;

    if ( initGpuState() == LW_ERR_GENERIC )
    {
        dprintf("lw: Error: Unable to init gpu test harness\n");
        return LW_ERR_GENERIC;
    }

    dprintf("\n\tlw: ******** Checking bus interrupts... ********\n");

    if(  LW_OK == pBus[indexGpu].busTestBusInterrupts() )
    {
        dprintf("lw: ******** PASSED: no pending interrupts. ******** \n");
    }
    else
    {
        dprintf("lw: ******** FAILED: interrupts are pending. ******** \n");

        addOutputErrors("Bus interrupts test failed \n");
        status = LW_ERR_GENERIC;
    }

    if (lwMode == MODE_LIVE)
    {
        dprintf("\n\tlw: ******** TLB test... ********\n");

        if(  LW_OK == pFb[indexGpu].fbTestTLB() )
        {
            dprintf("lw: ******** TLB test PASSED. ******** \n");
        }
        else
        {
            dprintf("lw: ******** TLB test FAILED. ******** \n");
            addOutputErrors("TLB test failed\n");

            //bail on error
            printGpuState();
            destroyGpuState();
            return LW_ERR_GENERIC;
        }

        dprintf("\n\tlw: ******** Frame Buffer test... ********\n");

        if(  LW_OK == pFb[indexGpu].fbTest() )
        {
            dprintf("lw: ******** Frame Buffer test PASSED. ******** \n");
        }
        else
        {
            dprintf("lw: ******** Frame Buffer test FAILED. ******** \n");
            addOutputErrors("Frame Buffer test failed\n");

            //bail on error
            printGpuState();
            destroyGpuState();
            return LW_ERR_GENERIC;
        }

        dprintf("\n\tlw: ******** System Memory test... ********\n");

        ret = pFb[indexGpu].fbTestSysmem();

        if(  LW_OK == ret )
        {
            dprintf("lw: ******** System Memory test PASSED. ******** \n");
        }
        else if ( LW_ERR_NOT_SUPPORTED == ret )
        {
            dprintf("lw: ******** System Memory test not supported. ******** \n");
        }
        else
        {
            dprintf("lw: ******** System Memory test FAILED. ******** \n");
            addOutputErrors("System Memory test failed\n");

            //bail on error
            printGpuState();
            destroyGpuState();
            return LW_ERR_GENERIC;
        }
    }

    dprintf("\n\t lw: ******** ELPG state test... ********\n");

    if(  LW_OK == pPmu[indexGpu].pmuTestElpgState() )
    {
        dprintf("\n\t lw: ******** ELPG state test PASSED. ********\n");
    }
    else
    {
        dprintf("\n\t lw: ******** ELPG state test FAILED. ********\n");
        addOutputErrors("ELPG test failed. Reasons: \n");

        //bail on error
        printGpuState();
        destroyGpuState();
        return LW_ERR_GENERIC;
    }

    dprintf("\n\t lw: ******** Host state test... ********\n");

    if(  LW_OK == pFifo[indexGpu].fifoTestHostState() )
    {
        dprintf("lw: ******** Host state test PASSED. ********\n");
    }
    else
    {
        dprintf("lw: ******** Host state test FAILED. ********\n");
        addOutputErrors("Host test failed. Reasons: \n");
        status = LW_ERR_GENERIC;
    }


    if( gpuState.busyGr )
    {
        dprintf("\n\t lw: ******** Graphics state test... ********\n");

        if(  LW_OK == pGr[indexGpu].grTestGraphicsState( grIdx ) )
        {
            dprintf("lw: ******** Graphics state test PASSED. ********\n");
        }
        else
        {
            dprintf("lw: ******** Graphics state test FAILED. ********\n");
            addOutputErrors("Graphics test failed. Reasons: \n");
            status = LW_ERR_GENERIC;

            //sigdump on error
            fp = fopen("sigdump.txt", "w");
            if (fp == NULL)
            {
                dprintf("lw: Unable to open sigdump.txt\n");
            }
            else
            {
                dprintf("lw: sigdump.txt created in the current working directory.\n");
                pSig[indexGpu].sigGetSigdump(fp,
                                             regWriteOptimization,
                                             regWriteCheck,
                                             markerValuesCheck,
                                             verifySigdump,
                                             engineStatusVerbose,
                                             priCheckVerbose,
                                             multiSignalOptimization,
                                             chipletKeyword,
                                             chipletNumKeyword,
                                             domainKeyword,
                                             domainNumKeyword,
                                             instanceNumKeyword);
                fclose(fp);
            }
        }
    }


    if( gpuState.busyMsvld )
    {
        if(  msdecTestMsdecState( 0 ) == LW_ERR_GENERIC )
        {
            addOutputErrors("MSVLD test failed. Reasons:\n");
            status = LW_ERR_GENERIC;
        }
    }

    if( gpuState.busyMspdec )
    {
        if(  msdecTestMsdecState( 1 ) == LW_ERR_GENERIC )
        {
            addOutputErrors("MSPDEC test failed. Reasons:\n");
            status = LW_ERR_GENERIC;
        }
    }

    if( gpuState.busyMsppp )
    {
        if(  msdecTestMsdecState( 2 ) == LW_ERR_GENERIC )
        {
            addOutputErrors("MSPPP test failed. Reasons:\n");
            status = LW_ERR_GENERIC;
        }
    }

    if( gpuState.busyCe0 )
    {
        if(  pCe[indexGpu].ceTestCeState( indexGpu, 0 ) == LW_ERR_GENERIC )
        {
            addOutputErrors("CE0 test failed. Reasons:\n");
            status = LW_ERR_GENERIC;
        }
    }

    if( gpuState.busyCe1 )
    {
        if(  pCe[indexGpu].ceTestCeState( indexGpu, 1 ) == LW_ERR_GENERIC )
        {
            addOutputErrors("CE1 test failed. Reasons:\n");
            status = LW_ERR_GENERIC;
        }
    }

    dprintf("\n\t lw: ******** Display state test... ********\n");
    if(  pDisp[indexGpu].dispTestDisplayState() == LW_ERR_GENERIC )
    {
        dprintf("\n\t lw: ******** Display state test FAILED. ********\n");
        addOutputErrors("Display test failed. Reasons:\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t lw: ******** Display state test PASSED ********\n");
    }

    dprintf("\n\t lw: ******** VIC state test... ********\n");
    if (vicTestState(indexGpu) == LW_ERR_GENERIC)
    {
        dprintf("\n\t lw: ******** VIC state test FAILED. ********\n");
        addOutputErrors("VIC state test failed. Reasons:\n");
        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("\n\t lw: ******** VIC state test PASSED ********\n");
    }
    printGpuState();
    destroyGpuState();

    return status;
}
