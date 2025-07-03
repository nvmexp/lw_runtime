
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/********************************************************************************
*                                                                                *
*    REGISTER PARSE AND MUTEX FUNCTIONS FOR THE LWSR ANALYZE WINDBG EXTENSION    *
*                                                                                *
********************************************************************************/

#include "lwsr_msg.h"
#include "lwHmacSha1.h"

LwU8 lwsr_message[12];  //  message = ((TFC|TFRD) | (R) | (PVER | PVID | PDID))
LwU8 lwsr_digest[20];   //  digest = the result of HMAC-SHA1 computation
LwU8 lwsr_key[16];      //  key = private vendor key

//  FUNCTION TO COMPUTE MUTEX M AND WRITE IT TO REGISTER

RM_LWSR_STATUS lwsrMutexComputeM(LwU32 port, lwsr_mutex_unlock_fields *s_mutexunlock, lwsr_mutex_sprn_field *s_mutexsprn, LwU32 dArgc)
{
    RM_LWSR_STATUS rmStatus = 0;

    LwU8 mutex[4] = {0};
    LwU8 index = 0;

    // read registers
    rmStatus = lwsrReadMutexReg(port,s_mutexunlock,s_mutexsprn);
    if (rmStatus == RM_LWSR_PORT_ERROR)
    {
        dprintf("ERROR : Wrong port...\nERROR : LWSR Register reads failed...\n\n");
        return RM_LWSR_PORT_ERROR;
    }

    // Console welcome message
    dprintf("*********************************************************************************\n");
    dprintf("    Based on LWPU SR Controller Mutex Spec SP-05857-001_v1.0 | Jan 2015\n"    );
    dprintf("*********************************************************************************\n\n");

    dprintf("WARNING : Key entered is treated in little endian order !\n");
    dprintf("WARNING : For example, a key = 0x12AB34CD has the input CD 34 AB 12\n");
    if (dArgc < 18)
        dprintf("WARNING : Key entered is less than 16 bytes long. Unentered bytes set to value 0\n");

    //  Print the accepted key
    dprintf("\nDEBUG   : Key entered : 0x");
    for (index = 0; index < 16; index++)
        dprintf("%02x", lwsr_key[15-index]);
    dprintf("\n\n");

    //  Generate Message key
    lwsrGenerateMessageKey(lwsr_message,s_mutexunlock,s_mutexsprn);

    dprintf("Vendor Key : \n");
    printHexString(lwsr_key, 16);

    //  HMAC-SHA1 computation
    hmacSha1Generate(lwsr_digest, lwsr_message, 12, lwsr_key, 
                 16, lwsrAuthUnlockCopyFunc,
                 lwsrAuthUnlockAllocFunc, lwsrAuthUnlockFreeFunc);

    dprintf("Digest : \n");
    printHexString(lwsr_digest, 20);

    //  Mutex 'M' = the first 4 bytes of digest
    dprintf("Mutex 'M' : \n");
    printHexString(lwsr_digest, 4);

    // Extract Mutex 'M' value from digest for register write
    dprintf("Extracted Mutex 'M' : 0x");
    for (index = 0; index < 4 ; index++)
        mutex[index] = lwsr_digest[index];
    dprintf("%02x", mutex[3]);    
    dprintf("%02x", mutex[2]);
    dprintf("%02x", mutex[1]);
    dprintf("%02x", mutex[0]);

    dprintf("\n");

    // Write the computed mutex into SRMV register
    dprintf("Writing computed mutex 'M' to SRMV register...\n");
    pDpaux[indexGpu].dpauxChWriteMulti(port, 0x3AC, mutex, 4);
    dprintf("SRMV register updated with Mutex 'M'!\n\n");

    // Check if mutex is acquired
    rmStatus = lwsrCheckMutexAcquisition(port);
    if (rmStatus == RM_LWSR_PORT_ERROR)
        return RM_LWSR_PORT_ERROR;

    if (rmStatus == RM_LWSR_MUTEX_FAIL)
        return RM_LWSR_MUTEX_FAIL;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO CHECK IF MUTEX IS ALREADY ACQUIRED AND PANEL IS UNLOCKED

RM_LWSR_STATUS lwsrCheckMutexAcquisition(LwU32 port)
{
    LwU8 cap1_unlock_check_flag = 0;
    RM_AUX_STATUS lwsrStatus = 0;

    // Read Capabilities 1 bit 0 to check if mutex unlock failed or succeeded
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x335, &cap1_unlock_check_flag, 1);
    if (lwsrStatus == 0)
    {
        dprintf("ERROR   : LWSR Cap1 register read failed...\n\n");
        return RM_LWSR_PORT_ERROR;
    }

    if ((cap1_unlock_check_flag & 0x01) == 0)
        return RM_LWSR_MUTEX_FAIL;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO READ LWSR REGISTERS REQUIRED FOR MESSAGE KEY

RM_LWSR_STATUS lwsrReadMutexReg(LwU32 port, lwsr_mutex_unlock_fields *s_mutexunlock, lwsr_mutex_sprn_field *s_mutexsprn)
{
    LwS32 t_sprn = 0;
    LwU8 value[16] = {0};

    RM_AUX_STATUS lwsrStatus = 0;

    // Read PDID, PVID, VERSION;
    // read the 32 bits/4 bytes from 0x3A0
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x3A0, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_mutexunlock->lwsr_mutex_src_id = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    // Read TFC|TFRD;
    // read the 32 bits/4 bytes from 0x3A4
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x3A4, value, 4);
    if (lwsrStatus == 0)
       return RM_LWSR_PORT_ERROR;
    s_mutexunlock->lwsr_mutex_frame_stats = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    // Function call to read SPRN
    t_sprn = lwsrMutexSPRN(port, s_mutexsprn);    
    if (t_sprn == -1)
        return RM_LWSR_PORT_ERROR;

    s_mutexunlock->lwsr_mutex_pdid = (LwU16)(s_mutexunlock->lwsr_mutex_src_id & 0xFFFF);
    s_mutexunlock->lwsr_mutex_pvid = (LwU8)((s_mutexunlock->lwsr_mutex_src_id >> 16) & 0xFF);
    s_mutexunlock->lwsr_mutex_version = (LwU8)((s_mutexunlock->lwsr_mutex_src_id >> 24) & 0xFF);

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT SPRN REGISTER VALUE FOR RANDOMNESS CHECK

LwS32 lwsrMutexSPRN(LwU32 port, lwsr_mutex_sprn_field *s_mutexsprn)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /**************************************
              Interface Mutex 
             DPCD Addr : 003A8h
    **************************************/
    // read the 32 bits/4 bytes from 0x3A8
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x3A8, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_mutexsprn->lwsr_mutex_sprn = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    return s_mutexsprn->lwsr_mutex_sprn;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO GENERATE AND PRINT MESSAGE KEY
void lwsrGenerateMessageKey(LwU8 *pMsgKey, lwsr_mutex_unlock_fields *s_mutexunlock, lwsr_mutex_sprn_field *s_mutexsprn)
{
    memcpy(pMsgKey, &s_mutexunlock->lwsr_mutex_frame_stats, 4);
    memcpy(pMsgKey+4, &s_mutexsprn->lwsr_mutex_sprn, 4);
    memcpy(pMsgKey+8, &s_mutexunlock->lwsr_mutex_pdid, 2);
    memcpy(pMsgKey+10, &s_mutexunlock->lwsr_mutex_pvid, 1);
    memcpy(pMsgKey+11, &s_mutexunlock->lwsr_mutex_version, 1);

    dprintf("Message Key : \n");
    printHexString(pMsgKey, 12);
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT BYTE WISE

void printHexString(LwU8 *pField, LwU32 byteCnt)
{
    LwU32 i;
    for (i = 0; i < byteCnt; i++)
    {
        dprintf("0x%02X", pField[i]);
        if (i != byteCnt - 1)
        {
            dprintf(", ");
        }
        //  Only print 8 bytes a line to keep screen more clear
        if (i % 4 == 3)
        {
            dprintf("\n");
        }
    }
    dprintf("\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO ALIGN MESSAGES ON OUTPUT CONSOLE

void printAlignedString(const char *str, LwU32 alignment)
{
    LwU32 len = 0;
    LwU32 spaces = 0;

    // find title length
    len = (LwU32)strlen(str);
    if (len > alignment)
    {
        LW_ECHO("ERROR : String longer than desired position!");
        LW_ECHO("ERROR : Choose proper alignment!");
    }

    LW_ECHO(" %s", str);
    spaces = alignment - len;
    while(spaces > 0)
    {
        LW_ECHO(" ");
        spaces--;
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------

// 
//  SHA1 library can be used by many different clients, so we need to provide
//  the memory access functions (i.e. following 3 functions) which can work in
//  client's environment
// 
LwU32 lwsrAuthUnlockCopyFunc(LwU8 *pBuff, LwU32 index, LwU32 size, void *pInfo)
{
    LwU8 *pBytes = (LwU8 *)pInfo;
    memcpy(pBuff, pBytes + index, size);
    return size;
}


void * lwsrAuthUnlockAllocFunc(LwU32 size)
{
    return malloc(size);
}


void lwsrAuthUnlockFreeFunc(void *pAddress)
{
    free(pAddress);
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  ENTRY POINT FUNCTION FOR "lwsrcap"

void lwsrcap_entryfunction(LwU32 port)
{
    lwsr_capreg_fields *s_capreg;

    RM_LWSR_STATUS rmStatus = 0;

    // Check if mutex is acquired/unlocked
    rmStatus = lwsrCheckMutexAcquisition(port);
    if (rmStatus == RM_LWSR_PORT_ERROR)
    {
        dprintf("ERROR   : Command lwsrcap failed...\n\n");
        return;
    }

    if (rmStatus == RM_LWSR_MUTEX_FAIL)
    {
        dprintf("ERROR   : AUTHENTICATION FAILED...\n");
        dprintf("ERROR   : EITHER 'PANEL DOES NOT SUPPORT VRR' OR 'LWSR PANEL STILL LOCKED'...\n");
        dprintf("ERROR   : IF LWSR PANEL, PLEASE ACQUIRE LWSR MUTEX & THEN TRY AGAIN...\n\n");
        return;
    }

    dprintf("SUCCESS : Mutex already acquired and panel is unlocked !\n\n");

    // Start exelwtion
    dprintf("DEBUG   : Exelwting command lwsrcap...\n\n");

    s_capreg = (lwsr_capreg_fields *)malloc(sizeof(lwsr_capreg_fields));
    if (s_capreg == NULL)
    {
        dprintf("ERROR   : Memory allocation failed...\n\n");
        return;
    }
    else
    {
        // parse cap regs
        rmStatus = parseLWSRCapabilityRegs(port, s_capreg);
        if (rmStatus == RM_LWSR_PORT_ERROR)
        {
            dprintf("ERROR   : Command lwsrcap failed...\n\n");
            return;
        }
        else
        {
            // print parsed reg state to screen
            echoLWSRCapabilityRegs(s_capreg);
            dprintf("\n");
        }
        //  free memory allocated to struct
        free(s_capreg);
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  ENTRY POINT FUNCTION FOR "lwsrinfo"

void lwsrinfo_entryfunction(LwU32 port, LwU32 verbose_level)
{
    RM_LWSR_STATUS rmStatus = 0;

    // struct pointers
    lwsr_inforeg_fields *s_inforeg = NULL;
    lwsr_controlreg_fields *s_cntrlreg = NULL;
    lwsr_statusreg_fields *s_statusreg = NULL;
    lwsr_linkreg_fields *s_linkreg = NULL;
    lwsr_backlightreg_fields *s_blreg = NULL;
    lwsr_diagnosticreg_fields *s_diagreg = NULL;

    // Check if mutex is acquired/unlocked
    rmStatus = lwsrCheckMutexAcquisition(port);
    if (rmStatus == RM_LWSR_PORT_ERROR)
    {
        dprintf("ERROR   : Command lwsrinfo failed...\n\n");
        return;
    }

    if (rmStatus == RM_LWSR_MUTEX_FAIL)
    {
        dprintf("ERROR   : AUTHENTICATION FAILED...\n");
        dprintf("ERROR   : EITHER 'PANEL DOES NOT SUPPORT VRR' OR 'LWSR PANEL STILL LOCKED'...\n");
        dprintf("ERROR   : IF LWSR PANEL, PLEASE ACQUIRE LWSR MUTEX & THEN TRY AGAIN...\n\n");
        return;
    }

    dprintf("SUCCESS : Mutex already acquired and panel is unlocked !\n\n");

    // Start exelwtion
    dprintf("DEBUG   : Exelwting command lwsrinfo...\n\n");

    // Default verbose level 0
    // control regs
    s_cntrlreg = (lwsr_controlreg_fields *)malloc(sizeof(lwsr_controlreg_fields));
    if (s_cntrlreg == NULL)
    {
        dprintf("ERROR   : Memory allocation failed...\n\n");
        return;
    }
    else
    {
        // parse control regs
        rmStatus = parseLWSRControlRegs(port, s_cntrlreg);
        if (rmStatus == RM_LWSR_PORT_ERROR)
        {
            dprintf("ERROR   : Command lwsrinfo failed...\n\n");
            return;
        }
        else
        {
            // print parsed reg state to screen
            echoLWSRControlRegs(s_cntrlreg);
            dprintf("\n");
        }
    }

    // status regs
    s_statusreg = (lwsr_statusreg_fields *)malloc(sizeof(lwsr_statusreg_fields));
    if (s_statusreg == NULL)
    {
        dprintf("ERROR   : Memory allocation failed...\n\n");
        return;
    }
    else
    {
        // parse status regs
        rmStatus = parseLWSRStatusRegs(port, s_statusreg);
        if (rmStatus == RM_LWSR_PORT_ERROR)
        {
            dprintf("ERROR   : Command lwsrinfo failed...\n\n");
            return;
        }
        else
        {    
            // print parsed reg state to screen
            echoLWSRStatusRegs(s_statusreg, s_cntrlreg);
            dprintf("\n");
        }
    }

    if (verbose_level > 0)
    {
        // link regs
        s_linkreg = (lwsr_linkreg_fields *)malloc(sizeof(lwsr_linkreg_fields));
        if (s_linkreg == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }
        else
        {
            // parse link regs
            rmStatus = parseLWSRLinkRegs(port, s_linkreg);
            if (rmStatus == RM_LWSR_PORT_ERROR)
            {
                dprintf("ERROR   : Command lwsrinfo failed...\n\n");
                return;
            }    
            else
            {
                // print parsed reg state to screen
                echoLWSRLinkRegs(s_linkreg);
                dprintf("\n");
            }
        }

        // backlight regs
        s_blreg = (lwsr_backlightreg_fields *)malloc(sizeof(lwsr_backlightreg_fields));
        if (s_blreg == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }
        else
        {
            // parse backlight regs
            rmStatus = parseLWSRBacklightRegs(port, s_blreg);
            if (rmStatus == RM_LWSR_PORT_ERROR)
            {
                dprintf("ERROR   : Command lwsrinfo failed...\n\n");
                return;
            }
            else
            {
                // print parsed reg state to screen
                echoLWSRBacklightRegs(s_blreg);
                dprintf("\n");
            }
        }

        // diagnostic regs
        s_diagreg = (lwsr_diagnosticreg_fields *)malloc(sizeof(lwsr_diagnosticreg_fields));
        if (s_diagreg == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n");
            return;
        }
        else
        {
            // parse diagnostic regs
            rmStatus = parseLWSRDiagnosticRegs(port, s_diagreg);
            if (rmStatus == RM_LWSR_PORT_ERROR)
            {
                dprintf("ERROR   : Command lwsrinfo failed...\n\n");
                return;
            }
            else
            {
                // print parsed reg state to screen
                echoLWSRDiagnosticRegs(s_diagreg);
                dprintf("\n");
            }
        }

        // basic src regs
        s_inforeg = (lwsr_inforeg_fields *)malloc(sizeof(lwsr_inforeg_fields));
        if (s_inforeg == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }
        else
        {
            // parse frame stat regs
            rmStatus = parseLWSRInfoRegs(port, s_inforeg);
            if (rmStatus == RM_LWSR_PORT_ERROR)
            {
                dprintf("ERROR   : Command lwsrinfo failed...\n\n");
                return;
            }
            else
            {
                // print parsed reg state to screen
                echoLWSRInfoRegs(s_inforeg);
                dprintf("\n");
            }
        }
    }

    // free memory allocated to any structs
    if (s_inforeg)
        free(s_inforeg);
    if (s_cntrlreg)
        free(s_cntrlreg);
    if (s_statusreg)
        free(s_statusreg);
    if (s_linkreg)
        free(s_linkreg);
    if (s_blreg)
        free(s_blreg);
    if (s_diagreg)
        free(s_diagreg);
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  ENTRY POINT FUNCTION FOR "lwsrtiming"

void lwsrtiming_entryfunction(LwU32 port)
{
    RM_LWSR_STATUS rmStatus = 0;

    // struct pointers
    lwsr_timingreg_fields *s_timingreg = NULL;

    // Check if mutex is acquired/unlocked
    rmStatus = lwsrCheckMutexAcquisition(port);
    if (rmStatus == RM_LWSR_PORT_ERROR)
    {
        dprintf("ERROR   : Command lwsrtiming failed...\n\n");
        return;
    }

    if (rmStatus == RM_LWSR_MUTEX_FAIL)
    {
        dprintf("ERROR   : AUTHENTICATION FAILED...\n");
        dprintf("ERROR   : EITHER 'PANEL DOES NOT SUPPORT VRR' OR 'LWSR PANEL STILL LOCKED'...\n");
        dprintf("ERROR   : IF LWSR PANEL, PLEASE ACQUIRE LWSR MUTEX & THEN TRY AGAIN...\n\n");
        return;
    }

    dprintf("SUCCESS : Mutex already acquired and panel is unlocked !\n\n");

    // Start exelwtion
    dprintf("DEBUG   : Exelwting command lwsrtiming...\n\n");

    // timing regs
    s_timingreg = (lwsr_timingreg_fields *)malloc(sizeof(lwsr_timingreg_fields));
    if (s_timingreg == NULL)
    {
        dprintf("ERROR   : Memory allocation failed...\n\n");
        return;
    }
    else
    {
        // parse timing regs
        rmStatus = parseLWSRTimingRegs(port, s_timingreg);
        if (rmStatus == RM_LWSR_PORT_ERROR)
        {
            dprintf("ERROR   : Command lwsrtiming failed...\n\n");
            return;
        }    
        else
        {
            // print parsed reg state to screen
            echoLWSRTimingRegs(s_timingreg);
            dprintf("\n");
        }
    }

    // free memory allocated to any structs
    if (s_timingreg)
        free(s_timingreg);
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  ENTRY POINT FUNCTION FOR "lwsrmutex"

void lwsrmutex_entryfunction(LwU32 port, LwU32 sub_function_option, LwU32 dArgc)
{
    LwS32 sprn = 0;
    LwU8 index = 0;
    RM_LWSR_STATUS rmStatus = 0;

    char *string1 = "lwsrMutexSPRN()";
    char *string2 = "lwsrMutexUnlock()";

    lwsr_mutex_sprn_field *s_mutexsprn;
    lwsr_mutex_unlock_fields *s_mutexunlock;

    // Subfunction 1 processing
    if (sub_function_option== 1)
    {    
        if (dArgc > 2)
        {
            dprintf("Too many arguments\n");
            dprintf("Call !lw.help for a command list\n\n");
            return;
        }

        dprintf("DEBUG   : Exelwting lwsrmutex function %s...\n\n",string1);
        s_mutexsprn = (lwsr_mutex_sprn_field *)malloc(sizeof(lwsr_mutex_sprn_field));
        if (s_mutexsprn == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }
        else
        {
            for(index = 0 ; index < 3; index++)
            {
                // Read Client Value 'R'
                sprn = lwsrMutexSPRN(port, s_mutexsprn);
                if (sprn == (LwS32)RM_LWSR_PORT_ERROR)
                {
                    dprintf("ERROR   : Wrong port...\nERROR   : lwsrmutex function %s failed...\n\n",string1);
                    return;
                }
                else
                {
                    if (index == 0)
                    {
                        dprintf("*********************************************************************************\n");
                        dprintf("    Based on LWPU SR Controller Mutex Spec SP-05857-001_v1.0 | Jan 2015\n"    );
                        dprintf("*********************************************************************************\n\n");
                        dprintf("Reading the SPRN value 3 times...\n");
                    }
                    dprintf("SPRN : Client Value 'R' = 0x%x\n",sprn);
                    sprn = 0;
                }
            }
            dprintf("\n");
        }
        if (s_mutexsprn)
            free(s_mutexsprn);
    }

    // Subfunction 2 processing
    else 
    {
        dprintf("DEBUG   : Exelwting lwsrmutex function %s...\n\n",string2);
        s_mutexsprn = (lwsr_mutex_sprn_field *)malloc(sizeof(lwsr_mutex_sprn_field));
        if (s_mutexsprn == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }

        s_mutexunlock = (lwsr_mutex_unlock_fields *)malloc(sizeof(lwsr_mutex_unlock_fields));
        if (s_mutexunlock == NULL)
        {
            dprintf("ERROR   : Memory allocation failed...\n\n");
            return;
        }

        // Compute mutex 'M' and write to register
        rmStatus = lwsrMutexComputeM(port,s_mutexunlock,s_mutexsprn,dArgc);
        if (rmStatus == RM_LWSR_PORT_ERROR)
        {
            dprintf("ERROR   : Mutex 'M' computation failed...\nERROR   : lwsrmutex function %s failed...\n\n",string2);
            return;
        }

        if (rmStatus == RM_LWSR_MUTEX_FAIL)
        {
            dprintf("ERROR   : MUTEX MISMATCH (M != M')...\nERROR   : FAILED TO ACQUIRE THE MUTEX...\n");
            dprintf("ERROR   : AUTHENTICATION FAILED...\n\n");
            return;
        }
        else
        {
            dprintf("SUCCESS : MUTEX MATCH (M = M')...\nSUCCESS : MUTEX IS ACQUIRED...\n");
            dprintf("SUCCESS : AUTHENTICATION SUCCESSFUL !!!\n\n");
        }

        // free allocated memory
        if (s_mutexsprn)
            free(s_mutexsprn);
        if (s_mutexunlock)
            free(s_mutexunlock);

        // clear key before leaving so that subfunction can be reused IN SAME DEBUG SESSION
        for(index = 0 ; index < 16; index++)
            lwsr_key[index] = 0;
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  ENTRY POINT FUNCTION FOR "lwsrsetrr"

RM_LWSR_STATUS lwsrsetrr_entryfunction(LwU32 port, LwU32 refresh_rate)
{
    LwU8 value[16] = {0};
    LwU16 Htotal, Vblank_new, Vfporch_new;
    LwU8 vblank[2] = {0}, vfporch[2] = {0};
    lwsr_timingreg_fields *s_timingreg = NULL;
    RM_AUX_STATUS lwsrStatus = 0;

    s_timingreg = (lwsr_timingreg_fields *)malloc(sizeof(lwsr_timingreg_fields));
    if (s_timingreg == NULL)
    {
        dprintf("ERROR   : Memory allocation failed...\n\n");
        return -1;
    }
    else
    {
        // Read SRC to panel pclk (Hz)
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x350, value, 2);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_srpc = (LwU16)value[0] | (LwU16)value[1]<<8;

        s_timingreg->lwsr_src_panel_srpc = (LwF32)s_timingreg->lwsr_selfrefresh_srpc*20000;

        // Read Vactive
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x35C, value, 2);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_srva = (LwU16)value[0] | (LwU16)value[1]<<8;

        // Read Hactive and Hblank
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x352, value, 2);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR; 
        s_timingreg->lwsr_selfrefresh_srha = (LwU16)value[0] | (LwU16)value[1]<<8;

        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x354, value, 2);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_srhbl = (LwU16)value[0] | (LwU16)value[1]<<8;

        // Callwlate Htotal
        Htotal = s_timingreg->lwsr_selfrefresh_srha + s_timingreg->lwsr_selfrefresh_srhbl;

        // Callwlate new Vblank
        Vblank_new = (LwU16)(s_timingreg->lwsr_src_panel_srpc/(refresh_rate*Htotal) - s_timingreg->lwsr_selfrefresh_srva);
        vblank[0] = Vblank_new & 0xFF;
        vblank[1] = Vblank_new >> 8;
        dprintf("\nNew Vblank = %d lines\n", (LwU32)Vblank_new);

        // Adjust vertical front porch accordingly
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x362, value, 2);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_srvbp = (LwU16)value[0] | (LwU16)value[1]<<8;

        // read the 8 bits/1 bytes from 0x364
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x364, value, 1);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_srvs = value[0];

        // read the 8 bits/1 bytes from 0x365
        lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x365, value, 1);
        if (lwsrStatus == 0)
            return RM_LWSR_PORT_ERROR;
        s_timingreg->lwsr_selfrefresh_365h = value[0];

        s_timingreg->lwsr_selfrefresh_srvb = s_timingreg->lwsr_selfrefresh_365h & 0xEF;    // / in lines

        Vfporch_new = (LwU16)(Vblank_new - s_timingreg->lwsr_selfrefresh_srvbp - s_timingreg->lwsr_selfrefresh_srvs - s_timingreg->lwsr_selfrefresh_srvb);
        vfporch[0] = Vfporch_new & 0xFF;
        vfporch[1] = Vfporch_new >> 8;

        // Write the computed Vblank and adjusted vertical front porch
        dprintf("Writing new computed vertical blank and front porch...\n");
        pDpaux[indexGpu].dpauxChWriteMulti(port, 0x35E, vblank, 1);
        pDpaux[indexGpu].dpauxChWriteMulti(port, 0x360, vfporch, 1);
        dprintf("Refresh rate updated to user request! \n\n");

        if (s_timingreg)
            free(s_timingreg);
        return 0;
    }
}
// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR INFO REGISTERS

RM_LWSR_STATUS parseLWSRInfoRegs(LwU32 port, lwsr_inforeg_fields *s_inforeg)
{
     LwU8 value[16] = {0};
     RM_AUX_STATUS lwsrStatus = 0;

     /**************************************
              Frame statistics 
             DPCD Addr : 003A4h
    **************************************/
    // read the 32 bits/4 bytes from 0x3A4
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x3A4, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_inforeg->lwsr_frame_stats = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_inforeg->lwsr_tfc = s_inforeg->lwsr_frame_stats & 0x7fffffff;    // / num of frames
    s_inforeg->lwsr_tfrd = (LwU8)(s_inforeg->lwsr_frame_stats >> 31 & 0x01);

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR CAPABILITY REGISTERS

RM_LWSR_STATUS parseLWSRCapabilityRegs(LwU32 port, lwsr_capreg_fields *s_capreg)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /*********************************
                   SRC ID 
           DPCD Addr : 003A0h
    *********************************/
    // read the 32 bits/4 bytes from 0x3A0
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x3A0, value, 4);
    if (lwsrStatus == 0)
       return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_src_id = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_capreg->lwsr_pdid = (LwU16)(s_capreg->lwsr_src_id & 0xFFFF);
    s_capreg->lwsr_pvid = (LwU8)((s_capreg->lwsr_src_id >> 16) & 0xFF);
    s_capreg->lwsr_version = (LwU8)((s_capreg->lwsr_src_id >> 24) & 0xFF);

    /***********************************
             NLT Capabilities
        DPCD Addr : 00330h ~ 00333h
    ************************************/
    // read the 32 bits/4 bytes from 0x330
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x330, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_cap0 = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_capreg->lwsr_nltc = (LwU8)(s_capreg->lwsr_cap0 & 0x01);
    s_capreg->lwsr_ccdn = (LwU8)(s_capreg->lwsr_cap0 & 0x02);
    s_capreg->lwsr_ncc = (LwU8)(s_capreg->lwsr_cap0 & 0x04);
    s_capreg->lwsr_mlir = ((LwF32)(s_capreg->lwsr_cap0 >> 8));   // / time in microsecs    

    /*************************
           Capabilities1
         DPCD Addr : 00335h
    **************************/
    // read the 8 bits/1 bytes from 0x335
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x335, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_cap1 = value[0];

    s_capreg->lwsr_srcc = s_capreg->lwsr_cap1 & 0x01;
    s_capreg->lwsr_ser = s_capreg->lwsr_cap1 & 0x06;
    s_capreg->lwsr_sxr = s_capreg->lwsr_cap1 & 0x18;
    s_capreg->lwsr_srcp = s_capreg->lwsr_cap1 & 0xE0;

    /*************************
          Capabilities2
         DPCD Addr : 00336h
    **************************/
    // read the 8 bits/1 bytes from 0x336
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x336, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_cap2 = value[0];

    s_capreg->lwsr_sel = s_capreg->lwsr_cap2 & 0x0F;
    s_capreg->lwsr_spd = s_capreg->lwsr_cap2 & 0x10;
    s_capreg->lwsr_std = s_capreg->lwsr_cap2 & 0x20;
    s_capreg->lwsr_s3d = s_capreg->lwsr_cap2 & 0x40;
    s_capreg->lwsr_scl = s_capreg->lwsr_cap2 & 0x80;

    /*************************
           Capabilities3
        DPCD Addr : 00337h
    **************************/
    // read the 8 bits/1 bytes from 0x337
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x337, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_cap3 = value[0];

    s_capreg->lwsr_sspc = s_capreg->lwsr_cap3 & 0x01;
    s_capreg->lwsr_srbr = s_capreg->lwsr_cap3 & 0x02;
    s_capreg->lwsr_srfc = s_capreg->lwsr_cap3 & 0x04;
    s_capreg->lwsr_srcf = s_capreg->lwsr_cap3 & 0x08;
    s_capreg->lwsr_srec = s_capreg->lwsr_cap3 & 0x10;
    s_capreg->lwsr_sred = s_capreg->lwsr_cap3 & 0x20;
    s_capreg->lwsr_srss = s_capreg->lwsr_cap3 & 0x40;
    s_capreg->lwsr_srgc = s_capreg->lwsr_cap3 & 0x80;

    /*************************       
            Buffer cap
        DPCD Addr : 00338h
    **************************/
    // read the 8 bits/1 bytes from 0x338
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x338, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_bufcap = value[0];

    s_capreg->lwsr_srbs = ((LwF32)s_capreg->lwsr_bufcap+1)*100/1000;    // / size in MB

    /*****************************************
               Max pixel clk
        DPCD Addr : 00339h~0033Ah
    ****************************************/
    // read the 16 bits/2 bytes from 0x338
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x339, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_max_pixel_clk = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    s_capreg->lwsr_smpc = (LwF32)s_capreg->lwsr_max_pixel_clk*20/1000;    // / max clk supported in MHz

    /***************************************    
             Cache latency
        DPCD Addr : 0033Bh~0033Dh
    ***************************************/
    // read the 16 bits/2 bytes from 0x33B
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x33B, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_srcl = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    /***************************************
          Burst mode Cache latency
        DPCD Addr : 0033Dh ~ 0033Eh
    *****************************************/
    // read the 16 bits/2 bytes from 0x33D
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x33D, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_srbl = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    /************************************    
         Capabilities 4
         DPCD Addr : 0033Fh
    ************************************/
    // read the 8 bits/1 bytes from 0x33F
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x33F, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_capreg->lwsr_cap4 = value[0]; 

    s_capreg->lwsr_srlw_support = s_capreg->lwsr_cap4 & 0x01;
    s_capreg->lwsr_srao_support = s_capreg->lwsr_cap4 & 0x02;
    s_capreg->lwsr_scsc_support = s_capreg->lwsr_cap4 & 0x1C;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR CONTROL REGISTERS

RM_LWSR_STATUS parseLWSRControlRegs(LwU32 port, lwsr_controlreg_fields *s_cntrlreg)
{
    LwU8 value[2] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /********************************
             NLT TRANSITION
              DPCD Addr : 00334h
    ********************************/
    // read the 8 bits/1 bytes from 0x334
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x334, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_nlt_trans = value[0];

    s_cntrlreg->lwsr_nlts = s_cntrlreg->lwsr_nlt_trans & 0x01;

    /*************************
          SRC CONTROL
         DPCD Addr : 00340h
    **************************/
    // read the 8 bits/1 bytes from 0x340
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x340, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_src_control = value[0];

    s_cntrlreg->lwsr_sec = s_cntrlreg->lwsr_src_control & 0x01;
    s_cntrlreg->lwsr_senm = s_cntrlreg->lwsr_src_control & 0x02;
    s_cntrlreg->lwsr_srrd_disable = s_cntrlreg->lwsr_src_control & 0x04;
    s_cntrlreg->lwsr_srm = s_cntrlreg->lwsr_src_control & 0x08;
    s_cntrlreg->lwsr_ser = s_cntrlreg->lwsr_src_control & 0x10;
    s_cntrlreg->lwsr_seym = s_cntrlreg->lwsr_src_control & 0x20;
    s_cntrlreg->lwsr_sexr = s_cntrlreg->lwsr_src_control & 0x40;
    s_cntrlreg->lwsr_sexm = s_cntrlreg->lwsr_src_control & 0x80;

    /*************************
          MISC CONTROL1
         DPCD Addr : 00341h
    **************************/
    // read the 8 bits/1 bytes from 0x341
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x341, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_misc_control1 = value[0];

    s_cntrlreg->lwsr_gstl = s_cntrlreg->lwsr_misc_control1 & 0x01;
    s_cntrlreg->lwsr_gstm = s_cntrlreg->lwsr_misc_control1 & 0x02;
    s_cntrlreg->lwsr_sptl = s_cntrlreg->lwsr_misc_control1 & 0x04;
    s_cntrlreg->lwsr_sptm = s_cntrlreg->lwsr_misc_control1 & 0x08;
    s_cntrlreg->lwsr_sese = s_cntrlreg->lwsr_misc_control1 & 0x10;
    s_cntrlreg->lwsr_sesm = s_cntrlreg->lwsr_misc_control1 & 0x20;
    s_cntrlreg->lwsr_sxse = s_cntrlreg->lwsr_misc_control1 & 0x40;
    s_cntrlreg->lwsr_sxsm = s_cntrlreg->lwsr_misc_control1 & 0x80;

    /*******************************
          MISC CONTROL2
         DPCD Addr : 00342h
    ********************************/
    // read the 8 bits/1 bytes from 0x342
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x342, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_misc_control2 = value[0];

    s_cntrlreg->lwsr_scnf = s_cntrlreg->lwsr_misc_control2 & 0x0F;
    s_cntrlreg->lwsr_s3ds = s_cntrlreg->lwsr_misc_control2 & 0x10;
    s_cntrlreg->lwsr_sbse = s_cntrlreg->lwsr_misc_control2 & 0x20;
    s_cntrlreg->lwsr_sbfe = s_cntrlreg->lwsr_misc_control2 & 0x40;
    s_cntrlreg->lwsr_ssde = s_cntrlreg->lwsr_misc_control2 & 0x80;

    /*************************************
          Interrupt mask
         DPCD Addr : 00343h
    ************************************/
    // read the 8 bits/1 bytes from 0x343
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x343, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_intrpt_mask = value[0];

    s_cntrlreg->lwsr_isef = s_cntrlreg->lwsr_intrpt_mask & 0x01;
    s_cntrlreg->lwsr_isem = s_cntrlreg->lwsr_intrpt_mask & 0x02;
    s_cntrlreg->lwsr_isxd = s_cntrlreg->lwsr_intrpt_mask & 0x04;
    s_cntrlreg->lwsr_isxm = s_cntrlreg->lwsr_intrpt_mask & 0x08;
    s_cntrlreg->lwsr_isbo = s_cntrlreg->lwsr_intrpt_mask & 0x10;
    s_cntrlreg->lwsr_isbm = s_cntrlreg->lwsr_intrpt_mask & 0x20;
    s_cntrlreg->lwsr_isvb = s_cntrlreg->lwsr_intrpt_mask & 0x40;
    s_cntrlreg->lwsr_isvm = s_cntrlreg->lwsr_intrpt_mask & 0x80;

    /**********************************
          Interrupt enable
         DPCD Addr : 00344h
    ***********************************/
    // read the 8 bits/1 bytes from 0x344
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x344, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_intrpt_enable = value[0];

    s_cntrlreg->lwsr_ieac = s_cntrlreg->lwsr_intrpt_enable & 0x01;
    s_cntrlreg->lwsr_ieam = s_cntrlreg->lwsr_intrpt_enable & 0x02;
    s_cntrlreg->lwsr_iscd = s_cntrlreg->lwsr_intrpt_enable & 0x04;
    s_cntrlreg->lwsr_iscm = s_cntrlreg->lwsr_intrpt_enable & 0x08;
    s_cntrlreg->lwsr_isnd = s_cntrlreg->lwsr_intrpt_enable & 0x10;
    s_cntrlreg->lwsr_isnm = s_cntrlreg->lwsr_intrpt_enable & 0x20;
    s_cntrlreg->lwsr_isxe = s_cntrlreg->lwsr_intrpt_enable & 0x40;
    s_cntrlreg->lwsr_extension_isxm = s_cntrlreg->lwsr_intrpt_enable & 0x80;    

    /*************************************
              DPCD Addr : 00345h
    *************************************/
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x345, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_345h = value[0];

    s_cntrlreg->lwsr_ispe = s_cntrlreg->lwsr_345h & 0x01;
    s_cntrlreg->lwsr_ispm = s_cntrlreg->lwsr_345h & 0x02;

    /*************************************
        RESYNC Control1
         DPCD Addr : 00346h
    *************************************/
    // read the 8 bits/1 bytes from 0x346
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x346, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_resync1 = value[0];

    s_cntrlreg->lwsr_srrm = s_cntrlreg->lwsr_resync1 & 0x07;
    s_cntrlreg->lwsr_srcf = s_cntrlreg->lwsr_resync1 & 0x08;
    s_cntrlreg->lwsr_res2 = s_cntrlreg->lwsr_resync1 & 0x30;
    s_cntrlreg->lwsr_srrd_delay = s_cntrlreg->lwsr_resync1 & 0xC0;    

    /*************************************
        RESYNC Control2
         DPCD Addr : 00347h
    *************************************/
    // read the 8 bits/1 bytes from 0x347
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x347, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_cntrlreg->lwsr_resync2 = value[0];

    s_cntrlreg->lwsr_srfe = s_cntrlreg->lwsr_resync2 & 0x03;
    s_cntrlreg->lwsr_res3 = s_cntrlreg->lwsr_resync2 & 0xFC;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR STATUS REGISTERS

RM_LWSR_STATUS parseLWSRStatusRegs(LwU32 port, lwsr_statusreg_fields *s_statusreg)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /**************************************
           SRC Status1
         DPCD Addr : 00348h
    ***************************************/
    // read the 8 bits/1 bytes from 0x348
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x348, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_statusreg->lwsr_src_status1 = value[0];

    s_statusreg->lwsr_srst = s_statusreg->lwsr_src_status1 & 0x07;
    s_statusreg->lwsr_srsf = s_statusreg->lwsr_src_status1 & 0x18;
    s_statusreg->lwsr_srbs = s_statusreg->lwsr_src_status1 & 0x60;
    s_statusreg->lwsr_srbo = s_statusreg->lwsr_src_status1 & 0x80;

    /**************************************
             SRC Status2
           DPCD Addr : 00349h
    ***************************************/
    // read the 8 bits/1 bytes from 0x349
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x349, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_statusreg->lwsr_src_status2 = value[0];

    s_statusreg->lwsr_sint = s_statusreg->lwsr_src_status2 & 0x01;
    s_statusreg->lwsr_sinm = s_statusreg->lwsr_src_status2 & 0x02;
    s_statusreg->lwsr_scom = s_statusreg->lwsr_src_status2 & 0x04;
    s_statusreg->lwsr_srcv = s_statusreg->lwsr_src_status2 & 0x18;
    s_statusreg->lwsr_srcr = s_statusreg->lwsr_src_status2 & 0x20;
    s_statusreg->lwsr_srct = s_statusreg->lwsr_src_status2 & 0x40;

    /*************************************
              SRC Status3
            DPCD Addr : 0034Ah
    *************************************/
    // read the 8 bits/1 bytes from 0x34A
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x34A, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_statusreg->lwsr_src_status3 = value[0];

    s_statusreg->lwsr_srs4 = s_statusreg->lwsr_src_status3 & 0x03;
    s_statusreg->lwsr_sps4 = s_statusreg->lwsr_src_status3 & 0x40;
    s_statusreg->lwsr_spst = s_statusreg->lwsr_src_status3 & 0x80;

    /*************************
        Interrupt Status
         DPCD Addr : 0034Bh
    **************************/
    // read the 8 bits/1 bytes from 0x34B
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x34B, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_statusreg->lwsr_interrupt_status = value[0];

    s_statusreg->lwsr_bit0 = s_statusreg->lwsr_interrupt_status & 0x01;
    s_statusreg->lwsr_bit1 = s_statusreg->lwsr_interrupt_status & 0x02;
    s_statusreg->lwsr_bit2 = s_statusreg->lwsr_interrupt_status & 0x04;
    s_statusreg->lwsr_bit3 = s_statusreg->lwsr_interrupt_status & 0x08;
    s_statusreg->lwsr_bit4 = s_statusreg->lwsr_interrupt_status & 0x10;
    s_statusreg->lwsr_bit5 = s_statusreg->lwsr_interrupt_status & 0x20;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR TIMING REGISTERS

RM_LWSR_STATUS parseLWSRTimingRegs(LwU32 port, lwsr_timingreg_fields *s_timingreg)
{
    RM_AUX_STATUS lwsrStatus = 0;
    LwU8 value[16] = {0};

    /*************************
        SR mode timing
     DPCD Addr : 00350h~00367h
    **************************/
    // read the 16 bits/2 bytes from 0x350
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x350, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srpc = (LwU16)value[0] | (LwU16)value[1]<<8;

    s_timingreg->lwsr_src_panel_srpc = (LwF32)s_timingreg->lwsr_selfrefresh_srpc*20/1000;    // / pixel clk supported in MHz

    // read the 16 bits/2 bytes from 0x352
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x352, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR; 
    s_timingreg->lwsr_selfrefresh_srha = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x354
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x354, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srhbl = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x356
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x356, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srhfp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x358
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x358, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srhbp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 8 bits/1 bytes from 0x35A
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x35A, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srhs = value[0];

    // read the 8 bits/1 bytes from 0x35B
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x35B, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_35Bh = value[0];

    s_timingreg->lwsr_selfrefresh_srhb = s_timingreg->lwsr_selfrefresh_35Bh & 0xEF;    // / in pixels
    s_timingreg->lwsr_selfrefresh_srhsp = s_timingreg->lwsr_selfrefresh_35Bh & 0x80;

    // read the 16 bits/2 bytes from 0x35C
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x35C, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srva = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x35E
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x35E, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srvbl = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x360
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x360, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srvfp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x362
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x362, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srvbp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 8 bits/1 bytes from 0x364
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x364, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_srvs = value[0];

    // read the 8 bits/1 bytes from 0x365
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x365, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_365h = value[0];

    s_timingreg->lwsr_selfrefresh_srvb = s_timingreg->lwsr_selfrefresh_365h & 0xEF;    // / in lines
    s_timingreg->lwsr_selfrefresh_srvsp = s_timingreg->lwsr_selfrefresh_365h & 0x80;

    // read the 8 bits/1 bytes from 0x366
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x366, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_selfrefresh_366h = value[0];

    s_timingreg->lwsr_selfrefresh_srcs = s_timingreg->lwsr_selfrefresh_366h & 0x07;
    s_timingreg->lwsr_selfrefresh_srfp = s_timingreg->lwsr_selfrefresh_366h & 0x30;

    // Callwlate SRC-Panel refresh rate
    // Refresh rate = pclk / [(v_active+v_blank)(h_active+h_blank)]
    s_timingreg->lwsr_src_panel_vtotal = s_timingreg->lwsr_selfrefresh_srva + s_timingreg->lwsr_selfrefresh_srvbl;
    s_timingreg->lwsr_src_panel_htotal = s_timingreg->lwsr_selfrefresh_srha + s_timingreg->lwsr_selfrefresh_srhbl;
    s_timingreg->lwsr_src_panel_refreshrate = (s_timingreg->lwsr_src_panel_srpc*1000000)/(s_timingreg->lwsr_src_panel_vtotal*s_timingreg->lwsr_src_panel_htotal); //in hertz

    /*********************************************
          PASS THRU MODE TIMING
         DPCD Addr : 00368h~0037Fh
    **********************************************/
    // read the 16 bits/2 bytes from 0x368
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x368, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_srpc = (LwU16)value[0] | (LwU16)value[1]<<8;

    s_timingreg->lwsr_gpu_src_srpc = (LwF32)s_timingreg->lwsr_passthrough_srpc*20/1000;    // / pixel clk supported in MHz

    // read the 16 bits/2 bytes from 0x36A
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x36A, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptha = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x36C
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x36C, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_pthbl = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x36E
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x36E, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_pthfp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 16 bits/2 bytes from 0x370
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x370, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_pthbp = (LwU16)value[0] | (LwU16)value[1]<<8;

    // read the 8 bits/1 bytes from 0x372
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x372, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_pths = value[0];

    // read the 8 bits/1 bytes from 0x373
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x373, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_373h = value[0];

    s_timingreg->lwsr_passthrough_pthb = s_timingreg->lwsr_passthrough_373h & 0xEF;    // / in pixels
    s_timingreg->lwsr_passthrough_pthsp = s_timingreg->lwsr_passthrough_373h & 0x80;

    // read the 16 bits/2 bytes from 0x374
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x374, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptva = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x376
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x376, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptvbl = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x378
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x378, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptvfp = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x37A
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x37A, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptvbp = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 8 bits/1 bytes from 0x37C
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x37C, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_ptvs = value[0];

    // read the 8 bits/1 bytes from 0x37D
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x37D, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_passthrough_37Dh = value[0];

    s_timingreg->lwsr_passthrough_ptvb = s_timingreg->lwsr_passthrough_37Dh & 0xEF;    // / in lines
    s_timingreg->lwsr_passthrough_ptvsp = s_timingreg->lwsr_passthrough_37Dh & 0x80;

    /***************************************
              Blank timing limits
            DPCD Addr : 00388h~0038Eh
    ****************************************/
    // read the 16 bits/2 bytes from 0x388
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x388, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_vbmn = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x38A
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x38A, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_vbmx = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x38C
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x38C, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_hbmn = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    // read the 16 bits/2 bytes from 0x38E
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x38E, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_timingreg->lwsr_hbmx = (LwU16)value[0] | (LwU16)value[1]<<8 ;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR LINK REGISTERS

RM_LWSR_STATUS parseLWSRLinkRegs(LwU32 port, lwsr_linkreg_fields *s_linkreg)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /**********************************
          LINK INTERFACE gpu-src
             DPCD Addr : 00380h
    **********************************/
    // read the 8 bits/1 bytes from 0x380
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x380, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_linkreg->lwsr_link_gpu_src = value[0];

    s_linkreg->lwsr_lgsl = s_linkreg->lwsr_link_gpu_src & 0x0F;
    s_linkreg->lwsr_lgsg = s_linkreg->lwsr_link_gpu_src & 0x10;
    s_linkreg->lwsr_lgspf = s_linkreg->lwsr_link_gpu_src & 0xE0;

    /***************************************
          LINK INTERFACE src-panel1
               DPCD Addr : 00381h
    ***************************************/
    // read the 8 bits/1 bytes from 0x381
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x381, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_linkreg->lwsr_link_src_panel1 = value[0];

    s_linkreg->lwsr_lspf = s_linkreg->lwsr_link_src_panel1 & 0x07;

    /*************************************
          LINK INTERFACE src-panel2
               DPCD Addr : 00382h
    *************************************/
    // read the 8 bits/1 bytes from 0x382
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x382, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_linkreg->lwsr_link_src_panel2 = value[0];

    s_linkreg->lwsr_lsvc = s_linkreg->lwsr_link_src_panel2 & 0x07; // /in columns
    s_linkreg->lwsr_lshr = s_linkreg->lwsr_link_src_panel2 & 0x38;

    /************************************
             LINK INTERFACE type
               DPCD Addr : 00383h
    ************************************/
    // read the 8 bits/1 bytes from 0x383
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x383, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_linkreg->lwsr_link_type = value[0];

    s_linkreg->lwsr_ltyp_gpu_src   = s_linkreg->lwsr_link_type & 0x0F;
    s_linkreg->lwsr_ltyp_src_panel = s_linkreg->lwsr_link_type & 0xF0;

    /**************************************
                  LINK control
                DPCD Addr : 00384h
    ***************************************/
    // read the 8 bits/1 bytes from 0x384
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x384, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_linkreg->lwsr_link_control = value[0];

    s_linkreg->lwsr_lgss = s_linkreg->lwsr_link_control & 0x0F;
    s_linkreg->lwsr_lsps = s_linkreg->lwsr_link_control & 0xF0;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR BACKLIGHT REGISTERS

RM_LWSR_STATUS parseLWSRBacklightRegs(LwU32 port, lwsr_backlightreg_fields *s_blreg)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /****************************************
            BACKLIGHT CAPABILITY #1
               DPCD Addr : 00701h
    ****************************************/
    // read the 8 bits/1 bytes from 0x701
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x701, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_blcap1 = value[0];

    s_blreg->lwsr_bl_adjustment_capable = s_blreg->lwsr_blcap1 & 0x01;
    s_blreg->lwsr_bl_pin_en_capable = s_blreg->lwsr_blcap1 & 0x02;
    s_blreg->lwsr_bl_aux_en_capable = s_blreg->lwsr_blcap1 & 0x04;
    s_blreg->lwsr_pslftst_pin_en_capable = s_blreg->lwsr_blcap1 & 0x08;
    s_blreg->lwsr_pslftst_aux_en_capable = s_blreg->lwsr_blcap1 & 0x10;
    s_blreg->lwsr_frc_en_capable = s_blreg->lwsr_blcap1 & 0x20;
    s_blreg->lwsr_color_eng_capable = s_blreg->lwsr_blcap1 & 0x40;
    s_blreg->lwsr_set_pwr_capable = s_blreg->lwsr_blcap1 & 0x80;

    /****************************************
            BACKLIGHT ADJUST CAPABILITY
               DPCD Addr : 00702h
    ****************************************/
    // read the 8 bits/1 bytes from 0x702
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x702, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_blcap_adj = value[0];

    s_blreg->lwsr_bl_bright_pwm_pin_capable = s_blreg->lwsr_blcap_adj & 0x01;
    s_blreg->lwsr_bl_bright_aux_set_capable = s_blreg->lwsr_blcap_adj & 0x02;
    s_blreg->lwsr_bl_bright_aux_byte_count = s_blreg->lwsr_blcap_adj & 0x04;
    s_blreg->lwsr_bl_aux_pwm_prod_capable = s_blreg->lwsr_blcap_adj & 0x08;
    s_blreg->lwsr_bl_pwm_freq_pin_pt_capable = s_blreg->lwsr_blcap_adj & 0x10;
    s_blreg->lwsr_bl_aux_freq_set_capable = s_blreg->lwsr_blcap_adj & 0x20;
    s_blreg->lwsr_bl_dynamic_capable = s_blreg->lwsr_blcap_adj & 0x40;
    s_blreg->lwsr_bl_vsync_update_capable = s_blreg->lwsr_blcap_adj & 0x80;

     /****************************************
             BACKLIGHT CAPABILITY #2
               DPCD Addr : 00703h
    ****************************************/
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x703, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_blcap2 = value[0];

    s_blreg->lwsr_lcd_ovrdrv = s_blreg->lwsr_blcap2 & 0x01;
    s_blreg->lwsr_bl_1reg_drv = s_blreg->lwsr_blcap2 & 0x02;
    s_blreg->lwsr_bl_1str_drv = s_blreg->lwsr_blcap2 & 0x04;

    /****************************************
             BACKLIGHT CAPABILITY #3
               DPCD Addr : 00704h
    ****************************************/
    // read the 8 bits/1 bytes from 0x704
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x704, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_blcap3 = value[0];

    s_blreg->lwsr_x_region_cap = s_blreg->lwsr_blcap3 & 0x0F;
    s_blreg->lwsr_y_region_cap = s_blreg->lwsr_blcap3 & 0xF0;

    /***********************************************
             DISPLAY PANEL FEATURE CONTROL
                 DPCD Addr : 00720h
    ***********************************************/
    // read the 8 bits/1 bytes from 0x720
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x720, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_disp_cntrl = value[0];

    s_blreg->lwsr_bl_enable = s_blreg->lwsr_disp_cntrl & 0x01;
    s_blreg->lwsr_blackvideo_enable = s_blreg->lwsr_disp_cntrl & 0x02;
    s_blreg->lwsr_frc_enable = s_blreg->lwsr_disp_cntrl & 0x04;
    s_blreg->lwsr_clreng_enable = s_blreg->lwsr_disp_cntrl & 0x08;
    s_blreg->lwsr_vsync_bl_updt_en = s_blreg->lwsr_disp_cntrl & 0x80;

    /****************************************
             BACKLIGHT MODE SET
                   DPCD Addr : 00721h
    ****************************************/
    // read the 8 bits/1 bytes from 0x721
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x721, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_mode_set = value[0];

    s_blreg->lwsr_bl_bright_cntrl_mode = s_blreg->lwsr_bl_mode_set & 0x03;
    s_blreg->lwsr_bl_pwm_freq_pin_pt_enable = s_blreg->lwsr_bl_mode_set & 0x04;
    s_blreg->lwsr_bl_aux_freq_set_enable = s_blreg->lwsr_bl_mode_set & 0x08;
    s_blreg->lwsr_bl_dynamic_enable = s_blreg->lwsr_bl_mode_set & 0x10;
    s_blreg->lwsr_bl_rg_bl_enable = s_blreg->lwsr_bl_mode_set & 0x20;
    s_blreg->lwsr_bl_updt_britnes = s_blreg->lwsr_bl_mode_set & 0x40;

    /***********************************************
                BACKLIGHT BRIGHTNESS
             DPCD Addr : 00722h ~ 00723h
    ***********************************************/
    // read the 16 bits/2 bytes from 0x722
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x722, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_brightness = (LwU16)value[0] | (LwU16)value[1]<<8;

    s_blreg->lwsr_bl_brightness_msb = s_blreg->lwsr_bl_brightness & 0xFF;
    s_blreg->lwsr_bl_brightness_lsb = (s_blreg->lwsr_bl_brightness >> 8) & 0xFF;
    s_blreg->lwsr_bl_brightness = ((LwU16)s_blreg->lwsr_bl_brightness_msb << 8) | (LwU16)s_blreg->lwsr_bl_brightness_lsb;

    /***************************************************
              PWM GENERATION AND BL CONTROLLER STATUS
                DPCD Addr : 00724h ~ 00727h 
    ****************************************************/
    // read the 32 bits/4 bytes from 0x724
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x724, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_pwm = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_blreg->lwsr_pwmgen_bit_count = s_blreg->lwsr_bl_pwm & 0x1F;
    s_blreg->lwsr_pwmgen_bit_count_min = (s_blreg->lwsr_bl_pwm >> 8) & 0x1F;
    s_blreg->lwsr_pwmgen_bit_count_max = (s_blreg->lwsr_bl_pwm >> 16) & 0x1F;
    s_blreg->lwsr_bl_cntrl_status = (s_blreg->lwsr_bl_pwm >> 24) & 0x01;

    /***************************************************
                         BL PWM FREQUENCY
                       DPCD Addr : 728h 
    ****************************************************/
    // read the 8 bits/1 bytes from 0x728
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x728, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_freq_set = value[0];

    /***************************************************
             BACKLIGHT BRIGHTNESS RANGE
                DPCD Addr : 00732h ~ 00733h 
    ****************************************************/
    // read the 8 bits/1 bytes from 0x732
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x732, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_brighness_min = value[0];

    // read the 8 bits/1 bytes from 0x733
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x733, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_blreg->lwsr_bl_brighness_max = value[0];

    s_blreg->lwsr_bl_brighness_min = s_blreg->lwsr_bl_brighness_min & 0x1F;
    s_blreg->lwsr_bl_brighness_max = s_blreg->lwsr_bl_brighness_max & 0x1F;

    return RM_LWSR_OK;
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PARSE LWSR DIAGNOSTIC REGISTERS

RM_LWSR_STATUS parseLWSRDiagnosticRegs(LwU32 port, lwsr_diagnosticreg_fields *s_diagreg)
{
    LwU8 value[16] = {0};
    RM_AUX_STATUS lwsrStatus = 0;

    /*****************************************
                 Diagnostic Registers
               DPCD Addr : 00390h~0039Bh
    *****************************************/
    // read the 32 bits/4 bytes from 0x390
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x390, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_diagreg->lwsr_diagnostic_390h = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_diagreg->lwsr_dsfc = s_diagreg->lwsr_diagnostic_390h & 0x7FFFFFFF;        // /in frames
    s_diagreg->lwsr_dsrd = (LwU8)(s_diagreg->lwsr_diagnostic_390h >> 31 & 0x01);

    // read the 16 bits/2 bytes from 0x394        // lines
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x394, value, 2);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_diagreg->lwsr_xtcsl = (LwU32)value[0] | (LwU32)value[1]<<8 ;

    // read the 8 bits/1 bytes from 0x396  
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x396, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_diagreg->lwsr_drfc = value[0];

    // read the 8 bits/1 bytes from 0x397        // /in frames
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x397, value, 1);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_diagreg->lwsr_diagnostic_397h = value[0];

    s_diagreg->lwsr_srts = s_diagreg->lwsr_diagnostic_397h & 0x07;

    // read the 32 bits/4 bytes from 0x398
    lwsrStatus = pDpaux[indexGpu].dpauxChReadMulti(port, 0x398, value, 4);
    if (lwsrStatus == 0)
        return RM_LWSR_PORT_ERROR;
    s_diagreg->lwsr_diagnostic_398h = (LwU32)value[0] | (LwU32)value[1]<<8 | (LwU32)value[2]<<16 | (LwU32)value[3]<<24;

    s_diagreg->lwsr_frtf = s_diagreg->lwsr_diagnostic_398h & 0x7FFFFFFF;        // /in frames
    s_diagreg->lwsr_srrd_diag = (LwU8)(s_diagreg->lwsr_diagnostic_398h >> 31 & 0x01);

    return RM_LWSR_OK;
}

// ---------------------------------------------------------------------------------------------------------------------------------
