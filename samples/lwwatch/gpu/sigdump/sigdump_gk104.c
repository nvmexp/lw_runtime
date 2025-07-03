/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "chip.h"
#include "sig.h"
#include "gr.h"

#include "g_sig_private.h"     // (rmconfig)  implementation prototypes


#ifndef SIGDUMP_ENABLE

void sigInitInstanceInfo_GK104(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigPrintLegend_GK104(FILE* fp)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

#else // SIGDUMP_ENABLE

//Initialize the Instance limits based on fs config
void sigInitInstanceInfo_GK104(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
    LwU32 i;
    for (i = 0; i < NUMINSTANCES; i++)
    {
        pInfo[i].instanceName = sigGetInstanceName(i);
        switch(i)
        {
            case gpc:
                pInfo[i].chipletLimit = ngpc-1;   //gpc chiplet limit
                pInfo[i].instanceLimit = 3;       //four instances
                pInfo[i].bValid = TRUE;
                break;

            case gpc_tpc:
                pInfo[i].chipletLimit = ngpc-1;   //gpc chiplet limit
                pInfo[i].instanceLimit = 1;       //two instances
                pInfo[i].bValid = TRUE;
                break;

            case gpc_ppc:
                pInfo[i].chipletLimit = ngpc-1;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case fbp:
                pInfo[i].chipletLimit = nfbp-1;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case sys:
                pInfo[i].chipletLimit = 0;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case sys_mxbar_cs_daisy:
                pInfo[i].chipletLimit = 0;        //only one chiplet
                pInfo[i].instanceLimit = 8;       // 9 instances
                pInfo[i].bValid = TRUE;
                break;

            case sys_wxbar_cs_daisy:
                pInfo[i].chipletLimit = 0;        //only one chiplet
                pInfo[i].instanceLimit = 4;       // 5 instances
                pInfo[i].bValid = TRUE;
                break;

            default:
                pInfo[i].chipletLimit = 0;
                pInfo[i].instanceLimit = 0;
                pInfo[i].bValid = FALSE;        //dont include this stray
        }
    }
}

void sigPrintLegend_GK104(FILE* fp)
{
    fprintf(fp, " ****** GK104 Sigdump LEGEND ****** \n");
    fprintf(fp, " There are 5 columns (source, chiplet, instance, signal, value).\n");
    fprintf(fp, "   'source' - It represents hierarchy at which the PM signal exists.\n");
    fprintf(fp, "       fbp     => in FBP.\n");
    fprintf(fp, "       gpc     => in GPC(outside TPCs).\n");
    fprintf(fp, "       gpc_tpc => in GPC(inside TPCs).\n");
    fprintf(fp, "       gpc_ppc => in GPC(inside PPCs).\n");
    fprintf(fp, "       sys     => in SYS.\n");
    fprintf(fp, "       sys_mxbar_cs_daisy/unrolled => in MXBAR.\n");
    fprintf(fp, "       sys_wxbar_cs_daisy/unrolled => in WXBAR.\n");
    fprintf(fp, "   'chiplet' - It represents the instance # of the chiplet in which the signal exists.\n");
    fprintf(fp, "       Chiplet numbers are virtualized on floorswept chips.\n");
    fprintf(fp, "       For 'source' value\n");
    fprintf(fp, "           fbp     => fbp# : 0..%u based on FBP floorsweeping\n",
        pGr[indexGpu].grGetMaxFbp());
    fprintf(fp, "           gpc     => gpc# : 0..%u based on GPC floorsweeping\n",
        pGr[indexGpu].grGetMaxGpc());
    fprintf(fp, "           gpc_tpc => gpc# : 0..%u based on GPC/TPC floorsweeping\n",
        pGr[indexGpu].grGetMaxTpcPerGpc());
    fprintf(fp, "           sys, sys_mxbar_cs_daisy, sys_wxbar_cs_daisy     => 0 always\n");
    fprintf(fp, "   'instance' - It represents the instance # of the unit in the chiplet in which the signal exists.\n");
    fprintf(fp, "       For 'source' value\n");
    fprintf(fp, "           fbp, gpc, gpc_ppc, sys     => 0 always\n");
    fprintf(fp, "           gpc_tpc => tpc# : 0..3 based on TPC floorsweeping, virtualized when floorswept.\n");
    fprintf(fp, "           sys_mxbar_cs_daisy => 0..10 (numGPC + numFBP + numSYS), independent of floorsweeping, not virtualized.\n");
    fprintf(fp, "           sys_wxbar_cs_daisy => 0..4  (numGPC + numSYS), independent of floorsweeping, not virtualized.\n");
    fprintf(fp, "                xbar Details with David Tang\n");
    fprintf(fp, "   'signal' -  Name of the PM signal.\n");
    fprintf(fp, "       Each row represents a PM signal, which can be a 1 bit wire or a bus.\n");
    fprintf(fp, "   'value' -   Value of the PM signal.\n");
    fprintf(fp, " ****************************** \n");
    fprintf(fp, "\n");
}

#endif

#ifndef SIGDUMP_ENABLE

void sigGetSigdump_GK104 (
    FILE *fp,
    int regWriteOptimization,
    int regWriteCheck,
    int markerValuesCheck,
    int verifySigdump,
    int engineStatusVerbose,
    int priCheckVerbose,
    int multiSignalOptimization,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigGetSigdumpFloorsweep_GK104(FILE *fp, LwU32 gpc, LwU32 tpc, LwU32 fbp)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigGetSelectSigdump_GK104(FILE *fp, char *sfile)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigGetSelectSigdumpFloorsweep_GK104(FILE *fp, LwU32 gpc, LwU32 tpc, LwU32 fbp, char *sfile)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

#else // SIGDUMP_ENABLE


//#define SIGDUMP_VERBOSE  1

static InstanceInfo instanceInfo[NUMINSTANCES];
#define TPC_IN_GPC_ARR_MAX 50
static LwU32 TpcInGpc[TPC_IN_GPC_ARR_MAX];
static BOOL sigdumpFindString(char *string, char **hash[]);
static LwU32 sigdumpGetKeyForString(char *string);

//TO:DO - Needs to be changed to support SMC
static void getFsInfo(LwU32* ngpc, LwU32* nfbp, LwU32* pTpc)
{
    LwU32 i;
    *ngpc = pGr[indexGpu].grGetNumActiveGpc(0);
    *nfbp = pGr[indexGpu].grGetNumActiveFbp();

    if (*ngpc > pGr[indexGpu].grGetMaxGpc())
    {
        *ngpc = pGr[indexGpu].grGetMaxGpc();
    }

    if (*nfbp > pGr[indexGpu].grGetMaxFbp())
    {
        *nfbp = pGr[indexGpu].grGetMaxFbp();
    }

    assert(TPC_IN_GPC_ARR_MAX >= *ngpc);
    for (i=0; i<*ngpc; i++)
    {
        pTpc[i] = pGr[indexGpu].grGetNumTpcForGpc(i, 0);
        if (pTpc[i] > pGr[indexGpu].grGetMaxTpcPerGpc())
        {
            pTpc[i] = pGr[indexGpu].grGetMaxTpcPerGpc();
        }
    }
}

static LW_STATUS openSigdumpFiles(FILE** pfp_sig, FILE** pfp_reg)
{
    FILE *sigfp, *regsfp;
    char *mode = "r";
    char *signal_file;
    char *regs_file;

    if (IsGM107())
    {
        signal_file = "signals_gm107.sig";
        regs_file   = "regsinfo_gm107.sig";
    }
    else if (IsGM200())
    {
        signal_file = "signals_gm200.sig";
        regs_file   = "regsinfo_gm200.sig";
    }
    else if (IsGM204())
    {
        signal_file = "signals_gm204.sig";
        regs_file   = "regsinfo_gm204.sig";
    }
    else if (IsGM206())
    {
        signal_file = "signals_gm206.sig";
        regs_file   = "regsinfo_gm206.sig";
    }
    else
    {
        dprintf("lw: Incorrect Architecture \n");
        return LW_ERR_GENERIC;
    }

    dprintf("lw: Trying to open %s and %s in current directory.\n", signal_file, regs_file);
    sigfp = fopen(signal_file, mode);
    regsfp = fopen(regs_file, mode);

    if (sigfp == NULL || regsfp == NULL)
    {
        dprintf("lw: Could not open %s and %s in current directory.\n", signal_file, regs_file);
        return LW_ERR_GENERIC;
    }

    *pfp_sig = sigfp;
    *pfp_reg = regsfp;
    return LW_OK;
}

static LW_STATUS closeSigdumpFiles(FILE* fp_sig, FILE* fp_reg)
{
    fclose(fp_sig);
    fclose(fp_reg);
    return LW_OK;
}

static LwU32 getInstanceClass(char* instance)
{
    LwU32 i;
    for (i=0; i<NUMINSTANCES; i++)
    {
        if (!strcmp(instance, sigGetInstanceName(i)))
        {
            return i;
        }
    }
    return NUMINSTANCES;
}

static BOOL isValidSignal(sigdump_GK104_t* pSignal, InstanceInfo* pInfo, FILE *isgFp, char **isgHash[])
{
    LwU32 inst;
    LwU32 chipletLimit;
    LwU32 instanceLimit;

    if (pSignal == NULL)
        return FALSE;

    inst = pSignal->instanceClass;
    if (pInfo[inst].bValid == FALSE)
        return FALSE;

    chipletLimit = pInfo[inst].chipletLimit;
    // if gpc_tpc instanceClass; then num of instances (tpc) is equal to num of valid tpc for the given chiplet (gpc)
    assert(TPC_IN_GPC_ARR_MAX > pSignal->chiplet);
    instanceLimit = inst == gpc_tpc ? TpcInGpc[pSignal->chiplet] - 1 : pInfo[inst].instanceLimit ;

    if(chipletLimit == (LwU32)-1) //To handle special case -f 000. Return false for signals pertaining to gpc,tpc & fbp.
        return FALSE;

    if(isgFp && (sigdumpFindString(pSignal->str, isgHash) == FALSE))
        return FALSE;

    if (pSignal->chiplet <= chipletLimit && pSignal->instance <= instanceLimit)
        return TRUE;

    return FALSE;
}

//parse the signal name to find out instanceClass, chiplet and instance
static LW_STATUS getSignalInfo(sigdump_GK104_t* pSignal)
{
    char strInstanceClass[64];
    char signame[SIG_STR_MAX];
    LwU32 chiplet, instance, inst;
    char  str[SIG_STR_MAX];
    LwU32 i, tokens;

    strncpy(str, pSignal->str, SIG_STR_MAX);

    for (i=0; str[i] != '\0'; i++)
    {
        if (str[i] == '.' || str[i] == '[' || str[i] == ']' || str[i] == ':' || str[i] == ',')
        {
            str[i] = ' ';
        }
    }

    tokens = sscanf(str, "%s %s %d %d", strInstanceClass, signame, &chiplet, &instance);
    if( tokens != 4 )
    {
        dprintf("lw: getSignalInfo malformed signal '%s'\n", pSignal->str );
        return LW_ERR_GENERIC;
    }

    inst = getInstanceClass(strInstanceClass);
    if (inst >= NUMINSTANCES)
    {
        dprintf("lw: unidentified instance %s for signal %s \n", strInstanceClass, pSignal->str);
        return LW_ERR_GENERIC;
    }

    strncpy(pSignal->str, signame, SIG_STR_MAX);
    pSignal->chiplet = chiplet;       //instanceClass.signalname[chiplet,instance]
    pSignal->instanceClass = inst;
    pSignal->instance = instance;
    return LW_OK;
}

static LW_STATUS getSignal(FILE* fp, sigdump_GK104_t* pSignal)
{
    LwU32   tokens;
    char    temp[SIG_STR_MAX];
    if(fp == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if(NULL == fgets(temp, SIG_STR_MAX, fp))
    {
        return LW_ERR_GENERIC;
    }

    tokens = sscanf(temp, "%s 0x%x %d %d %d", pSignal->str, &(pSignal->addr), &(pSignal->lsb), &(pSignal->msb), &(pSignal->num_writes));

    if(tokens != 5)
    {
        dprintf("lw: getSignal: malformed signal '%s'\n", temp );
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static LW_STATUS getRegWrites(FILE* fp, reg_write_GK104_t* pRegWrite)
{
    LwU32 tokens;
    tokens = fscanf(fp, "0x%x 0x%x 0x%x\n", &pRegWrite->addr, &pRegWrite->value, &pRegWrite->mask);
    if (tokens != 3)
    {
        return LW_ERR_GENERIC;
    }
    return LW_OK;
}

static void printHeader(FILE* fp)
{
   fprintf(fp, "source   chiplet  instance      signal[bitwidth]      value\n");
}

static void outputSignalFmt(FILE *fp, sigdump_GK104_t* pSignal)
{
    const char* instanceClass;
    LwU32 sigVal;

    if (!fp || !pSignal)
        return;

    if (pSignal->instanceClass >= NUMINSTANCES)
        return;

    instanceClass = sigGetInstanceName(pSignal->instanceClass);
    RegBitfieldRead (&sigVal, pSignal->addr, pSignal->lsb, pSignal->msb, LW_TRUE);
    fprintf(fp, "%s  %d  %d  %s[%d:0]  0x%x\n",
            instanceClass, pSignal->chiplet, pSignal->instance, pSignal->str,
            (pSignal->msb - pSignal->lsb),
            sigVal);
}

void sigGetSigdump_GK104
(
    FILE *fp,
    int regWriteOptimization,
    int regWriteCheck,
    int markerValuesCheck,
    int verifySigdump,
    int engineStatusVerbose,
    int priCheckVerbose,
    int multiSignalOptimization,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
)
{
    LwU32 ngpc;
    LwU32 nfbp;

    getFsInfo(&ngpc, &nfbp, TpcInGpc);
    sigGetSigdumpFloorsweep_GK104(fp, ngpc, pGr[indexGpu].grGetMaxTpcPerGpc(), nfbp);
}


#define HASH_RESOLUTION 1024
static char *selectSignalsFile = NULL;

void sigGetSelectSigdump_GK104(FILE *fp, char *sfile)
{
    LwU32 ngpc;
    LwU32 nfbp;

    getFsInfo(&ngpc, &nfbp, TpcInGpc);
    sigGetSelectSigdumpFloorsweep_GK104(fp, ngpc, pGr[indexGpu].grGetMaxTpcPerGpc(), nfbp, sfile);
}

void sigGetSelectSigdumpFloorsweep_GK104(FILE *fp, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, char *sfile)
{
    selectSignalsFile = sfile;
    sigGetSigdumpFloorsweep_GK104(fp, ngpc, ntpc, nfbp);
    selectSignalsFile = NULL;
}

static LwU32 sigdumpGetKeyForString(char *string)
{
    size_t ind   = 0, max = strlen(string);
    size_t count = 0;
    while(ind < max)
    {
        count += (string[ind]*(ind+1));
        ++ind;
    }
    return count%HASH_RESOLUTION;
}


//First element of hash[key] holds the number of strings present for that particular key.
static void sigdumpAddToHash(char *string, char **hash[HASH_RESOLUTION])
{
    LwU32 key = sigdumpGetKeyForString(string);
    LwU32 totalElem = 0, newsize = 0;
    char **temp = NULL;

    if(!hash[key])
    {
        totalElem = 1;
        newsize = (totalElem+1) * sizeof(char*);
        hash[key]=(char **)malloc(newsize);
        memset(hash[key], 0, newsize);
        hash[key][0]=(char *)malloc(sizeof(LwU32));
    }
    else
    {
        memcpy((char *)&totalElem, hash[key][0], sizeof(LwU32));
        totalElem += 1;
        newsize = (totalElem+1) * sizeof(char*);
        temp = (char **)malloc(totalElem*sizeof(char*));
        memcpy(temp, hash[key], totalElem*sizeof(char*));
        hash[key]=(char **)malloc(newsize);
        memcpy(hash[key], temp, totalElem*sizeof(char*));
        free(temp);

    }
    memcpy(hash[key][0], (char *)&totalElem, sizeof(LwU32));
    hash[key][totalElem]=(char *)malloc(sizeof(char) * strlen(string));
    memcpy(hash[key][totalElem],string, strlen(string));
    hash[key][totalElem][strlen(string)]='\0';
}


static LW_STATUS SigdumpReadFile(FILE *pfp, char **hash[])
{
    char temp[SIG_STR_MAX], *ptr;
    if(!pfp)
        return LW_ERR_GENERIC;
    while(fgets(temp, SIG_STR_MAX, pfp))
    {
        ptr = strchr(temp,'\n');
        if(ptr)
            *ptr='\0';
        sigdumpAddToHash(temp, hash);
    }
    return LW_OK;
}

static BOOL sigdumpFindString(char *string, char **hash[])
{
    LwU32 key = sigdumpGetKeyForString(string);
    LwU32 noElem = 0;
    if(!hash[key])
        return FALSE;
    memcpy((char *)&noElem, hash[key][0], sizeof(LwU32));
    while(noElem)
    {
        if(strcmp(string, hash[key][noElem]) == 0)
            return TRUE;
        --noElem;
    }
    return FALSE;
}

static void sigdumpCleanUp(FILE *fp, char **hash[])
{
    LwU32 ind = 0, noElem = 0;
    if(fp)
        fclose(fp);
    else
        return;

    return; //Free gives assert.. needs to fix it. so just returning as temp fix

    while(ind < HASH_RESOLUTION)
    {
        LwU32 jind = 1;
        if(!hash[ind])
        {
            ++ind;
            continue;
        }
        memcpy((char *)&noElem, hash[ind][0], sizeof(LwU32));
        while(jind <= noElem)
        {
            free(hash[ind][jind]);
            hash[ind][jind] = NULL;
            ++jind;
        }
        free(hash[ind]);
        hash[ind] = NULL;
        ++ind;
    }
}

void sigGetSigdumpFloorsweep_GK104(FILE *fp, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp)
{
    LwU32               r;
    LwU32               i;
    LwU32               skipped_writes = 0;
    sigdump_GK104_t     signal;
    reg_write_GK104_t   regWrite;
    reg_write_GK104_t   lastRegWrite;
    FILE*               sigFp;
    FILE*               regFp;
    FILE*               isgFp = NULL;
    LW_STATUS           status = LW_OK;
    LwU32               numWrites = 0;
    BOOL                bValidSig = FALSE;
    LwU32               option = COMMON_DUMP;
    char                **isgHash[HASH_RESOLUTION];
    #ifdef SIGDUMP_VERBOSE
    FILE*               verifFp = fopen("sigdump_verif.txt", "w");
    #endif

    //
    //this is a hack to ilwoke sigdump for the huge xbar signals list
    //if all gpc tpc fbp are 0 ; do XBAR_DUMP
    //
    if ((ngpc == 0)&&(ntpc == 0)&&(nfbp == 0))
    {
        option = XBAR_DUMP;
    }

    memset(isgHash, 0, HASH_RESOLUTION *sizeof(char **));
    if(selectSignalsFile)
    {
        dprintf("lw: Trying to open %s in current directory.\n", selectSignalsFile);
        isgFp = fopen(selectSignalsFile, "r");
        if(!isgFp)
        {
            dprintf("lw: Could not open %s in current directory.\n", selectSignalsFile);
            selectSignalsFile = NULL;
            return;
        }
    }
    if(isgFp && (SigdumpReadFile(isgFp, isgHash) == LW_ERR_GENERIC))
    {
        selectSignalsFile = NULL;
        return;
    }
    selectSignalsFile = NULL;

    r = GPU_REG_RD32(0);
    if (r == 0)
        return;

    status = openSigdumpFiles(&sigFp, &regFp);
    if (status == LW_ERR_GENERIC)
        return;

    if(ngpc || ntpc || nfbp) //To handle special case 000, let the values be 0. dont set it to default
    {
        if ( ngpc > pGr[indexGpu].grGetMaxGpc() || ngpc == 0)
        {
            ngpc = pGr[indexGpu].grGetMaxGpc();
        }

        if ( ntpc > pGr[indexGpu].grGetMaxTpcPerGpc() || ntpc == 0)
        {
            ntpc = pGr[indexGpu].grGetMaxTpcPerGpc();
        }

        if ( nfbp > pGr[indexGpu].grGetMaxFbp() || nfbp == 0)
        {
            nfbp = pGr[indexGpu].grGetMaxFbp();
        }
    }

    pSig[indexGpu].sigInitInstanceInfo(instanceInfo, ngpc, ntpc, nfbp, option);

    r = 0;
    dprintf("Starting to dump signals...\n");
    dprintf("lw: For config: \n");
    dprintf("lw: GPC: %d, FBP: %d\n", ngpc, nfbp);
    dprintf("lw: TPCs for GPCs : ");
    assert(TPC_IN_GPC_ARR_MAX >= ngpc);
    for (i=0; i<ngpc; i++)
    {
         dprintf("%d ", TpcInGpc[i]);
    }
    dprintf("\n");

    pSig[indexGpu].sigPrintLegend(fp);
    fprintf(fp, "Floorsweep Config: \n");
    fprintf(fp, "GPC: %d, FBP: %d\n", ngpc, nfbp);
    fprintf(fp, "TPCs for GPCs : ");
    for (i=0; i<ngpc; i++)
    {
         fprintf(fp, "%d ", TpcInGpc[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "\n");
    printHeader(fp);

    while(LW_OK == getSignal(sigFp, &signal))
    {
        //parse through the signal name to get instanceClass,chiplet and instance
        status = getSignalInfo(&signal);
        bValidSig = (status == LW_ERR_GENERIC)? FALSE : isValidSignal(&signal, instanceInfo, isgFp, isgHash);

        for(i = 0; i < signal.num_writes; i++)
        {
            status = getRegWrites(regFp, &regWrite);
            if(status == LW_OK)
            {
                if (bValidSig)
                {
                    RegWrite(regWrite.addr, regWrite.value, regWrite.mask);
                    numWrites++;
                    lastRegWrite = regWrite;

                    #ifdef SIGDUMP_VERBOSE
                    if (verifFp)
                    {
                        fprintf(verifFp, "RegWrite(0x%.8x,0x%.8x,0x%.8x);\n", regWrite.addr, regWrite.value,
                                regWrite.mask);
                    }
                    #endif
                }
                else
                {
                    skipped_writes++;
                }
            }
            else
            {
                dprintf("lw: Error getting register writes for signal %s, chiplet %u\n",
                        signal.str, signal.chiplet);
            }
        }
        if (bValidSig)
        {
            // Print out the signal.
            outputSignalFmt(fp, &signal);
            r++;
#ifdef SIGDUMP_VERBOSE
            if (verifFp)
            {
                fprintf(verifFp, "OutputSignal(fp,\"%s[%d,%d]: \", RegBitRead(0x%.8x,%d,%d));\n", signal.str, signal.chiplet, signal.instance, signal.addr, signal.lsb, signal.msb);
            }
#endif
        }
        if (osCheckControlC())
        {
            dprintf("lw: Aborting ... \n");
            break;
        }
    }
    fprintf(fp, "\n");
    fprintf(fp, "Signals dumped: %d\n", r);
    if (skipped_writes)
    {
        dprintf("Skipped %d setup writes - is this a floorswept part?  Sigdump for FBs and TPCs in FS GPCs on GM10x and later is likely corrupt.\n", skipped_writes);
        dprintf("See http://lwbugs/1481767\n");
    }

    closeSigdumpFiles(sigFp, regFp);
    sigdumpCleanUp(isgFp, isgHash);
    dprintf("lw: Done! with writes: %d\n", numWrites);
    dprintf("lw:             reads: %d\n", r);
    dprintf("lw:           signals: %d\n", r);
    #ifdef SIGDUMP_VERBOSE
    if (verifFp)
    {
        fclose(verifFp);
    }
    #endif
}


#endif  // SIGDUMP_ENABLE
