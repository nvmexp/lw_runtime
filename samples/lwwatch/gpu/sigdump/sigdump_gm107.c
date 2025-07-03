/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <time.h>
#include <stdio.h>
#include <sys/types.h>

#include "string.h"
#if defined(SIGDUMP_ENABLE)
// PMLSplitter library C interface.
#include "PmlsplitterCApi.h"
#endif // defined(SIGDUMP_ENABLE)
#include "chip.h"
#include "sig.h"
#include "gr.h"
#include "fb.h"

#include "sigdump_helper.h"

#include "inst.h"
#include "print.h"
#include "gpuanalyze.h"
#include "regex/regex.h"

#include "g_sig_private.h"                                 // (rmconfig)  implementation prototypes
#include "maxwell/gm107/dev_master.h"

#if defined (_WIN32) || defined (_WIN64)
# define strtok_r strtok_s
#endif

// The following three definitions are used for "sigdump_debug" signal verification (GM20x and later only)
#define countof(array)  (sizeof(array) / sizeof(array[0]))

typedef struct _dbgSigTab_t
{
    const char* signalName;
    regex_t     signalRegEx;
    LwU32       signalValue;

} dbgSigTab_t;

static dbgSigTab_t  debugSignalTable[] = {
                                            {"^(sm[mv]_mio([0-9]*)2pm)[_0-9a-zA-Z]+_sigdump_debug$",       {0}, 0xABCD},
                                            {"^(sm[mv]_core([0-9]*)2pm)[_0-9a-zA-Z]+_sigdump_debug$",      {0}, 0x1234},
                                            {"^(sm[mv]_sctl[01]?([0]{1})2pm)[_0-9a-zA-Z]+_sigdump_debug$", {0}, 0x1234},
                                            {"^(sm[mv]_sctl[01]?([1]{1})2pm)[_0-9a-zA-Z]+_sigdump_debug$", {0}, 0x4567},
                                            {"^(sm[mv]_sctl[01]?([2]{1})2pm)[_0-9a-zA-Z]+_sigdump_debug$", {0}, 0x89AB},
                                            {"^(sm[mv]_sctl[01]?([3]{1})2pm)[_0-9a-zA-Z]+_sigdump_debug$", {0}, 0xCDEF},
                                         };

// Declaring the programming guide path messages as macros dependent on the OS.
#if defined (_WIN32) || defined (_WIN64)
#define DEFAULT_PROG_GUIDE_PATH_MESSAGE "The environment variable LWW_PROG_GUIDE_SDK is not set. \
Using default path: .\\chips\\%s\\pm_programming_guide.txt\n"

#define SET_PROG_GUIDE_PATH_MESSAGE "PM programming guide path set as: %s\\chips\\%s\\pm_programming_guide.txt\n"
#else
#define DEFAULT_PROG_GUIDE_PATH_MESSAGE "The environment variable LWW_PROG_GUIDE_SDK is not set. \
Using default path: ./chips/%s/pm_programming_guide.txt\n"

#define SET_PROG_GUIDE_PATH_MESSAGE "PM programming guide path set as: %s/chips/%s/pm_programming_guide.txt\n"
#endif

#define MIN_CHIPLET_NUM              0
#define MAX_CHIPLET_NUM              16
#define MIN_INSTANCE_NUM             0
#define MAX_INSTANCE_NUM             16

#define POLL_TIMEOUT_COUNT           20
#define REG_WRITE_CHECK_FAILED       -1
#define REG_WRITE_SKIPPED            0
#define REG_WRITE_SUCCESSFUL         1

static LwBool SIGDUMP_VERBOSE                = LW_FALSE;    // DEFAULT: Don't generate a file containing
                                                            // the perfmon/perfmux programming information
                                                            // (actual register reads/writes done).
static LwBool ENGINE_STATUS_VERBOSE          = LW_FALSE;    // DEFAULT: Don't print signal details if the
                                                            // ENGINE_STATUS for that signal fails to read
                                                            // EMPTY after POLL_TIMEOUT_COUNT number of polls.
                                                            // Print the number of failures to the display
                                                            // and also at the end of the generated dump.
static LwBool PRI_CHECK_VERBOSE              = LW_FALSE;    // DEFAULT: Don't print signal details if the
                                                            // ENGINE_STATUS/perfmon PRI read reads 0xBADFxxxx.
                                                            // Print the number of failures to the display and
                                                            // also at the end of the generated dump.
static LwBool SIGDUMP_SELECTIVE              = LW_FALSE;    // DEFAULT: Don't filter signals.
static LwBool CHIPLET_FILTER                 = LW_FALSE;    // DEFAULT: Don't filter signals by "chiplet".
static LwBool CHIPLET_NUM_FILTER             = LW_FALSE;    // DEFAULT: Don't filter signals by "chiplet#".
static LwBool DOMAIN_FILTER                  = LW_FALSE;    // DEFAULT: Don't filter signals by "domain".
static LwBool DOMAIN_NUM_FILTER              = LW_FALSE;    // DEFAULT: Don't filter signals by "domain#".
static LwBool INSTANCE_NUM_FILTER            = LW_FALSE;    // DEFAULT: Don't filter signals by "instance#".
static LwBool SIGDUMP_OPTIMIZATION           = LW_TRUE;     // DEFAULT: Dump with optimized register writes.
                                                            // Register writes already performed are "cached"
                                                            // and are not repeated for succeeding signals.
                                                            // The optimization logic is present in "helper.cpp".
static LwBool CHECK_REGWRITES                = LW_FALSE;    // DEFAULT: Don't check for correctness of reg
                                                            // writes by default.
                                                            // If enabled, this reads back the value from the
                                                            // address where the write was done and compares with
                                                            // the value that was attempted to be written.
static LwBool CHECK_MARKER_VALUES            = LW_TRUE;     // DEFAULT: Check (during run-time) if "sigdump_debug"
                                                            // (GM20x onwards) and "static_pattern" (GP100 onwards)
                                                            // match their expected values.
                                                            // Expected values for "sigdump_debug" signals are
                                                            // hard-coded. Expected values for "static_pattern"
                                                            // signals are extracted from the signal name.
static LwBool MULTI_SIGNAL_OPTIMIZATION      = LW_FALSE;    // DEFAULT: Dump single signal at a time
                                                            // Multiple signals can be optimized to mux several
                                                            // signals for reading back at the same time.

#ifndef SIGDUMP_ENABLE

void sigGetSigdump_GM107
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
    dprintf ("This function needs to be built with SIGDUMP enabled\n");
    dprintf ("Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

#else // SIGDUMP_ENABLE

// Declaration of floorsweeping related constants/variables follows.
// Floorsweep configs are displayed for GPCs, TPCs, ZLWLLs, FBPs, FBPAs, CEs and DISP_HEADs.
// Only GPC, TPC and FBP FS configs are passed onto PMLSplitter.
#define MAX_NUM_OF_GPCS 200


// Max num of LTCs is based on LW_SCAL_FAMILY_MAX_FBPS (16) x LW_SCAL_LITTER_NUM_LTC_PER_FBP (2) in hwproject.h
#define MAX_NUM_OF_LTC 32

// LW_SCAL_LITTER_NUM_LTC_SLICES
#define MAX_NUM_OF_LTS 8

struct floorsweep
{
    LwU32 activePartitionConfig;
    LwU32 numberOfActivePartitions;
    LwU32 maxNumberOfPartitions;
};
struct floorsweep GPC, FBP, FBPA, LWENC, CE, DISP_HEAD;
struct floorsweep TPC [MAX_NUM_OF_GPCS], ZLWLL [MAX_NUM_OF_GPCS], CPC[MAX_NUM_OF_GPCS], ROP[MAX_NUM_OF_GPCS] ;
struct floorsweep LTC[MAX_NUM_OF_LTC], LTS[MAX_NUM_OF_LTC][MAX_NUM_OF_LTS];

LwU32 chipArchImpl;
char chipName[PMCS_CHIP_NAME_MAX_LENGTH];

// Declaration of counter variables follows.
static unsigned int engineStatusFailureCount = 0;
static unsigned int badPriReadCount = 0;
static unsigned int numberOfFixedMarkerSignals = 0;
static unsigned int markerSignalFailures = 0;
static unsigned int staticPatternSignalFailures = 0;
static unsigned int numberOfStaticPatternSignals = 0;

// METHOD DECLARATIONS BEGIN.
// Descriptions of each method are given before their definitions.
static LwBool getAndSetFs (void);
static void printGK110WarningHeader (void);
static void printSigdumpLegend (FILE *fp);
static void printUnitFSConfig
(
    FILE *fp,
    char *unitName,
    struct floorsweep *unit
);
static void printSubunitFSConfig
(
    FILE *fp,
    char *unitName,
    struct floorsweep *unit,
    unsigned int unitID,
    char *subunitName,
    struct floorsweep *subunit
);
static void printFSConfig (FILE *fp);
static void printHeader (FILE *fp);
static int readAndPrintSignalInfo
(
    FILE *fp,
    FILE *verifFp,
    LwBool signalValid,
    pmcsSigdump *dump,
    LwU32 i
);
void sigGetSigdump_GM107
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
);
static LwBool verifyStaticPattern (char *signalName, LwU32 signalValue);
static LwBool verifySigdumpDebugMarker (char *signalName, LwU32 signalValue);
static LwBool parseStaticPattern (char *signalPattern, LwU32 *patterlwalue, LwU32 *patternMask, LwU32 *signalMask);
static void printStaticPatternSignalInfo (FILE *fp);
static void printSigdumpDebugSignalInfo (FILE *fp);
static LwBool filterSignal
(
    char *signalName,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
);
static LwBool pollEngineStatusEmpty (pmcsSigdump *dump);
static void sigGetSigdumpFloorsweep
(
    FILE *fp,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
);

// METHOD DEFINITIONS BEGIN.

// getAndSetFs: Reads the GPC, TPC and FBP FS configs through methods defined in GRStatus codes,
// and passes them on to PMLSplitter. Reads other FS configs and prints them out.
static LwBool getAndSetFs ()
{
    LwU32 GPCCount = 0;
    LwU32 GPCConfig;
    LwU32 i;

    // Use methods defined in GRStatus code to fetch the active floorsweep configs.
    // The values returned are the enable masks.
    if (pGr[indexGpu].grGetActiveGpcConfig (&GPC.activePartitionConfig, &GPC.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the GPC config!\n");
        return LW_FALSE;
    }
    if (pGr[indexGpu].grGetActiveFbpConfig (&FBP.activePartitionConfig, &FBP.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the FBP config!\n");
        return LW_FALSE;
    }
    if (pGr[indexGpu].grGetActiveFbpaConfig (&FBPA.activePartitionConfig, &FBPA.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the FBPA config!\n");
        return LW_FALSE;
    }
    if (pGr[indexGpu].grGetActiveLwencConfig (&LWENC.activePartitionConfig, &LWENC.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the LWENC config!\n");
        return LW_FALSE;
    }
    if (pGr[indexGpu].grGetActiveCeConfig (&CE.activePartitionConfig, &CE.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the CE config!\n");
        return LW_FALSE;
    }
    if (pGr[indexGpu].grGetActiveDispHeadConfig (&DISP_HEAD.activePartitionConfig, &DISP_HEAD.maxNumberOfPartitions) == LW_FALSE)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the DISP_HEAD config!\n");
        return LW_FALSE;
    }

    // Count the number of active partitions.
    GPC.numberOfActivePartitions = lwPopCount32 (GPC.activePartitionConfig);
    FBP.numberOfActivePartitions = lwPopCount32 (FBP.activePartitionConfig);
    FBPA.numberOfActivePartitions = lwPopCount32 (FBPA.activePartitionConfig);
    LWENC.numberOfActivePartitions = lwPopCount32 (LWENC.activePartitionConfig);
    CE.numberOfActivePartitions = lwPopCount32 (CE.activePartitionConfig);
    DISP_HEAD.numberOfActivePartitions = lwPopCount32 (DISP_HEAD.activePartitionConfig);

    assert ( (MAX_NUM_OF_GPCS >= GPC.numberOfActivePartitions));

    // Basic check: Number of active partitions must be < maximum number of partitions.
    if ( GPC.numberOfActivePartitions > GPC.maxNumberOfPartitions )
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the GPC config! "
            "Number of active GPCs read = %u, maximum number of GPCs = %u.\n",
            GPC.maxNumberOfPartitions,
            GPC.numberOfActivePartitions );
        return LW_FALSE;
    }

    if ( FBP.numberOfActivePartitions > FBP.maxNumberOfPartitions)
    {
        dprintf ( "FATAL ERROR! Could not correctly fetch the FBP config! "
            "Number of active FBPs read = %u, maximum number of FBPs = %u.\n",
            FBP.maxNumberOfPartitions,
            FBP.numberOfActivePartitions );
        return LW_FALSE;
    }

    if ( FBPA.numberOfActivePartitions > FBPA.maxNumberOfPartitions)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the FBPA config! "
            "Number of active FBPAs read = %u, maximum number of FBPAs = %u.\n",
            FBPA.maxNumberOfPartitions,
            FBPA.numberOfActivePartitions );
        return LW_FALSE;
    }

    if ( LWENC.numberOfActivePartitions > LWENC.maxNumberOfPartitions)
    {
        dprintf ("FATAL ERROR! Could not correctly fetch the LWENC config! "
            "Number of active LWENCs read = %u, maximum number of LWENCs = %u.\n",
            LWENC.maxNumberOfPartitions,
            LWENC.numberOfActivePartitions);
        return LW_FALSE;
    }

    if ( CE.numberOfActivePartitions > CE.maxNumberOfPartitions )
    {
        dprintf ( "FATAL ERROR! Could not correctly fetch the CE config! "
            "Number of active CEs read = %u, maximum number of CEs = %u.\n",
            CE.maxNumberOfPartitions,
            CE.numberOfActivePartitions );
        return LW_FALSE;
    }

    if ( DISP_HEAD.numberOfActivePartitions > DISP_HEAD.maxNumberOfPartitions )
    {
        dprintf ( "FATAL ERROR! Could not correctly fetch the DISP_HEAD config! "
            "Number of active DISP_HEADs read = %u, maximum number of DISP_HEADs = %u.\n",
            DISP_HEAD.maxNumberOfPartitions,
            DISP_HEAD.numberOfActivePartitions );
        return LW_FALSE;
    }

    dprintf ("NOTE: PMLSplitter supports floorsweeping for GPC/CPC/TPC, FBP/LTC/LTS partitions only."
        "\n      PM signals from other partitions will be dumped even if they are floorswept.\n");

    // Print the active GPC config onto the screen.
    dprintf ("Floorsweep config:\n    GPC config: 0x%X", GPC.activePartitionConfig);
    dprintf (" (%u GPCs)\n", GPC.numberOfActivePartitions);

    // Pass on the active GPC configuration to PMLSplitter.
    // Print error and exit on failure to set the config.
    if (pmcsInputSetNumGPCs (GPC.activePartitionConfig) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set the number of GPCs in PMLSplitter.\n");
        return LW_FALSE;
    }

    GPCConfig = GPC.activePartitionConfig;

    // Fetch and print the active TPC config for each GPC onto the screen.
    dprintf ("    CPC/TPC/ROP config: [");
    for (i = 0; i < GPC.numberOfActivePartitions; i++)
    {
        pGr[indexGpu].grGetActiveTpcConfig ( GPCCount,
                                             &TPC [GPCCount].activePartitionConfig,
                                             &TPC [GPCCount].maxNumberOfPartitions );

        if ( (GPCConfig & 0x1) )          // If the current GPC is not floorswept.
        {
            CPC[GPCCount].numberOfActivePartitions = pGr[indexGpu].grGetNumCPCsforGpc(GPCCount);
            if (CPC[GPCCount].numberOfActivePartitions != 0) // are there CPCs for the GPC, meaning arch is GH100 or later.
            {
                CPC[GPCCount].activePartitionConfig = pGr[indexGpu].grGetActiveCPCsForGpc(GPCCount);
                if (pmcsInputSetNumCPCs(GPCCount, CPC[GPCCount].activePartitionConfig) != PMCS_SUCCESS)
                    dprintf("FATAL error! setting FS CPC for gpc %u, mask 0x%x\n", GPCCount, CPC[GPCCount].activePartitionConfig);
                else
                    dprintf(" CPC  0x%x (GPC %u)", CPC[GPCCount].activePartitionConfig, GPCCount);
            }

            dprintf (" TPC 0x%X (GPC %u)", TPC [GPCCount].activePartitionConfig, GPCCount);
            // Pass on the active TPC configuration to PMLSplitter.
            // Print error and exit on failure to set the config.
            if (pmcsInputSetNumTPCs (GPCCount, TPC [GPCCount].activePartitionConfig) != PMCS_SUCCESS)
            {
                dprintf ("FATAL ERROR! Failed to set the number of TPCs in PMLSplitter.\n");
                return LW_FALSE;
            }
            // Set the number of active TPCs for the GPC "GPCCount".
            TPC [GPCCount].numberOfActivePartitions = lwPopCount32 (TPC [GPCCount].activePartitionConfig);
            if (TPC [GPCCount].numberOfActivePartitions > TPC [GPCCount].maxNumberOfPartitions)
            {
                dprintf ("ERROR: Could not correctly fetch the TPC config for GPC %u! "
                "Number of active CEs read = %u, maximum number of CEs = %u.\n",
                GPCCount,
                TPC [GPCCount].numberOfActivePartitions,
                TPC [GPCCount].maxNumberOfPartitions );
                return LW_FALSE;
            }

            ROP[GPCCount].numberOfActivePartitions = pGr[indexGpu].grGetNumRopForGpc(GPCCount, 0 /*grIdx*/);
            // are there ROPs for the GPC, meaning arch is GA102 or later.
            if (ROP[GPCCount].numberOfActivePartitions != 0)
            {
                ROP[GPCCount].activePartitionConfig= pGr[indexGpu].grGetActiveRopsForGpc(GPCCount);
                if (pmcsInputSetNumROPs(GPCCount, ROP[GPCCount].activePartitionConfig) != PMCS_SUCCESS)
                    dprintf("FATAL error! setting FS rop value for gpc %u, mask 0x%x\n", GPCCount, ROP[GPCCount].activePartitionConfig);
                else
                    dprintf(" ROP  0x%x (GPC %u)", ROP[GPCCount].activePartitionConfig, GPCCount);
            }
        }
        else    // If the current GPC is floorswept.
        {
            dprintf (" x (GPC %u) ", GPCCount);
            TPC[GPCCount].numberOfActivePartitions = 0;
            CPC[GPCCount].numberOfActivePartitions = 0;
            ROP[GPCCount].numberOfActivePartitions = 0;
        }
        GPCCount = GPCCount + 1;
        GPCConfig = GPCConfig >> 1;
    }
    dprintf (" ]    ( TPCs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
         dprintf ("%u ", TPC [i].numberOfActivePartitions);
    }

    GPCConfig = GPC.activePartitionConfig;
    GPCCount = 0;

    // Fetch and print the active ZLWLL config for each GPC onto the screen.
    dprintf (")\n    ZLWLL config: [");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
        pGr[indexGpu].grGetActiveZlwllConfig ( GPCCount,
                                               &ZLWLL [GPCCount].activePartitionConfig,
                                               &ZLWLL [GPCCount].maxNumberOfPartitions );

        if ( (GPCConfig & 0x1) != 0)      // If the current GPC is not floorswept.
        {
            dprintf (" 0x%X (GPC %u)", ZLWLL [GPCCount].activePartitionConfig, GPCCount);
            // Set the number of active ZLWLLs for this GPC.
            ZLWLL [GPCCount].numberOfActivePartitions = lwPopCount32 (ZLWLL [GPCCount].activePartitionConfig);
            if (ZLWLL [GPCCount].numberOfActivePartitions > ZLWLL [GPCCount].maxNumberOfPartitions)
            {
                dprintf ("ERROR: Could not correctly fetch the ZLWLL config for GPC %u! "
                "Number of active CEs read = %u, maximum number of CEs = %u.\n",
                GPCCount,
                ZLWLL [GPCCount].numberOfActivePartitions,
                ZLWLL [GPCCount].maxNumberOfPartitions );
                return LW_FALSE;
            }
        }
        else    // If the current GPC is floorswept.
        {
            dprintf (" x (GPC %u) ", GPCCount);
            ZLWLL [GPCCount].numberOfActivePartitions = 0;
        }
        GPCCount = GPCCount + 1;
        GPCConfig = GPCConfig >> 1;
    }
    dprintf (" ]    ( ZLWLLs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
         dprintf ("%u ", ZLWLL [i].numberOfActivePartitions);
    }

    // Print the active FBP config onto the screen.
    dprintf (")\n    FBP config: 0x%X", FBP.activePartitionConfig);
    dprintf (" (%u FBPs)\n", FBP.numberOfActivePartitions);

    // Pass on the active FBP configuration to PMLSplitter.
    // Print error and exit on failure to set the config.
    if (pmcsInputSetNumFBPs (FBP.activePartitionConfig) == PMCS_SUCCESS)
    {
        LwU32 fbpIdx = 0;
        FOR_EACH_INDEX_IN_MASK(32, fbpIdx, FBP.activePartitionConfig)
        {
            LwU32 ltcIdx = 0;
            LTC[fbpIdx].activePartitionConfig = pFb[indexGpu].fbGetActiveLtcMaskforFbp(fbpIdx);
            LTC[fbpIdx].numberOfActivePartitions = lwPopCount32(LTC[fbpIdx].activePartitionConfig);
            LTC[fbpIdx].maxNumberOfPartitions = 2; // LW_SCAL_LITTER_NUM_LTC_PER_FBP

            pmcsInputSetNumLTCs(fbpIdx, LTC[fbpIdx].activePartitionConfig);
            dprintf("\tfbp #%u ltcActiveMask 0x%x\n", fbpIdx, LTC[fbpIdx].activePartitionConfig);
            FOR_EACH_INDEX_IN_MASK(32, ltcIdx, LTC[fbpIdx].activePartitionConfig)
            {
                LTS[fbpIdx][ltcIdx].activePartitionConfig = pFb[indexGpu].fbGetActiveLtsMaskForLTC(ltcIdx);
                LTS[fbpIdx][ltcIdx].numberOfActivePartitions = lwPopCount32(LTS[fbpIdx][ltcIdx].activePartitionConfig);
                LTS[fbpIdx][ltcIdx].maxNumberOfPartitions = 8; // LW_SCAL_LITTER_NUM_LTC_SLICES
                dprintf("\t    fbp #%u ltcIdx %u ltsActiveMask 0x%x\n",
                        fbpIdx, ltcIdx, LTS[fbpIdx][ltcIdx].activePartitionConfig);
                if (pmcsInputSetNumLTSs(fbpIdx, ltcIdx, LTS[fbpIdx][ltcIdx].activePartitionConfig) != PMCS_SUCCESS)
                {
                    dprintf("\t\tFATAL ERROR! pmcsInputSetNumLTSs fbp #%u ltcIdx %u ltsActiveMask 0x%x\n",
                            fbpIdx, ltcIdx, LTS[fbpIdx][ltcIdx].activePartitionConfig);
                    break;
                }
            }
            FOR_EACH_INDEX_IN_MASK_END;
        }
        FOR_EACH_INDEX_IN_MASK_END
    }
    else
    {
        dprintf ("FATAL ERROR! Failed to set the number of FBPs in PMLSplitter.\n");
        return LW_FALSE;
    }

    // Print the active FBPA config onto the screen.
    dprintf ("    FBPA config: 0x%X", FBPA.activePartitionConfig);
    dprintf (" (%u FBPAs)\n", FBPA.numberOfActivePartitions);

    // Print the active LWENC config onto the screen.
    dprintf ("    LWENC config: 0x%X", LWENC.activePartitionConfig);
    dprintf (" (%u LWENCs)\n", LWENC.numberOfActivePartitions);

    // Print the active CE config onto the screen.
    dprintf ("    CE config: 0x%X", CE.activePartitionConfig);
    dprintf (" (%u CEs)\n", CE.numberOfActivePartitions);

    // Print the active DISP_HEAD config onto the screen.
    dprintf ("    DISP_HEAD config: 0x%X", DISP_HEAD.activePartitionConfig);
    dprintf (" (%u DISP_HEADs)\n", DISP_HEAD.numberOfActivePartitions);

    return LW_TRUE;
}

// printSigdumpLegend: Prints the sigdump legend into the output file.
static void printSigdumpLegend (FILE *fp)
{
    // Colwert the chip name to uppercase for the sole purpose of printing in the legend.
    LwU32 i;
    char chipNameUppercase[PMCS_CHIP_NAME_MAX_LENGTH];
    for (i = 0; chipName[i] != '\0'; ++i)
    {
        chipNameUppercase[i] = TOUPPER (chipName[i]);
    }
    chipNameUppercase[i] = TOUPPER (chipName[i]);

    // Print the legend.
    fprintf (fp, " ****** %s Sigdump LEGEND ****** \n", chipNameUppercase);
    fprintf (fp, " There are 7 columns (chiplet, chiplet#, domain, domain#, instance#, signal[msb:lsb], value).\n");
    fprintf (fp, "   'chiplet' - It represents the top-level hierarchy at which the PM signal exists.\n");
    fprintf (fp, "       fbp     => in FBP.\n");
    fprintf (fp, "       gpc     => in GPC.\n");
    fprintf (fp, "       sys     => in SYS.\n");
    fprintf (fp, "       sys_mxbar_cs_daisy => in MXBAR.\n");
    fprintf (fp, "       sys_wxbar_cs_daisy => in WXBAR.\n");
    fprintf (fp, "   'chiplet#' - It represents the # of the chiplet in which the signal exists.\n");
    fprintf (fp, "       Chiplet numbers are virtualized on floorswept chips.\n");
    fprintf (fp, "       For 'chiplet' value\n");
    fprintf (fp, "           fbp     => fbp# : 0..%u based on FBP floorsweeping\n", FBP.maxNumberOfPartitions - 1);
    fprintf (fp, "           gpc     => gpc# : 0..%u based on GPC floorsweeping\n", GPC.maxNumberOfPartitions - 1);
    fprintf (fp, "           sys, sys_mxbar_cs_daisy, sys_wxbar_cs_daisy     => 0 always\n");
    fprintf (fp, "   'domain' - It represents the next-level hierarchy of the PM signal. For example,\n");
    fprintf (fp, "       gpctpc => in TPC domain in GPC chiplet.\n");
    fprintf (fp, "       rop    => in ROP domain in FBP chiplet.\n");
    fprintf (fp, "       xbar   => in XBAR domain in SYS chiplet.\n");
    fprintf (fp, "   'domain#' - It represents the # of the domain in the chiplet in which the signal exists.\n");
    fprintf (fp, "       For 'domain' value\n");
    fprintf (fp, "           gpctpc => tpc# : 0..%u based on TPC floorsweeping\n", TPC [0].maxNumberOfPartitions - 1);
    fprintf (fp, "   'instance#' - It represents the instance # of the signal in which the signal exists.\n");
    fprintf (fp, "       For 'chiplet' value\n");
    fprintf (fp, "           fbp, gpc, sys      => 0 always\n");
    fprintf (fp, "           sys_mxbar_cs_daisy => 0..%u (numGPC + numFBP + numSYS), independent of floorsweeping, "
                            "not virtualized.\n", (GPC.maxNumberOfPartitions + FBP.maxNumberOfPartitions + 1) - 1);
    fprintf (fp, "           sys_wxbar_cs_daisy => 0..%u (numGPC + numSYS), independent of floorsweeping, "
                            "not virtualized.\n", (GPC.maxNumberOfPartitions + 1) - 1);
    fprintf (fp, "   'signal[msb:lsb]' -  Name of the PM signal and its bitwidth.\n");
    fprintf (fp, "       Each row represents a PM signal, which can be a 1 bit wire or a bus.\n");
    fprintf (fp, "   'value' -   Value of the PM signal.\n");
    fprintf (fp, " Signals marked with <POSSIBLY INVALID> may be corrupted due to floorsweeping "
                            "and/or power/clock gating.\n");
    fprintf (fp, " More details on floorsweeping here: "
        "https://wiki.lwpu.com/gpuhwdept/index.php/GPU_Performance_Infrastructure_Group/pmlsplitter/Floorsweeping\n");
    fprintf (fp, " ****************************** \n");
    fprintf (fp, "\n");
}

// printUnitFSConfig: Prints the FS config for a unit.
static void printUnitFSConfig
(
    FILE *fp,
    char *unitName,
    struct floorsweep *unit
)
{
    fprintf ( fp, "\n    %s config: 0x%X",
        unitName,
        unit -> activePartitionConfig );
    fprintf ( fp, " (%u %ss)",
        unit -> numberOfActivePartitions,
        unitName );
}

// printSubunitFSConfig: Prints the FS config for a subunit.
static void printSubunitFSConfig
(
    FILE *fp,
    char *unitName,
    struct floorsweep *unit,
    unsigned int unitID,
    char *subunitName,
    struct floorsweep *subunit
) {
    if (subunit -> numberOfActivePartitions)
    {
        fprintf ( fp, " 0x%X (%s %u)",
            subunit -> activePartitionConfig,
            unitName,
            unitID );
    }
    else
    {
        fprintf ( fp, " x (%s %u) ",
            unitName,
            unitID );
    }
}

// printFSConfig: Prints the FS config into the output file.
static void printFSConfig (FILE *fp)
{
    LwU32 i,j;

    fprintf (fp, "Floorsweep config:");
    // Print GPC config.
    printUnitFSConfig (fp, "GPC", &GPC);

    // Print CPC config.
    fprintf (fp, "\n    CPC config: [");
    for (i = 0; i < pGr[indexGpu].grGetNumCPCsforGpc(i); i++)
    {
        printSubunitFSConfig (fp, "GPC", &GPC, i, "CPC", &(CPC [i]));
    }
    fprintf (fp, " ]    ( CPCs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
        fprintf (fp, "%u ", CPC [i].numberOfActivePartitions);
    }
    fprintf (fp, ")");

    // Print TPC config.
    fprintf (fp, "\n    TPC config: [");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
        printSubunitFSConfig (fp, "GPC", &GPC, i, "TPC", &(TPC [i]));
    }
    fprintf (fp, " ]    ( TPCs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
         fprintf (fp, "%u ", TPC [i].numberOfActivePartitions);
    }
    fprintf (fp, ")");

    // Print ROP config.
    fprintf (fp, "\n    GPCROP config: [");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
        printSubunitFSConfig (fp, "GPC", &GPC, i, "ROP", &(ROP [i]));
    }

    fprintf (fp, " ]   ( ROPs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
         fprintf (fp, "%u ", ROP[i].numberOfActivePartitions);
    }
    fprintf (fp, ")");

    // Print ZLWLL config.
    fprintf (fp, "\n    ZLWLL config: [");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
        printSubunitFSConfig (fp, "GPC", &GPC, i, "ZLWLL", &(ZLWLL [i]));
    }
    fprintf (fp, " ]    ( ZLWLLs per GPC: ");
    for (i = 0; i < GPC.maxNumberOfPartitions; i++)
    {
         fprintf (fp, "%u ", ZLWLL [i].numberOfActivePartitions);
    }
    fprintf (fp, ")");

    // Print FBP config.
    printUnitFSConfig (fp, "FBP", &FBP);

    // Print FBP/LTC/LTS config.
    fprintf (fp, "\n    LTC per FBP config: [\n");

    FOR_EACH_INDEX_IN_MASK(32, i, FBP.activePartitionConfig)
    {
        LwBool bPrintLTS = (LTC[i].numberOfActivePartitions > 0);
        fprintf (fp, "    "); // indent for LTC
        printSubunitFSConfig (fp, "FBP", &FBP, i, "LTC", &(LTC [i]));

        if (bPrintLTS)
        {
            fprintf (fp, "\n\tLTS config: [");

            FOR_EACH_INDEX_IN_MASK(32, j, LTC[i].activePartitionConfig)
            {
                printSubunitFSConfig (fp, "LTC", &(LTC [i]), i, "LTS", &(LTS[i][j]));
            }
            FOR_EACH_INDEX_IN_MASK_END
            fprintf (fp, "]    ");
        }
        fprintf(fp, "\n");
    }
    FOR_EACH_INDEX_IN_MASK_END;
    fprintf (fp, "    ]"); // end of FBP

    // Print FBPA config.
    printUnitFSConfig (fp, "FBPA", &FBPA);

    // Print LWENC config.
    printUnitFSConfig (fp, "LWENC", &LWENC);

    // Print CE config.
    printUnitFSConfig (fp, "CE", &CE);

    // Print DISP_HEAD config.
    printUnitFSConfig (fp, "DISP_HEAD", &DISP_HEAD);

    fprintf (fp, "\n\n");
}

// printHeader: Prints the header into the output file.
static void printHeader (FILE* fp)
{
   fprintf (fp, "\nchiplet  chiplet#  domain  domain#  instance#  signal[msb:lsb]  value\n");
}

// pollEngineStatusEmpty: Reads the ENGINE_STATUS register provided by PMLSplitter
// and polls until it reads EMPTY (0x0).
static LwBool pollEngineStatusEmpty (pmcsSigdump *dump)
{
    LwU32 engineStatusValue;
    LwU32 readEngineStatusCounter;

    for ( readEngineStatusCounter = 0;
          readEngineStatusCounter < POLL_TIMEOUT_COUNT;
          readEngineStatusCounter ++ )
    {
        if (RegBitfieldRead ( &engineStatusValue,
                              dump -> engine_status_read.address,
                              dump -> engine_status_read.lsbPosition,
                              (dump -> engine_status_read.lsbPosition) + (dump -> engine_status_read.width) - 1,
                              PRI_CHECK_VERBOSE ) == LW_FALSE)
        {
            badPriReadCount ++;
            return LW_FALSE;           // ENGINE_STATUS check failed due to bad PRI read.
        }
        if (engineStatusValue == PMCS_PERFMON_ENGINE_STATUS_EMPTY)
        {
            return LW_TRUE;
        }
    }
    engineStatusFailureCount ++;
    // If the user has enabled prints for ENGINE_STATUS not going to EMPTY.
    if (ENGINE_STATUS_VERBOSE)
    {
        dprintf ("WARNING: ENGINE_STATUS (read @(0x%X)) did not "
            "become EMPTY after %u reads.\n", dump -> engine_status_read.address, POLL_TIMEOUT_COUNT);
    }
    return LW_FALSE;                   // ENGINE_STATUS did not read EMPTY after POLL_TIMEOUT_COUNT reads.
}

// verifySigdumpDebugMarker: Checks if the "sigdump_debug" signal matched its value
// (expected values are hard-coded). Returns 1 for a match, 0 for not a match.
static LwBool verifySigdumpDebugMarker (char *signalName, LwU32 signalValue)
{
    int signal;
    int reResult;
    regmatch_t reMatch[5];

    numberOfFixedMarkerSignals++;

    // Check for a debug signal match
    for (signal = 0; signal < countof(debugSignalTable); signal++)
    {
        reResult = regexec(&debugSignalTable[signal].signalRegEx, signalName, countof(reMatch), reMatch, 0);
        if (reResult == REG_NOERROR)
        {
            if (signalValue == debugSignalTable[signal].signalValue)
            {
                return LW_TRUE;
            }
            else
            {
                markerSignalFailures++;
                return LW_FALSE;
            }
        }
    }
    dprintf ("Error while verifying sigdump debug signal \"%s\"!\n", signalName);
    markerSignalFailures ++;
    return LW_FALSE;
}

// verifyStaticPattern: Checks if the "static_pattern" signal matched its value.
// Returns 1 for a match, 0 for not a match.
static LwBool verifyStaticPattern (char *signalName, LwU32 signalValue)
{
    char localSignalName[PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH];
    char *signalPattern;
    LwU32 patterlwalue, patternMask, signalMask;
    LwBool result = LW_TRUE;

    // Copy signal name since parsing will modify the signal name
    strncpy(localSignalName, signalName, PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);

    // Try to find the signal pattern in the local signal name
    signalPattern = strstr(localSignalName, "_static_pattern_");
    if (signalPattern != NULL)
    {
        // Increment the number of static signal patterns
        numberOfStaticPatternSignals++;

        // Try to parse the static signal pattern (Will modify signalPattern)
        if (parseStaticPattern(signalPattern, &patterlwalue, &patternMask, &signalMask))
        {
            // Check pattern value against signal value (Mask to signal size)
            if ((patterlwalue & signalMask) != ((signalValue & patternMask) & signalMask))
            {
                // Increment number of pattern failures and set error
                staticPatternSignalFailures ++;
                result = LW_FALSE;
            }
        }
    }
    return result;
}

typedef enum
{
    findPattern,
    getPattern,
    getSize,
    allDone
} ParseState;

static LwBool parseStaticPattern (char *signalPattern, LwU32 *patterlwalue, LwU32 *patternMask, LwU32 *signalMask)
{
    char pattern[32] = {0};
    char mask[32] = {0};
    char size[8] = "16";
    char *saveptr, *token;
    LwU32 length, character;
    LwU32 signalSize;
    ParseState state = findPattern;
    LwBool result = LW_FALSE;

    // Initialize the pattern value, mask, and signal mask
    *patterlwalue = 0;
    *patternMask  = 0;
    *signalMask   = 0;

    // Loop parsing all the tokens in the signal pattern (Split by "_", ".", and " ")
    token = strtok_r(signalPattern, "_. ", &saveptr);
    while(token)
    {
        // Switch on the current parse state
        switch(state)
        {
            case findPattern:

                // Check for start of the pattern value
                if (!strcmp(token, "pattern"))
                {
                    state = getPattern;
                }
                break;

            case getPattern:

                // Get the pattern value
                strncpy(pattern, token, 32);
                state = getSize;

                // Indicate result is good (Size defaults to 16 bits)
                result = LW_TRUE;

                break;

            case getSize:

                strncpy(size, token, 8);
                state = allDone;

                break;

            default:

                break;
        }
        // Get the next token from the signal pattern
        token = strtok_r(NULL, "_. ", &saveptr);
    }
    // Check for static pattern found
    if (result)
    {
        // Get the length of the pattern
        length = (LwU32)strlen(pattern);

        // Loop processing any mask characters in the pattern
        for (character = 0; character < length; character++)
        {
            if ((pattern[character] == 'x') || (pattern[character] == 'X'))
            {
                // Mask character, set pattern and mask string values
                pattern[character] = '0';
                mask[character]    = 'f';
            }
            else    // Non-mask character
            {
                // Set mask string value (leave pattern alone)
                mask[character] = '0';
            }
        }
        // Terminate the mask string
        mask[character] = '\0';

        // Colwert pattern value, mask, and size to values
        sscanf(pattern, "%x", patterlwalue);
        sscanf(mask,    "%x", patternMask);
        sscanf(size,    "%d", &signalSize);

        // Ilwert pattern mask to correct state
        *patternMask = ~*patternMask;

        // Check for invalid signal size value
        if ((signalSize < 1) || (signalSize > 32))
        {
            // Default to 16-bits if invalid size detected
            signalSize = 16;
        }
        // Colwert signal size to signal mask
        *signalMask = (1 << signalSize) - 1;
    }
    return result;
}

// Read cache for multi-signal optimization
static  LwU32   signalRegister;
static  LwBool  priReadCheck;

// readAndPrintSignalInfo: Reads the signal value from the register provided by PMLSplitter,
// parses the signal name and prints it into the output file. Also does checking for 0xBADFxxxx
// PRI reads, "sigdump_debug" and "static_pattern" signals (if checking is enabled).
static int readAndPrintSignalInfo
(
    FILE *fp,
    FILE *verifFp,
    LwBool signalValid,
    pmcsSigdump *dump,
    LwU32 i
)
{
    char chiplet[CHIPLET_NAME_MAX_LENGTH];
    char domainName[DOMAIN_NAME_MAX_LENGTH + DOMAIN_NUM_MAX_LENGTH];
    char domain[DOMAIN_NAME_MAX_LENGTH];
    char *p = domainName;
    char signal[PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH];
    char str[PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH];
    LwU32 chipletNum, instanceNum;
    LwU32 domainNum = 0;
    LwU32 j, k, lsb, msb, signalValue = 0;
    LwBool debugSignalCheck = 1;
    LwBool domainFound = 0;
    int readStatus = REG_READ_SUCCESSFUL;

    assert (CHIPLET_NAME_MAX_LENGTH < PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);
    assert ((DOMAIN_NAME_MAX_LENGTH + DOMAIN_NUM_MAX_LENGTH) < PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);

    strncpy (str, dump->reads[i].signalName, PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);
    // str has the format instanceClass.domainName_signalName[chiplet, instance]
    for (j = 0; str[j] != '\0'; j ++)
    {
        if (str[j] == '.' || str[j] == '[' || str[j] == ']' || str[j] == ':' || str[j] == ',')
        {
            str[j] = ' ';
        }
        else if (str[j] == '_' && str[j + 1] == '_')
        {
            if (!domainFound)
            {
                str[j] = ' ';
                str[j + 1] = ' ';

                domainFound = 1;
            }
        }
    }
    sscanf (str, "%s %s %s %u %u", chiplet, domainName, signal, &chipletNum, &instanceNum);

    // Separate out domain and domainNum from domainName.
    k = 0;
    while (*p)
    {
        if (isdigit (*p))
        {
            domainNum = strtol (p, &p, 10);
        }
        else
        {
            domain [k] = *p;
            k ++;
            p ++;
        }
    }
    domain [k] = '\0';

    // Compute next signal start/end bitfield positions
    lsb = dump->reads[i].lsbPosition;
    msb = lsb + dump->reads[i].width - 1;

    // If multi-signal optimization only read register once (Extract multiple signals after that)
    if (MULTI_SIGNAL_OPTIMIZATION)
    {
        // Check for initial register read (Only one actually performed)
        if (i == 0)
        {
            priReadCheck = 1;
            if (RegRead ( &signalRegister, dump->reads[i].address, PRI_CHECK_VERBOSE ) == LW_FALSE)
            {
                readStatus = REG_READ_FAILED;
                priReadCheck = 0;
            }
        }
        else    // Skip this register read
        {
            readStatus = REG_READ_SKIPPED;
        }
        // Extract next signal bitfield value
        if ((msb - lsb + 1) < 32)
        {
            signalValue = (signalRegister >> lsb) & ((1 << (msb - lsb + 1)) - 1);
        }
    }
    else    // Single signal read
    {
        priReadCheck = 1;
        if (RegBitfieldRead ( &signalValue, dump->reads[i].address, lsb, msb, PRI_CHECK_VERBOSE ) == LW_FALSE)
        {
            readStatus = REG_READ_FAILED;
            priReadCheck = 0;
        }
    }
    // Checkpoint: "sigdump_debug"/"static_pattern" matching.
    // (check only if CHECK_MARKER_VALUES is LW_TRUE)
    if (CHECK_MARKER_VALUES)
    {
        // If the signal name contains "sigdump_debug".
        if (strstr (signal, "_sigdump_debug") != NULL)
        {
            //
            // For turing, we treat sigdump_debug gpc/tpc/sm/lts ID signals as normal ones
            // and no longer compared to pre-defined static values. http://lwbugs/2022792/26
            //
            if (strstr(signal, "_id") == NULL)
            {
                debugSignalCheck = verifySigdumpDebugMarker (signal, signalValue);
            }
        }
        // If the signal name contains "static_pattern".
        else if (strstr (signal, "_static_pattern_") != NULL)
        {
            debugSignalCheck = verifyStaticPattern (signal, signalValue);
        }
    }

    fprintf ( fp,
              "%s  %d  %s  %d  %d  %s[%d:0]  0x%x",
              chiplet,
              chipletNum,
              domain,
              domainNum,
              instanceNum,
              signal,
              (dump->reads[i].width) - 1,
              signalValue );

    // If any of the validation checks have failed,
    // dump the signal with the remark "<POSSIBLY INVALID>".
    if (!(signalValid) || !(priReadCheck) || (!(debugSignalCheck) && CHECK_MARKER_VALUES))
    {
        fprintf (fp, "  <POSSIBLY INVALID>");
    }
    fprintf (fp, "\n");

    // If "-verify" argument was specified, write the
    // register read operation into the verifFp file.
    if (verifFp)
    {
        fprintf ( verifFp,
                  "OutputSignal (fp,\"%s: \", RegBitRead (0x%.8x,%d,%d));\n",
                  dump->reads[i].signalName,
                  dump->reads[i].address,
                  dump->reads[i].lsbPosition,
                 (dump->reads[i].lsbPosition) + (dump->reads[i].width) - 1 );
    }
    return readStatus;
}

// printGK110WarningHeader: Prints the warnings for GPC0 sigdump corruption for GK110.
static void printGK110WarningHeader ( void )
{
    dprintf("lw: =================================================================================="
        "=========================================\n");
    dprintf("lw:  SIGDUMP NOTE:\n");
    dprintf("lw:  SIGDUMP NOTE: SIGDUMP SIGNAL CONTENT MAY BE CORRUPTED BY GPU POWER FEATURES!!!  "
        "See GK110 Bug http://lwbugs/1028519 \n");
    dprintf("lw:  SIGDUMP NOTE:\n");
    dprintf("lw:  SIGDUMP NOTE: The following links provide details on how to statically disable "
        "lwrrently supported power features:\n");
    dprintf("lw:  SIGDUMP NOTE:  https://wiki.lwpu.com/engwiki/index.php/Resman/Resman_Components"
        "/LowPower/RegKeys\n");
    dprintf("lw:  SIGDUMP NOTE:  https://wiki.lwpu.com/gpuhwkepler/index.php/Emulation_Feature_"
        "Enablement_Plan#Regkey \n");
    dprintf("lw:  SIGDUMP NOTE:\n");
    dprintf("lw:  SIGDUMP NOTE: When providing sigdump results, be sure to inform consumers of power "
        "feature disablement (or lack thereof).\n");
    dprintf("lw:  SIGDUMP NOTE:\n");
    dprintf("lw: ===================================================================================="
        "=======================================\n");
}

// sigGetSigdump_GM107: This is the method ilwoked by LWWatch.
// It takes information provided by the command line arguments
// through LWWatch, as parameters.
void sigGetSigdump_GM107
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
    int signal;
    int reResult;
    char *progGuideElwPath = getelw ("LWW_PROG_GUIDE_SDK");
    LwU32 LwPmcBoot0RegVal = 0x0;
    LwU32 syspipeMask = 0x0;

    // Compile the debug signal regular expressions
    for (signal = 0; signal < countof(debugSignalTable); signal++)
    {
        reResult = regcomp(&debugSignalTable[signal].signalRegEx, debugSignalTable[signal].signalName, REG_EXTENDED);
        if (reResult != REG_NOERROR)
        {
            dprintf ("FATAL ERROR! Failed to compile debug signal regular expressions!\n");
            return;
        }
    }
    // Begin PMCS Session.
    if (pmcsBeginSession () != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to initialize the pmcs session. "
            "Error message: %s\n", pmcsGetLastErrorMessage ());
        return;
    }

    // Set chip name enum for PMLSplitter.
    LwPmcBoot0RegVal = GPU_REG_RD32 (LW_PMC_BOOT_0);
    if ((LwPmcBoot0RegVal == 0x0) || ((LwPmcBoot0RegVal & 0xFFFF0000) == 0xBADF0000))
    {
        dprintf ("FATAL ERROR! LW_PMC_BOOT_0 register read returned 0x%X\n",
            LwPmcBoot0RegVal);
        return;
    }

    // Extract chip architecture and implementation from the LW_PMC_BOOT_0 register.
    chipArchImpl = (DRF_VAL (_PMC, _BOOT_0, _ARCHITECTURE, LwPmcBoot0RegVal)
        << DRF_SIZE (LW_PMC_BOOT_0_IMPLEMENTATION)) | (DRF_VAL (_PMC, _BOOT_0, _IMPLEMENTATION, LwPmcBoot0RegVal));
    if (pmcsInputSetChip (chipArchImpl) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set the chip config in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    // Get chip name from PMLSplitter.
    if (pmcsOutputGetChipName (chipName, sizeof (chipName) ) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to get the chip name from PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    // Set output format to be sigdump format.
    if (pmcsInputSetOutputType (PMCS_OUTPUT_TYPE_SIGDUMP) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set output type as sigdump in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    // Set the ROOT path to the PM Programming Guide.
    if (progGuideElwPath == NULL)
    {
        dprintf (DEFAULT_PROG_GUIDE_PATH_MESSAGE, chipName);

        dprintf ("Loading programming guide ..\n");

        // This PMLSplitter lib function checks if the PM Programming Guide
        // exists at the path provided appended to the root of /chips/<chip_name>/.
        if (pmcsInputLoadProgGuideFromRoot ("./") != PMCS_SUCCESS)
        {
            dprintf ("FATAL ERROR! Failed to access the PM programming guide text file. "
                "Error details: %s\n", pmcsGetLastErrorMessage ());
            pmcsEndSession ();
            return;
        }
    }
    else
    {
        dprintf (SET_PROG_GUIDE_PATH_MESSAGE, progGuideElwPath, chipName);

        dprintf ("Loading programming guide ..\n");

        // This PMLSplitter lib function checks if the PM Programming Guide exists
        // at the path provided appended to the root of /chips/<chip_name>/.
        if (pmcsInputLoadProgGuideFromRoot (progGuideElwPath) != PMCS_SUCCESS)
        {
            dprintf ("FATAL ERROR! Failed to access the PM programming guide text file. "
                "Error details: %s\n", pmcsGetLastErrorMessage ());
            pmcsEndSession ();
            return;
        }
    }

    //
    // Is SMC is enabled, set SMC config.
    // Lwrrently we do not take input from user for syspipes and simply set
    // smc config to all syspipes
    //
    if(pGr[indexGpu].grGetSmcState())
    {
        syspipeMask = LWBIT32(MAX_GR_IDX) - 1;
    }
    else
    {
        syspipeMask = 0;
    }

    if (pmcsInputSetSMCConfig(syspipeMask) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set SMC Config for PmlSplitter."
                 "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    // Set some high chiplet and instance ranges. PMLSplitter will take care
    // of the maximum number of available chiplets and instances by itself.
    if (pmcsInputSetChipletRange (MIN_CHIPLET_NUM, MAX_CHIPLET_NUM) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set the chiplet range in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    if (pmcsInputSetInstanceRange (MIN_INSTANCE_NUM, MAX_INSTANCE_NUM) != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to set the instance range in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    // Get FS info for the chip and call PMLSplitter library methods to set them.
    if (getAndSetFs () == LW_FALSE)
    {
        dprintf ( "FATAL ERROR! Failure oclwrred in function: getAndSetFs().\n" );
    }

    // Print the sigdump legend.
    printSigdumpLegend (fp);

    // Print the FS config.
    printFSConfig (fp);

    // PRI read checks enabled by default (only counts).
    dprintf ("Checks on PRI reads for 0xBADFxxxx are enabled.\n");
    if (priCheckVerbose)
    {
        PRI_CHECK_VERBOSE = 1;
        dprintf ("Prints on 0xBADFxxxx PRI reads enabled.\n");
    }
    else
    {
        PRI_CHECK_VERBOSE = 0;
    }
    fprintf (fp, "Checks on PRI reads for 0xBADFxxxx are enabled.\n");
    fprintf (fp, "    Number of 0xBADFxxxx PRI reads are reported "
        "at the end of this file.\n");

    // Check if multi-signal optimization feature enabled.
    if (multiSignalOptimization)
    {
        MULTI_SIGNAL_OPTIMIZATION = 1;
        pmcsInputSigdumpEnableMultiSignalMux();
        dprintf ("Multi-signal optimization feature enabled.\n");
        fprintf (fp, "Multi-signal optimization feature enabled.\n");
    }
    else
    {
        MULTI_SIGNAL_OPTIMIZATION = 0;
    }
    // Polling until ENGINE_STATUS reads EMPTY is *always* enabled.
    // See http://lwbugs/1351292.
    dprintf ("ENGINE_STATUS polling enabled.\n");
    if (engineStatusVerbose)
    {
        ENGINE_STATUS_VERBOSE = 1;
        dprintf ("Prints on ENGINE_STATUS not going to EMPTY enabled.\n");
    }
    else
    {
        ENGINE_STATUS_VERBOSE = 0;
    }
    fprintf (fp, "ENGINE_STATUS polling enabled.\n");
    fprintf (fp, "    Number of cases where ENGINE_STATUS did not go to "
        "EMPTY are reported at the end of this file.\n");

    // Check if -verify command line arg is used.
    if (verifySigdump)
    {
        SIGDUMP_VERBOSE = 1;
        dprintf ("Sigdump verification enabled. \"sigdump_verif.txt\" will be generated.\n");
        fprintf (fp, "Sigdump verification enabled. \"sigdump_verif.txt\" will be generated.\n");
    }
    else
    {
        SIGDUMP_VERBOSE = 0;
    }
    // Check if RegWrite checks are enabled.
    if (regWriteCheck)
    {
        CHECK_REGWRITES = 1;
        dprintf ("Checks on register writes enabled.\n");
        fprintf (fp, "Checks on register writes enabled.\n");
    }
    else
    {
        CHECK_REGWRITES = 0;
    }
    // Check if "sigdump_debug" and "static_pattern" signal checks are enabled.
    if (markerValuesCheck)
    {
        CHECK_MARKER_VALUES = 1;
        dprintf ("Checks on \"sigdump_debug\" and \"static_pattern\" signal values enabled.\n");
        fprintf (fp, "Checks on \"sigdump_debug\" and \"static_pattern\" signal values enabled.\n");
        fprintf (fp, "    Number of mismatches are reported at the end of this file.\n");
    }
    else
    {
        CHECK_MARKER_VALUES = 0;
    }
    // Check if RegWrite optimized sigdump is enabled.
    if (regWriteOptimization)
    {
        SIGDUMP_OPTIMIZATION = 1;
        dprintf ("Optimized sigdump register writes enabled.\n");
        fprintf (fp, "Optimized sigdump register writes enabled.\n");
    }
    else
    {
        SIGDUMP_OPTIMIZATION = 0;
    }
    // Check if selective sigdump (filtering signals by "chiplet") is enabled.
    if (chipletKeyword != NULL)
    {
        CHIPLET_FILTER = 1;
        dprintf ("Selective sigdump enabled. Signals will be filtered using the keyword: "
            "chiplet = %s.\n", chipletKeyword);
        fprintf (fp, "Selective sigdump enabled. Signals will be filtered using the keyword: "
            "chiplet = %s.\n", chipletKeyword);
    }
    else
    {
        CHIPLET_FILTER = 0;
    }
    // Check if selective sigdump (filtering signals by "chiplet#") is enabled.
    if (chipletNumKeyword != NULL)
    {
        CHIPLET_NUM_FILTER = 1;
        dprintf ("Selective sigdump enabled. Signals will be filtered using the keyword: "
            "chiplet# = %s.\n", chipletNumKeyword);
        fprintf (fp, "Selective sigdump enabled. Signals will be filtered using the keyword: "
            "chiplet# = %s.\n", chipletNumKeyword);
    }
    else
    {
        CHIPLET_NUM_FILTER = 0;
    }
    // Check if selective sigdump (filtering signals by "domain") is enabled.
    if (domainKeyword != NULL)
    {
        DOMAIN_FILTER = 1;
        dprintf ("Selective sigdump enabled. Signals will be filtered using the keyword: "
            "domain = %s.\n", domainKeyword);
        fprintf (fp, "Selective sigdump enabled. Signals will be filtered using the keyword: "
            "domain = %s.\n", domainKeyword);
    }
    else
    {
        DOMAIN_FILTER = 0;
    }
    // Check if selective sigdump (filtering signals by "domain#") is enabled.
    if (domainNumKeyword != NULL)
    {
        DOMAIN_NUM_FILTER = 1;
        dprintf ("Selective sigdump enabled. Signals will be filtered using the keyword: "
            "domain# = %s.\n", domainNumKeyword);
        fprintf (fp, "Selective sigdump enabled. Signals will be filtered using the keyword: "
            "domain# = %s.\n", domainNumKeyword);
    }
    else
    {
        DOMAIN_NUM_FILTER = 0;
    }
    // Check if selective sigdump (filtering signals by "instance#") is enabled.
    if (instanceNumKeyword != NULL)
    {
        INSTANCE_NUM_FILTER = 1;
        dprintf ("Selective sigdump enabled. Signals will be filtered using the keyword: "
            "instance# = %s.\n", instanceNumKeyword);
        fprintf (fp, "Selective sigdump enabled. Signals will be filtered using the keyword: "
            "instance# = %s.\n", instanceNumKeyword);
    }
    else
    {
        INSTANCE_NUM_FILTER = 0;
    }
    // Check for selective sigdump requested
    if (CHIPLET_FILTER || CHIPLET_NUM_FILTER || DOMAIN_FILTER || DOMAIN_NUM_FILTER || INSTANCE_NUM_FILTER)
    {
        SIGDUMP_SELECTIVE = 1;
    }
    else
    {
        SIGDUMP_SELECTIVE = 0;
    }
    // Run the method to get signal programming information from PMLSplitter
    // and do the required register write and reads.
    sigGetSigdumpFloorsweep ( fp,
                              chipletKeyword,
                              chipletNumKeyword,
                              domainKeyword,
                              domainNumKeyword,
                              instanceNumKeyword );

    // End the PMCS Session.
    if (pmcsEndSession () != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to close the pmcs session.\n");
        return;
    }
    // Free the debug signal regular expressions
    for (signal = 0; signal < countof(debugSignalTable); signal++)
    {
        regfree(&debugSignalTable[signal].signalRegEx);
    }
}

// printStaticPatternSignalInfo: Prints the number of "static_pattern" signal
// mismatches onto the display and the output file.
static void printStaticPatternSignalInfo (FILE *fp)
{
    if (staticPatternSignalFailures)
    {
        dprintf ( "    FAILED! %u/%u mismatches!\n",
            staticPatternSignalFailures,
            numberOfStaticPatternSignals );
    }
    else
    {
        dprintf ( "    PASSED! %u/%u signals matched!\n",
            ( numberOfStaticPatternSignals - staticPatternSignalFailures ),
            numberOfStaticPatternSignals );
    }
    fprintf ( fp, "Number of static pattern mismatches: %u/%u\n",
        staticPatternSignalFailures,
        numberOfStaticPatternSignals );
}

// printSigdumpDebugSignalInfo: Prints the number of "sigdump_debug"
// signal mismatches onto the display and the output file.
static void printSigdumpDebugSignalInfo (FILE *fp)
{
    if (markerSignalFailures)
    {
        dprintf ( "    FAILED! %u/%u mismatches!\n",
            markerSignalFailures,
            numberOfFixedMarkerSignals );
    }
    else
    {
        dprintf ( "    PASSED! %u/%u signals matched!\n",
            ( numberOfFixedMarkerSignals - markerSignalFailures ),
            numberOfFixedMarkerSignals );
    }
    fprintf ( fp, "Number of marker signal mismatches: %u/%u\n",
        markerSignalFailures,
        numberOfFixedMarkerSignals );
}

// filterSignal: Filter a signal name by keyword (s).
static LwBool filterSignal
(
    char *signalName,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
) {
    char chiplet[CHIPLET_NAME_MAX_LENGTH];
    char domainName[DOMAIN_NAME_MAX_LENGTH+DOMAIN_NUM_MAX_LENGTH];
    char domain[DOMAIN_NAME_MAX_LENGTH];
    char *p = domainName;
    char signal[PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH];
    char str[PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH];
    LwU32 chipletNum, instanceNum, chipletNumKeywordUInt32, domainNumKeywordUInt32, instanceNumKeywordUInt32;
    LwU32 domainNum = 0;
    LwU32 j, k;

    assert (CHIPLET_NAME_MAX_LENGTH < PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);
    assert ((DOMAIN_NAME_MAX_LENGTH + DOMAIN_NUM_MAX_LENGTH) < PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);

    strncpy (str, signalName, PMCS_SIGDUMP_MAX_SIGNAL_NAME_LENGTH);
    // str has the format instanceClass.domainName_signalName[chiplet, instance]
    for (j = 0; str[j] != '\0'; j ++)
    {
        if (str[j] == '.' || str[j] == '[' || str[j] == ']' || str[j] == ':' || str[j] == ',')
        {
            str[j] = ' ';
        }
        else if (str[j] == '_' && str[j+1] == '_')
        {
            str[j] = ' ';
            str[j + 1] = ' ';
        }
    }
    sscanf (str, "%s %s %s %u %u", chiplet, domainName, signal, &chipletNum, &instanceNum);

    // Separate out domain and domainNum from domainName.
    k = 0;
    while (*p)
    {
        if (isdigit (*p))
        {
            domainNum = strtol (p, &p, 10);
        }
        else
        {
            domain [k] = *p;
            k ++;
            p ++;
        }
    }
    domain [k] = '\0';

    // Apply the filters.
    if (CHIPLET_FILTER)
    {
        if (strcmp (chiplet, chipletKeyword))
        {
            return LW_FALSE;
        }
    }
    if (CHIPLET_NUM_FILTER)
    {
        sscanf (chipletNumKeyword, "%u", &chipletNumKeywordUInt32);
        if (chipletNumKeywordUInt32 != chipletNum)
        {
            return LW_FALSE;
        }
    }
    if (DOMAIN_FILTER)
    {
        if (strcmp (domain, domainKeyword))
        {
            return LW_FALSE;
        }
    }
    if (DOMAIN_NUM_FILTER)
    {
        sscanf (domainNumKeyword, "%u", &domainNumKeywordUInt32);
        if (domainNumKeywordUInt32 != domainNum)
        {
            return LW_FALSE;
        }
    }
    if (INSTANCE_NUM_FILTER)
    {
        sscanf (instanceNumKeyword, "%u", &instanceNumKeywordUInt32);
        if (instanceNumKeywordUInt32 != instanceNum)
        {
            return LW_FALSE;
        }
    }
    return LW_TRUE;
}

// sigGetSigdumpFloorsweep: Gets PM signal programming information
// from PMLSplitter and calls methods to do the register writes/reads.
static void sigGetSigdumpFloorsweep
(
    FILE *fp,
    char *chipletKeyword,
    char *chipletNumKeyword,
    char *domainKeyword,
    char *domainNumKeyword,
    char *instanceNumKeyword
) {
    LwU32               i;
    LwU32               ilwalidRegWrites = 0;
    LwU32               numWrites = 0;
    LwU32               numReads = 0;
    LwU32               numSignals = 0;
    int                 regWriteStatus;
    int                 regReadStatus;
    LwBool              signalValid[PMCS_SIGDUMP_MAX_REG_READS];
    LwBool              engineStatusPollPassed;
    LwBool              matchFound;

    // Defining PMLSplitter related variables.
    pmcsSigdump         dump;
    pmcsReturn          pmcsReturnCode;

    FILE*               verifFp = NULL;

    // Reset all counters.
    badPriReadCount = 0;
    engineStatusFailureCount = 0;
    numberOfFixedMarkerSignals = 0;
    markerSignalFailures = 0;
    staticPatternSignalFailures = 0;
    numberOfStaticPatternSignals = 0;

    if (SIGDUMP_VERBOSE)
    {
        verifFp = fopen ("sigdump_verif.txt", "w");
        if (verifFp == NULL)
        {
            dprintf ("FATAL ERROR! Could not open file \"sigdump_verif.txt\" to write to!\n");
            return;
        }
    }

    printHeader (fp);

    dprintf ("Initializing data from programming guide ..\n");

    // Code derived from PMLSplitter pages
    // (https://app-perf/home/app_perf_catalog/pmlsplitter/docs/html/sigdump.html).
    pmcsReturnCode = pmcsOutputSigdumpStartIteration ();
    if (pmcsReturnCode != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to start sigdump iteration in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    pmcsReturnCode = pmcsOutputSigdumpHasNextSignal ();
    if (pmcsReturnCode != PMCS_SUCCESS)
    {
        dprintf ("FATAL ERROR! Failed to check for next signal in PMLSplitter. "
            "Error details: %s\n", pmcsGetLastErrorMessage ());
        pmcsEndSession ();
        return;
    }

    if (SIGDUMP_OPTIMIZATION)
    {
        clearRegWriteCache ();
    }

    dprintf ("Starting to dump signals ..\n");

    while (pmcsOutputSigdumpHasNextSignal () == PMCS_SUCCESS)
    {
        pmcsOutputSigdumpGetNextSignal (&dump);
        // Check if signal filtering is enabled using the command-line
        // arguments "-chiplet", "-chipletNum", "-domain", "-domainName" or "-instanceNum".
        if (SIGDUMP_SELECTIVE)
        {
            matchFound = 1;
            for (i = 0; i < dump.numReads; ++i)
            {
                matchFound = filterSignal ( dump.reads[i].signalName,
                                            chipletKeyword,
                                            chipletNumKeyword,
                                            domainKeyword,
                                            domainNumKeyword,
                                            instanceNumKeyword );
                if (matchFound)
                {
                    break;
                }
            }
            // If the signal was filtered out, skip the rest of the
            // while () loop iteration and continue.
            if (!matchFound)
            {
                continue;
            }
        } // Done with signal filtering.
        engineStatusPollPassed = LW_TRUE;
        for (i = 0; i < dump.numReads; ++i)
            signalValid[i] = LW_TRUE;

        for (i = 0; i < dump.numWrites; ++i)
        {
            // If "-unoptimized" argument is used, do not use write
            // caching logic and use the RegWrite method.
            regWriteStatus = optimizedRegWriteWrapper ( &dump,
                                                        i,
                                                        SIGDUMP_OPTIMIZATION,
                                                        CHECK_REGWRITES );
            // If register write was not skipped, increment counter.
            if (regWriteStatus != REG_WRITE_SKIPPED)
            {
                numWrites ++;
            }
            if (regWriteStatus == REG_WRITE_SUCCESSFUL)
            {
                // If "-verify" argument was specified and this write was not skipped,
                // write the register write operation into the verifFp file.
                if (verifFp != NULL)
                {
                    fprintf ( verifFp, "RegWrite(0x%.8x,0x%.8x,0x%.8x);\n",
                        dump.writes[i].address,
                        dump.writes[i].value,
                        dump.writes[i].mask );
                }
            }
            else if (regWriteStatus == REG_WRITE_CHECK_FAILED)
            {
                ilwalidRegWrites ++;

                // Display the signal name and value that may have
                // been corrupted due to incomplete writes.
                if (MULTI_SIGNAL_OPTIMIZATION)
                {
                    signalValid[dump.writes[i].signalNumber] = LW_FALSE;
                }
                else
                    signalValid[0] = LW_FALSE;
            }
        }
        // Read engine status.
        if (dump.engine_status_valid)
        {
            engineStatusPollPassed = pollEngineStatusEmpty (&dump);
        }
        // WAR for GM10x chips. See http://lwbugs/1260703 for more information.
        // This replicates the functionality of checkPmmStatus ().
        else
        {
            if (MULTI_SIGNAL_OPTIMIZATION)
            {
                signalValid[dump.writes[dump.numWrites - 1].signalNumber] = checkRegWrite (&dump, dump.numWrites - 1);
            }
            else
                signalValid[0] = checkRegWrite (&dump, dump.numWrites - 1);
        }
        // Increment the number of signals (Number of reads but that may be optimized)
        numSignals += dump.numReads;

        // signalValid is LW_FALSE if either the ENGINE_STATUS polling didn't work
        // or if checkRegWrite () failed for this signal.
        for (i = 0; i < dump.numReads; ++i)
        {
            regReadStatus = readAndPrintSignalInfo (fp, verifFp, signalValid[i] & engineStatusPollPassed, &dump, i);

            // If register read was not skipped, increment counter.
            if (regReadStatus != REG_READ_SKIPPED)
            {
                numReads ++;
            }
            if (regReadStatus == REG_READ_FAILED)
            {
                badPriReadCount ++;
            }
            if ((regReadStatus == REG_READ_FAILED) || !(signalValid[i] & engineStatusPollPassed))
            {
                dprintf ( "WARNING: Read value of signal \"%s\" (Address = 0x%08x, "
                    "LSB = %d, MSB = %d) may be invalid.\n",
                    dump.reads[i].signalName,
                    dump.reads[i].address,
                    dump.reads[i].lsbPosition,
                    (dump.reads[i].lsbPosition + dump.reads[i].width - 1) );
            }
        }
    }
    fprintf (fp, "\n");

    dprintf ("Number of writes: %u\n", numWrites);
    fprintf (fp, "Number of writes: %u\n", numWrites);
    if (CHECK_REGWRITES)
    {
        dprintf ("    Number of invalid writes: %d\n", ilwalidRegWrites);
        fprintf (fp, "    Number of invalid writes: %d\n", ilwalidRegWrites);
    }
    dprintf ("Number of reads: %d\n", numReads);
    fprintf (fp, "Number of reads: %d\n", numReads);
    dprintf ("Number of signals: %d\n", numSignals);
    fprintf (fp, "Number of signals: %d\n", numSignals);

    dprintf ("Number of 0xBADFxxxx PRI reads: %u\n", badPriReadCount);
    fprintf (fp, "Number of 0xBADFxxxx PRI reads: %u\n", badPriReadCount);
    dprintf ("Number of cases where ENGINE_STATUS did not read EMPTY: %u\n", engineStatusFailureCount);
    fprintf (fp, "Number of cases where ENGINE_STATUS did not read EMPTY: %u\n", engineStatusFailureCount);
    if (verifFp)
    {
        fclose (verifFp);
    }

    if (CHECK_MARKER_VALUES)
    {
        if (numberOfStaticPatternSignals)
        {
            dprintf ("Verification of \"static_pattern\" signals:\n");
            printStaticPatternSignalInfo (fp);
        }
        else
        {
            dprintf ("WARNING: The \"static_pattern\" signals were not dumped!\n");
        }
        if (numberOfFixedMarkerSignals)
        {
            dprintf ("Verification of \"sigdump_debug\" signals:\n");
            printSigdumpDebugSignalInfo (fp);
        }
        else
        {
            dprintf ("WARNING: The \"sigdump_debug\" signals were not dumped!\n");
        }
    }
}

#endif  // SIGDUMP_ENABLE
