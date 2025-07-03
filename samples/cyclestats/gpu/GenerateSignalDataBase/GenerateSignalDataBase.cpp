 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lw32.h>
#include <lwos.h>
#include <lwcm.h>

#include <vector>
#include "lzmaEnc920/LzmaEnc.h"

#include "../../../drivers/common/cyclestats/gpu/lwPerfMonHW.h"
#include "../../../drivers/common/cyclestats/gpu/lwPerfMonCompressedTable.h"
#include "lwPerfMonTableCompress.h"

// command line globals
bool g_compress                     = false;
bool g_minimizeCppFileSize          = false;
const char *g_pDataBaseSymbolName   = "g_PMMuxSignalDataBase";

LwU32 getChipletIndexdFromChipletName(const char *pChipletName)
{
    if (strcmp(pChipletName, "sys") == 0) return 0;
    if (strcmp(pChipletName, "gpc") == 0) return 1;
    if (strcmp(pChipletName, "fbp") == 0) return 2;

    // gpu not supported?!?
    assert(0);
    return ~0;
}

LwU32 getInstanceTypeFromInstanceTypeName(const char *pInstanceTypeName)
{
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_NONE")           == 0) return LwSignalDataBase::PM_IT_NONE;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_TPC")            == 0) return LwSignalDataBase::PM_IT_TPC;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_GPC_PPC")        == 0) return LwSignalDataBase::PM_IT_GPC_PPC;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_WXBAR_CQ_DAISY") == 0) return LwSignalDataBase::PM_IT_WXBAR_CQ_DAISY;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_WXBAR_CS_DAISY") == 0) return LwSignalDataBase::PM_IT_WXBAR_CS_DAISY;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_MXBAR_CS_DAISY") == 0) return LwSignalDataBase::PM_IT_MXBAR_CS_DAISY;
    if (strcmp(pInstanceTypeName, "LwSignalDataBase::PM_IT_CXBAR_CQ_DAISY") == 0) return LwSignalDataBase::PM_IT_CXBAR_CQ_DAISY;

    // type not supported?!?
    assert(0);
    return ~0;
}

LwU32 getGpudIdFromGpuName(const char *pGpuName, const char *pGpuOverrideName)
{
    // all GK110B/GK110C to use GK110 signals
    if (pGpuOverrideName)
    {
        pGpuName = pGpuOverrideName;
    }

    // [Maxwell]
    {
        if (strcmp(pGpuName, "gm107")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM000 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM107;
        if (strcmp(pGpuName, "gm108")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM000 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM108;
        if (strcmp(pGpuName, "gm200")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM200;
        if (strcmp(pGpuName, "gm204")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM204;
        if (strcmp(pGpuName, "gm206")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM206;
        if (strcmp(pGpuName, "gm20b")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM20B;
        if (strcmp(pGpuName, "gm21b")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GM200 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GM21B;
    }

    // [Pascal]
    {
        if (strcmp(pGpuName, "gp100")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP100;
        if (strcmp(pGpuName, "gp102")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP102;
        if (strcmp(pGpuName, "gp104")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP104;
        if (strcmp(pGpuName, "gp106")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP106;
        if (strcmp(pGpuName, "gp107")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP107;
        if (strcmp(pGpuName, "gp108")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP108;
        if (strcmp(pGpuName, "gp10b")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GP100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GP10B;
    }

    // [Volta]
    {
        if (strcmp(pGpuName, "gv100")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GV100;
        if (strcmp(pGpuName, "gv11b")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GV110 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GV11B;
    }

    // [Turing]
    {
        if (strcmp(pGpuName, "tu102")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_TU102;
        if (strcmp(pGpuName, "tu104")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_TU104;
        if (strcmp(pGpuName, "tu106")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_TU106;
        if (strcmp(pGpuName, "tu116")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_TU116;
        if (strcmp(pGpuName, "tu117")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_TU100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_TU117;
    }

    // [Ampere]
    {
        if (strcmp(pGpuName, "ga100")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA100;
        if (strcmp(pGpuName, "ga102")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA102;
        if (strcmp(pGpuName, "ga103")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA103;
        if (strcmp(pGpuName, "ga104")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA104;
        if (strcmp(pGpuName, "ga106")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA106;
        if (strcmp(pGpuName, "ga107")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA107;
        if (strcmp(pGpuName, "ga10b")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA10B;
        if (strcmp(pGpuName, "ga10f")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GA100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GA10F;
    }

    // [Ada]
    {
        if (strcmp(pGpuName, "ad102")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_AD100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_AD102;
    }

    // [Hopper]
    {
        if (strcmp(pGpuName, "gh100")  == 0) return LW2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_GH100 | LW2080_CTRL_MC_ARCH_INFO_IMPLEMENTATION_GH100;
    }

    // gpu not supported?!?
    assert(0);
    return ~0;
}

// finds + init new LwSignalDataBase slot

LwSignalDataBase::GPUChipletSignalTable *findOrCreateGPUChipletSignalTable(LwSignalDataBase::GPUSignalTable *pPmSignalTable, LwU32 gpuId, LwU32 chipletId, LwU32 domainId)
{
    // first use?
    if (!pPmSignalTable->gpuId)
    {
        memset(pPmSignalTable, 0, sizeof(*pPmSignalTable));
        pPmSignalTable->gpuId = gpuId;
    }

    // double check
    if (pPmSignalTable->gpuId != gpuId)
    {
        assert(0);
        return NULL;
    }

    // search for existing {chiplet, domainIndex} entry
    for (LwU32 chipletTableIndex=0; chipletTableIndex<LwSignalDataBase::MAX_CHIPLET_TYPE_COUNT; chipletTableIndex++)
    {
        // reached end of list? need to cast away the "const" here :(
        LwSignalDataBase::GPUChipletSignalTable *pChipletTable = (LwSignalDataBase::GPUChipletSignalTable*) &pPmSignalTable->chipletTable[chipletTableIndex];
        if (!pChipletTable->pCoreSignals && !pChipletTable->pMuxSignals)
        {
            // create a new entry
            memset(pChipletTable, 0, sizeof(*pChipletTable));
            pChipletTable->chipletId = chipletId;
            pChipletTable->domainId  = domainId;
        }

        // matching existing entry
        if ((pChipletTable->chipletId == chipletId) && (pChipletTable->domainId == domainId))
        {
            return pChipletTable;
        }
    }
    assert(0);
    return NULL;
}

// "ZeroOrLog2" helper

LwU32 decodeZeroOrLog2(LwU32 v)
{
    return v ? (1<<v) : 0x0;
}

// "ZeroOrLog2" helper

LwU32 encodeZeroOrLog2(LwU32 v)
{
    // input must be zero or power-of-2. encoding:
    //      v == 0       => 0x0
    //      v == pow2()  => log2(v);
    LwU32 zeroOrlog2 = 0;
    _BitScanForward(&zeroOrlog2, v);
    assert(decodeZeroOrLog2(zeroOrlog2) == v);
    return zeroOrlog2;
}

// reads a text file into memory

class DataReader
{
    protected:
        // pointer to file's allocated memory. Freed automatically
        FILE   *m_pFile;
        char   *m_pBuf;
        char   *m_pEnd;
        size_t  m_fileSize;

        // ughh, sscanf() is really slow because it does a strlen on the input buffer => takes forever if the input string is like 50MB ...
        const LwU32 MAX_BYTES_PER_SNSCANF = 256;

    public:
        DataReader()
        {
            m_pFile    = NULL;
            m_pBuf     = NULL;
            m_pEnd     = NULL;
            m_fileSize = 0;
        }

        ~DataReader()
        {
            releaseAllResources();
        }

        char *read(const char *pFileName)
        {
            // allow reuse => release any old stuff
            releaseAllResources();

            if (fopen_s(&m_pFile, pFileName, "rb") == 0)
            {
                // determine file size (seek to end end get the filepos, the go back to start)
                if (fseek(m_pFile, 0, SEEK_END) == 0)
                {
                    m_fileSize = ftell(m_pFile);
                    if (fseek(m_pFile, 0, SEEK_SET) == 0)
                    {
                        // allocate temporary space
                        m_pBuf = new char[m_fileSize+1];
                        if (m_pBuf)
                        {
                            memset(m_pBuf, 0, m_fileSize+1);
                            printf("reading data from %s ...\n", pFileName);
                            size_t bytesTotal = fread(m_pBuf, 1, m_fileSize, m_pFile);
                            if (bytesTotal == m_fileSize)
                            {
                                m_pEnd = m_pBuf + m_fileSize;
                                return m_pBuf;
                            }
                        }
                    }
                }
            }
            printf("Failed reading file %s\n", pFileName);
            releaseAllResources();
            return NULL;
        }


        void releaseAllResources()
        {
            if (m_pBuf)
            {
                delete[] m_pBuf;
                m_pBuf = NULL;
                m_pEnd = NULL;
            }
            if (m_pFile)
            {
                fclose(m_pFile);
                m_pFile = NULL;
            }
            m_fileSize = 0;
        }

        void skipWhiteSpaces(char **pp)
        {
            // skip space/tabes
            char *p = *pp;
            for(;p<m_pEnd; p++)
            {
                if ((*p == ' ') || (*p == '\t')  || (*p == '\r') || (*p == '\n'))
                {
                    continue;
                }
                break;
            }
            *pp = p;
        }

        void skipComments(char **pp)
        {
            skipWhiteSpaces(pp);

            // skip entire line if it begins with //              
            char *p = *pp;
            for(;;)
            {
                if ((p[0] == '/') && (p[1] == '/'))
                {
                    p += 2;
                    for(;p<m_pEnd; p++)
                    {
                        if ((*p != '\n'))
                        {
                            continue;
                        }
                        p++;
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
            *pp = p;
        }
};

class MuxPerfmonDataReader : public DataReader
{
    public:
        MuxPerfmonDataReader()
        {
        }
        
        bool read(LwSignalDataBase::GPUSignalTable *ppPmSignalTable, const char *pFileName, const char *pGpuOverrideName)
        {
            if (!DataReader::read(pFileName))
            {
                return false;
            }

            // we'll compact muxSignals once we know how many signals we got total
            std::vector<LwSignalDataBase::MUXedSignal> muxSignals;

            // TODO: quick'n'dirty parse. maybe replace with real flex/bison parser...
            size_t totalSignals = 0;
            for(char *p=m_pBuf; p<m_pEnd;)
            {
                int bytesRead;

                // extract "static const LwSignalDataBase::MUXedSignal g_tu102_sys_domain0_signals[] ="
                char gpuName[256] = "";
                char chipletName[256] = "";
                char domainName[256] = "";
                bytesRead = 0;
                _snscanf(p, MAX_BYTES_PER_SNSCANF, " static const LwSignalDataBase::MUXedSignal g_%255[a-z0-9]_%255[a-z0-9]_%255[a-z0-9]_signals[] = { %n", gpuName, chipletName, domainName, &bytesRead);
                if (bytesRead)
                {
                    p += bytesRead;
                }
                else
                {
                    assert(0);
                    return false;
                }

                // extract "// perfmon_sys:0 = host0" data
                char chipletName2[256] = "";
                unsigned int domainIndex = ~0;
                char domainName2[256] = "";
                bytesRead = 0;
                _snscanf(p, MAX_BYTES_PER_SNSCANF, " // perfmon_%255[a-z0-9]:%i = %255[a-z0-9] %n", chipletName2, &domainIndex, domainName2, &bytesRead);
                if (bytesRead)
                {
                    p += bytesRead;
                }
                else
                {
                    assert(0);
                    return false;
                }

                // chiplet must match
                if (strcmp(chipletName, chipletName2) != 0)
                {
                    assert(0);
                    return false;
                }

                // decode
                LwU32 gpuId = getGpudIdFromGpuName(gpuName, pGpuOverrideName);
                if (gpuId == ~0)
                {
                    return false;
                }
                LwU32 chipletId = getChipletIndexdFromChipletName(chipletName);
                if (chipletId == ~0)
                {
                    return false;
                }

                printf("parsing mux perfmon signals for gpuId=0x%x, chipletId=%d, domainName=%s\n", gpuId, chipletId, domainName2);

                // avoid large and slow malloc/frees
                muxSignals.clear();

                // extract list of mux signals
                //      {
                //          {"host0.false", 1, {{"host0_sigval_zero"}}},
                //          ...
                //          {"host2pm_ext_bind_memop_issue", 1, {{"host2pm_host_host_muxed_bus_3", {{0x1A00, 24, 30, 20, 0x0, 0x0}}}}},
                //          {"host2pm_ext_bind_memop_throttle", 2, {{"host2pm_host_host_muxed_bus_6", {{0x1A00, 24, 30, 19, 0x0, 0x0}}}, {"host2pm_host_host_muxed_bus_0", {{0x1A00, 24, 30, 20, 0x0, 0x0}}}}},
                //          ...
                //          {NULL}
                //      }
                for(;p<m_pEnd;)
                {
                    // end of perfmon table? "{NULL}"
                    bytesRead = 0;
                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " { NULL } } ; %n", &bytesRead);
                    if (bytesRead)
                    {
                        p += bytesRead;
                        break;
                    }

                    // new mux perfmon signal
                    //      "{"host0.false", 1, {"
                    char muxSignalName[256] = "";
                    LwU32 setupCount = ~0;
                    bytesRead = 0;
                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " { \"%255[a-z0-9_.]\" , %i , { %n", muxSignalName, &setupCount, &bytesRead);
                    if (bytesRead)
                    {
                        p += bytesRead;

                        // add new node
                        muxSignals.emplace_back();
                        LwSignalDataBase::MUXedSignal *pMuxSignal = &muxSignals.back();
                        memset(pMuxSignal, 0, sizeof(*pMuxSignal));

                        // save parsed values
                        strncpy(pMuxSignal->name, muxSignalName, sizeof(pMuxSignal->name)-1);
                        pMuxSignal->setupCount = setupCount;

                        // parse a list of setups
                        //      {"host0_sigval_zero"}
                        //      {"host2pm_host_host_muxed_bus_3", {{0x1A00, 24, 30, 20, 0x0, 0x0}}}
                        //      {"host2pm_host_host_muxed_bus_6", {{0x1A00, 24, 30, 19, 0x0, 0x0}}}, {"host2pm_host_host_muxed_bus_0", {{0x1A00, 24, 30, 20, 0x0, 0x0}}}}},
                        LwU32 lwrrentSetupCount = 0;
                        for(;;)
                        {
                            char rawSignalName[256] = "";
                            bytesRead = 0;
                            _snscanf(p, MAX_BYTES_PER_SNSCANF, " { \"%255[a-z0-9_.]\" %n", rawSignalName, &bytesRead);
                            if (bytesRead)
                            {
                                p += bytesRead;

                                // enough space left?
                                LwSignalDataBase::MUXedSignalSetup *pMUXedSignalSetup = &pMuxSignal->setup[lwrrentSetupCount++];
                                if (lwrrentSetupCount >= LwSignalDataBase::MAX_SETUP_COUNT)
                                {
                                    assert(0);
                                    return false;
                                }

                                // save parsed values
                                pMUXedSignalSetup->pName = _strdup(rawSignalName);

                                // any mux info?
                                bytesRead = 0;
                                _snscanf(p, MAX_BYTES_PER_SNSCANF, " ,  { %n", &bytesRead);
                                if (bytesRead)
                                {
                                    p += bytesRead;

                                    // parse a list of muxes
                                    LwU32 lwrrentMuxSelectCount = 0;
                                    for(;;)
                                    {
                                        LwU32 offset           = ~0;
                                        LwU32 firstBit         = ~0;
                                        LwU32 lastBit          = ~0;
                                        LwU32 val              = ~0;
                                        LwU32 offsetChipletInc = ~0;
                                        LwU32 offsetInstInc    = ~0;
                                        bytesRead = 0;
                                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " { %i , %i , %i , %i , %i , %i} %n", &offset, &firstBit, &lastBit, &val, &offsetChipletInc, &offsetInstInc, &bytesRead);
                                        if (bytesRead)
                                        {
                                            p += bytesRead;
                                        }
                                        else
                                        {
                                            assert(0);
                                            return false;
                                        }

                                        // save mux select info
                                        LwSignalDataBase::MultiplexerSelectInfo *pMuxSelectInfo = &pMUXedSignalSetup->mux[lwrrentMuxSelectCount++];
                                        if (lwrrentMuxSelectCount > LwSignalDataBase::MAX_MUX_SELECT_COUNT)
                                        {
                                            assert(0);
                                            return false;
                                        }
                                        pMuxSelectInfo->offset           = offset;
                                        pMuxSelectInfo->firstBit         = firstBit;
                                        pMuxSelectInfo->lastBit          = lastBit;
                                        pMuxSelectInfo->val              = val;

                                        // store just the log2 value we can get everything into 8 bytes
                                        pMuxSelectInfo->offsetChipletInc_zeroOrlog2 = encodeZeroOrLog2(offsetChipletInc);
                                        pMuxSelectInfo->offsetInstInc_zeroOrlog2    = encodeZeroOrLog2(offsetInstInc);

                                        // make sure the bitfield didn't drop anybits
                                        if ((pMuxSelectInfo->offset                                        != offset)           || 
                                            (pMuxSelectInfo->firstBit                                      != firstBit)         || 
                                            (pMuxSelectInfo->lastBit                                       != lastBit)          || 
                                            (pMuxSelectInfo->val                                           != val)              || 
                                            (decodeZeroOrLog2(pMuxSelectInfo->offsetChipletInc_zeroOrlog2) != offsetChipletInc) || 
                                            (decodeZeroOrLog2(pMuxSelectInfo->offsetInstInc_zeroOrlog2)    != offsetInstInc)) 
                                        {
                                            assert(0);
                                            return false;
                                        }

                                        // more entries?
                                        bytesRead = 0;
                                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " , %n", &bytesRead);
                                        if (bytesRead)
                                        {
                                            p += bytesRead;
                                            continue;
                                        }

                                        // no more mux info
                                        bytesRead = 0;
                                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " } %n", &bytesRead);
                                        if (bytesRead)
                                        {
                                            p += bytesRead;
                                            break;
                                        }
                                        else
                                        {
                                            assert(0);
                                            return false;
                                        }
                                    }
                                }

                                // end of current setup
                                bytesRead = 0;
                                _snscanf(p, MAX_BYTES_PER_SNSCANF, " } %n", &bytesRead);
                                if (bytesRead)
                                {
                                    p += bytesRead;
                                }
                                else
                                {
                                    assert(0);
                                    return false;
                                }

                                // another setup?
                                bytesRead = 0;
                                _snscanf(p, MAX_BYTES_PER_SNSCANF, " , %n", &bytesRead);
                                if (bytesRead)
                                {
                                    p += bytesRead;
                                    continue;
                                }
                                break;
                            }
                            else
                            {
                                assert(0);
                                return false;
                            }
                        }

                        // end of all setups
                        bytesRead = 0;
                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " } %n", &bytesRead);
                        if (bytesRead)
                        {
                            p += bytesRead;
                        }
                        else
                        {
                            assert(0);
                            return false;
                        }

                        if (pMuxSignal->setupCount != lwrrentSetupCount)
                        {
                            assert(0);
                            return false;
                        }

                        // end of muxsignal
                        bytesRead = 0;
                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " } , %n", &bytesRead);
                        if (bytesRead)
                        {
                            p += bytesRead;
                            continue;
                        }
                        else
                        {
                            assert(0);
                            return false;
                        }
                    }
                    else
                    {
                        assert(0);
                        return false;
                    }
                }

                printf("parsed %zd muxed perfmon signals\n", muxSignals.size());
                totalSignals += muxSignals.size();

                // compact pMuxSignals
                LwSignalDataBase::MUXedSignal *pCompactedMuxSignals = new LwSignalDataBase::MUXedSignal[muxSignals.size()+1];
                if (!pCompactedMuxSignals)
                {
                    assert(0);
                    return false;
                }
                // copy one by one
                for(LwU32 i=0; i<muxSignals.size(); i++)
                {
                    pCompactedMuxSignals[i] = muxSignals[i];
                }
                // terminate with a NULL signal
                memset(&pCompactedMuxSignals[muxSignals.size()], 0, sizeof(pCompactedMuxSignals[0]));

                // init a MUXedSignal table (must be empty)
                LwSignalDataBase::GPUChipletSignalTable *pChipletTable = findOrCreateGPUChipletSignalTable(ppPmSignalTable, gpuId, chipletId, domainIndex);
                if (!pChipletTable || pChipletTable->pMuxSignals)
                {
                    return false;
                }
                pChipletTable->pMuxSignals = pCompactedMuxSignals;
            }
            printf("parsed %zd muxed perfmon signals total.\n", totalSignals);

            return true;
        }

        ~MuxPerfmonDataReader()
        {
        }
};

class PerfmonDataReader : public DataReader
{
    public:
        PerfmonDataReader()
        {
        }

        bool read(LwSignalDataBase::GPUSignalTable *ppPmSignalTable, const char *pFileName, const char *pGpuOverrideName)
        {
            if (!DataReader::read(pFileName))
            {
                return false;
            }

            // we'll compact coreSignals once we know how many signals we got total
            std::vector<LwSignalDataBase::SignalName2Numeric> coreSignals;

            // TODO: quick'n'dirty parse. maybe replace with real flex/bison parser...
            size_t totalSignals = 0;
            for(char *p=m_pBuf; p<m_pEnd;)
            {
                int bytesRead;

                // extract "// perfmon_sys:0 = host0" data
                char chipletName[256] = "";
                unsigned int domainIndex = ~0;
                char domainName[256] = "";
                bytesRead = 0;
                _snscanf(p, MAX_BYTES_PER_SNSCANF, " // perfmon_%255[a-z0-9]:%i = %255[a-z0-9]%n", chipletName, &domainIndex, domainName, &bytesRead);
                if (bytesRead)
                {
                    p += bytesRead;
                }
                else
                {
                    assert(0);
                    return false;
                }

                // extract "static const LwSignalDataBase::SignalName2Numeric lwga102_sys_host0[] ="
                char gpuName[256] = "";
                char chipletName2[256] = "";
                char domainName2[256] = "";
                bytesRead = 0;
                _snscanf(p, MAX_BYTES_PER_SNSCANF, " static const LwSignalDataBase::SignalName2Numeric lw%255[a-z0-9]_%255[a-z0-9]_%255[a-z0-9][] = { %n", gpuName, chipletName2, domainName2, &bytesRead);
                if (bytesRead)
                {
                    p += bytesRead;
                }
                else
                {
                    assert(0);
                    return false;
                }
                skipWhiteSpaces(&p);

                // domain names must match
                if (strcmp(domainName, domainName2) != 0)
                {
                    assert(0);
                    return false;
                }
                if (strcmp(chipletName, chipletName2) != 0)
                {
                    assert(0);
                    return false;
                }

                // decode
                LwU32 gpuId = getGpudIdFromGpuName(gpuName, pGpuOverrideName);
                if (gpuId == ~0)
                {
                    return false;
                }
                LwU32 chipletId = getChipletIndexdFromChipletName(chipletName);
                if (chipletId == ~0)
                {
                    return false;
                }

                printf("parsing raw perfmon signals for gpuId=0x%x, chipletId=%d, domainName=%s\n", gpuId, chipletId, domainName2);

                // avoid large and slow malloc/frees
                coreSignals.clear();

                // extract list of signals
                //      {"host0_sigval_zero",                                               0},
                //      ...
                //      {"crop2pm_rop1_perf_muxed_bus_0",                                   22, {{LwSignalDataBase::PM_IT_NONE, 1, 0x410510, 31, 31, 1, 0x800, 0x0}}},
                //      {"crop2pm_rop1_perf_muxed_bus_1",                                   23, {{LwSignalDataBase::PM_IT_NONE, 1, 0x410510, 31, 31, 1, 0x800, 0x0}}},
                //      ...
                //      {NULL}
                for(;p<m_pEnd;)
                {
                    // end of perfmon table? "{NULL}"
                    bytesRead = 0;
                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " { NULL } } ; %n", &bytesRead);
                    if (bytesRead)
                    {
                        p += bytesRead;
                        break;
                    }

                    // new perfmon signal
                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " { %n", &bytesRead);
                    if (bytesRead)
                    {
                        p += bytesRead;

                        // extract data from
                        //      "host0_sigval_zero",                                               0
                        char signalName[256] = "";
                        LwU32 signalId = ~0;
                        bytesRead = 0;
                        _snscanf(p, MAX_BYTES_PER_SNSCANF, " \"%255[a-z0-9_]\" , %i %n", signalName, &signalId, &bytesRead);
                        if (bytesRead)
                        {
                            p += bytesRead;

                            // add new node
                            coreSignals.emplace_back();
                            LwSignalDataBase::SignalName2Numeric *pCoreSignal = &coreSignals.back();
                            memset(pCoreSignal, 0, sizeof(*pCoreSignal));

                            // save parsed values
                            LwU32 lwrrentIinstanceSelectCount = 0;
                            pCoreSignal->pName = _strdup(signalName);
                            pCoreSignal->numericId = signalId;

                            // got a list of instance selects?
                            //      {"crop2pm_rop1_perf_muxed_bus_0",                                   22, {{LwSignalDataBase::PM_IT_NONE, 1, 0x410510, 31, 31, 1, 0x800, 0x0}}},
                            //      {"crop2pm_rop1_perf_muxed_bus_1",                                   23, {{LwSignalDataBase::PM_IT_NONE, 1, 0x410510, 31, 31, 1, 0x800, 0x0}}},
                            bytesRead = 0;
                            _snscanf(p, MAX_BYTES_PER_SNSCANF, " , { %n", &bytesRead);
                            if (bytesRead)
                            {
                                p += bytesRead;
                                for(;;)
                                {
                                    // extract {LwSignalDataBase::PM_IT_NONE, 1, 0x410510, 31, 31, 1, 0x800, 0x0}
                                    char instanceTypeName[256] = "";
                                    LwU32 count            = ~0;
                                    LwU32 offset           = ~0;
                                    LwU32 firstBit         = ~0;
                                    LwU32 lastBit          = ~0;
                                    LwU32 val              = ~0;
                                    LwU32 offsetChipletInc = ~0;
                                    LwU32 offsetInstInc    = ~0;
                                    bytesRead = 0;
                                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " {%255[a-zA-Z0-9_:] , %i , %i , %i , %i , %i , %i , %i} %n", instanceTypeName, &count, &offset, &firstBit, &lastBit, &val, &offsetChipletInc, &offsetInstInc, &bytesRead);
                                    if (bytesRead)
                                    {
                                        p += bytesRead;
                                    }
                                    else
                                    {
                                        assert(0);
                                        return false;
                                    }

                                    LwU32 instanceType = getInstanceTypeFromInstanceTypeName(instanceTypeName);
                                    if (instanceType == ~0)
                                    {
                                        return false;
                                    }

                                    // save instance select info
                                    LwSignalDataBase::InstanceSelectInfo *pInstanceSelectInfo = &pCoreSignal->inst[lwrrentIinstanceSelectCount++];
                                    if (lwrrentIinstanceSelectCount > LwSignalDataBase::MAX_INST_SELECT_COUNT)
                                    {
                                        assert(0);
                                        return false;
                                    }
                                    pInstanceSelectInfo->instanceType     = instanceType;
                                    pInstanceSelectInfo->count            = count;
                                    pInstanceSelectInfo->offset           = offset;
                                    pInstanceSelectInfo->firstBit         = firstBit;
                                    pInstanceSelectInfo->lastBit          = lastBit;
                                    pInstanceSelectInfo->val              = val;

                                    // store just the log2 value we can get everything into 8 bytes
                                    pInstanceSelectInfo->offsetChipletInc_zeroOrlog2 = encodeZeroOrLog2(offsetChipletInc);
                                    pInstanceSelectInfo->offsetInstInc_zeroOrlog2    = encodeZeroOrLog2(offsetInstInc);

                                    // make sure the bitfield didn't drop anybits
                                    if ((pInstanceSelectInfo->instanceType                                  != instanceType)                ||
                                        (pInstanceSelectInfo->count                                         != count)                       ||
                                        (pInstanceSelectInfo->offset                                        != offset)                      || 
                                        (pInstanceSelectInfo->firstBit                                      != firstBit)                    || 
                                        (pInstanceSelectInfo->lastBit                                       != lastBit)                     ||
                                        (pInstanceSelectInfo->val                                           != val)                         || 
                                        (decodeZeroOrLog2(pInstanceSelectInfo->offsetChipletInc_zeroOrlog2) != offsetChipletInc)            || 
                                        (decodeZeroOrLog2(pInstanceSelectInfo->offsetInstInc_zeroOrlog2)    != offsetInstInc)) 
                                    {
                                        assert(0);
                                        return false;
                                    }

                                    // done? next line
                                    bytesRead = 0;
                                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " , %n", &bytesRead);
                                    if (bytesRead)
                                    {
                                        p += bytesRead;
                                        continue;
                                    }
                                    _snscanf(p, MAX_BYTES_PER_SNSCANF, " } %n", &bytesRead);
                                    if (bytesRead)
                                    {
                                        p += bytesRead;
                                        break;
                                    }
                                    else
                                    {
                                        assert(0);
                                        return false;
                                    }
                                }
                            }

                            // no instance information (or everything parsed)?
                            //      {"rop1_sigval_pmm_context_halt",                                    20},
                            bytesRead = 0;
                            _snscanf(p, MAX_BYTES_PER_SNSCANF, " } , %n", &bytesRead);
                            if (bytesRead)
                            {
                                p += bytesRead;
                            }
                            else
                            {
                                assert(0);
                                return false;
                            }
                        }
                        else
                        {
                            assert(0);
                            return false;
                        }
                        continue;
                    }
                    else
                    {
                        assert(0);
                        return false;
                    }
                }

                printf("parsed %zd core perfmon signals\n", coreSignals.size());
                totalSignals += coreSignals.size();

                // compact pCoreSignals
                LwSignalDataBase::SignalName2Numeric *pCompactedCoreSignals = new LwSignalDataBase::SignalName2Numeric[coreSignals.size()+1];
                if (!pCompactedCoreSignals)
                {
                    assert(0);
                    return false;
                }
                // copy one by one
                for(LwU32 i=0; i<coreSignals.size(); i++)
                {
                    pCompactedCoreSignals[i] = coreSignals[i];
                }
                // terminate with a NULL signal
                memset(&pCompactedCoreSignals[coreSignals.size()], 0, sizeof(pCompactedCoreSignals[0]));

                // init a SignalName2Numeric table (must be empty)
                LwSignalDataBase::GPUChipletSignalTable *pChipletTable = findOrCreateGPUChipletSignalTable(ppPmSignalTable, gpuId, chipletId, domainIndex);
                if (!pChipletTable || pChipletTable->pCoreSignals)
                {
                    return false;
                }
                pChipletTable->pCoreSignals = pCompactedCoreSignals;
            }
            printf("parsed %zd core perfmon signals total.\n", totalSignals);

            return true;
        }

        ~PerfmonDataReader()
        {
        }
};

// LZMA compressor callbacks

static void *lzmaAlloc(void *p, size_t size)
{
    p = p;
    return malloc(size);
}

static void lzmaFree(void *p, void *address)
{
    p = p;
    free(address);
}

static ISzAlloc g_lzmaAlloc = {lzmaAlloc, lzmaFree};

static void printUsage()
{
    printf("usage: GenerateSignalDataBase [--compress] [--minimizeCppFileSize] [--dataBaseSymbolName <custom g_PMMuxSignalDataBase>] --minimizeCppFileSize --inFileNames[WithOverride] [<gpuOverrideName>] <muxPmFileName> <pmFileName> --out <filename>\n");
}

int main(int argc, char* argv[])
{
    struct InputData
    {
        char *pGpuOverrideName;
        char *pMuxSignalFileName;
        char *pSignalFileName;
    };
    const LwU32 MAX_GPU_TABLE_COUNT = 16;
    InputData input[MAX_GPU_TABLE_COUNT];
    LwU32 inputCount = 0;
    char *pOutFile = NULL;
    for(int k=1; k<argc; k++)
    {
        if (_stricmp(argv[k], "--compress") == 0)
        {
            g_compress = true;
            continue;
        }
        if (_stricmp(argv[k], "--minimizeCppFileSize") == 0)
        {
            g_minimizeCppFileSize = true;
            continue;
        }
        if ((k+1 < argc) && _stricmp(argv[k], "--dataBaseSymbolName") == 0)
        {
            g_pDataBaseSymbolName = argv[++k];
            continue;
        }
        if ((k+3 < argc) && _stricmp(argv[k], "--inFileNamesWithOverride") == 0)
        {
            if (inputCount >= MAX_GPU_TABLE_COUNT)
            {
                return 0;
            }
            input[inputCount].pGpuOverrideName   = argv[++k];
            input[inputCount].pMuxSignalFileName = argv[++k];
            input[inputCount].pSignalFileName    = argv[++k];
            inputCount++;
            continue;
        }

        if ((k+2 < argc) && _stricmp(argv[k], "--inFileNames") == 0)
        {
            if (inputCount >= MAX_GPU_TABLE_COUNT)
            {
                return 0;
            }
            input[inputCount].pGpuOverrideName   = NULL;
            input[inputCount].pMuxSignalFileName = argv[++k];
            input[inputCount].pSignalFileName    = argv[++k];
            inputCount++;
            continue;
        }
        if ((k+1 < argc) && _stricmp(argv[k], "--out") == 0)
        {
            pOutFile = argv[++k];
            continue;
        }
    }

    if (!pOutFile)
    {
        printUsage();
        return 0;
    }

    // read + parse input data
    const LwSignalDataBase::GPUSignalTable *ppPmSignalTable[MAX_GPU_TABLE_COUNT];
    memset(ppPmSignalTable, 0, sizeof(ppPmSignalTable[0]) * MAX_GPU_TABLE_COUNT);
    for(LwU32 i=0; i<inputCount; i++)
    {
        // need extra terminating table entry!
        LwSignalDataBase::GPUSignalTable *pNewPmSignalTable = new LwSignalDataBase::GPUSignalTable[2];
        if (!pNewPmSignalTable)
        {
            assert(0);
            return 0;
        }
        memset(pNewPmSignalTable, 0, sizeof(*pNewPmSignalTable) * 2);

        MuxPerfmonDataReader muxPerfmonSignals;
        if (!muxPerfmonSignals.read(pNewPmSignalTable, input[i].pMuxSignalFileName, input[i].pGpuOverrideName))
        {
            assert(0);
            return 0;
        }

        PerfmonDataReader perfmonSignals;
        if (!perfmonSignals.read(pNewPmSignalTable, input[i].pSignalFileName, input[i].pGpuOverrideName))
        {
            assert(0);
            return 0;
        }

        ppPmSignalTable[i] = pNewPmSignalTable;
    }

    // create the pm database
    void *pmSignalDataBaseData = NULL;
    LwU32 pmSignalDataBaseSize = 0;
    printf("generating dense pm tables ...\n");
    if (lwCompressPerfmonTables(&pmSignalDataBaseData, &pmSignalDataBaseSize, ppPmSignalTable))
    {
        printf("generated %.3fMB of dense pm tables.\n", pmSignalDataBaseSize/(1024.0f*1024.0f));

        // specify uncompressed data set
        LwU32 orgSize = pmSignalDataBaseSize;
        LwU32 writeSize = pmSignalDataBaseSize;
        void *pWriteData = pmSignalDataBaseData;

        // compress if requested
        if (g_compress)
        {
            printf("lzma compressing ...\n");

            // allocate temp space for compressed data (we compress using "ram to ram" mode)
            SizeT inSize = orgSize;
            SizeT outSize = orgSize;
            Byte *pCompressedData = new Byte[sizeof(LwCompressedSignalDataBase::COMPRESSED_SIGNAL_DATABASE_HEADER) + LZMA_PROPS_SIZE + inSize];
            if (pCompressedData)
            {
                // clear the entire buffer to get repeatable results, i.e. in debug builds lzma doesn't write the last LwU32/0xcdcdcdcd ?!?!?
                memset(pCompressedData, 0, sizeof(LwCompressedSignalDataBase::COMPRESSED_SIGNAL_DATABASE_HEADER) + LZMA_PROPS_SIZE + inSize);

                LwCompressedSignalDataBase::COMPRESSED_SIGNAL_DATABASE_HEADER *pHDR = (LwCompressedSignalDataBase::COMPRESSED_SIGNAL_DATABASE_HEADER*) pCompressedData;
                memset(pHDR, 0, sizeof(*pHDR));
                pHDR->fourcc = LwCompressedSignalDataBase::FOURCC_LZMA;
                pHDR->orgSize = (LwU32) inSize;
                pHDR->propOffset = sizeof(*pHDR);
                pHDR->propSize = LZMA_PROPS_SIZE;
                pHDR->comprOffset = pHDR->propOffset + pHDR->propSize;

                // ilwoke lzma compressor (ram to ram)
                CLzmaEncProps props;
                LzmaEncProps_Init(&props);
                SizeT propsSize = pHDR->propSize;
                SRes lzmaRes = LzmaEncode(pCompressedData + pHDR->comprOffset, &outSize, (Byte*)pmSignalDataBaseData, inSize, &props, pCompressedData + pHDR->propOffset, &propsSize, true, NULL, &g_lzmaAlloc, &g_lzmaAlloc);
                if (lzmaRes != SZ_OK)
                {
                    printf("lzma failed with res=0x%x\n", lzmaRes);
                    return 0;
                }
                printf("lzma compressed from %.3fMB to %.3fMB\n", inSize/(1024.0f*1024.0f), outSize/(1024.0f*1024.0f));

                // fill in the compressed size
                pHDR->comprSize = (LwU32) outSize;
                pHDR->propSize  = (LwU32) propsSize;

                // specify what to write
                pWriteData = pHDR;
                writeSize = sizeof(LwCompressedSignalDataBase::COMPRESSED_SIGNAL_DATABASE_HEADER) + pHDR->propSize + pHDR->comprSize;
            }
            else
            {
                printf("failed to allocate temp space for lzma compression\n");
                return 0;
            }
        }

        printf("writing binary hex dump to \"%s\" ... (%.3fMB)\n", pOutFile, writeSize/(1024.0f*1024.0f));
        FILE *f = fopen(pOutFile, "wb");
        if (f)
        {
            fprintf(f, " /************************ BEGIN COPYRIGHT NOTICE ***************************\\\n");
            fprintf(f, "|*                                                                           *|\n");
            fprintf(f, "|* Copyright 2003-2019 by LWPU Corporation.  All rights reserved.  All     *|\n");
            fprintf(f, "|* information contained herein is proprietary and confidential to LWPU    *|\n");
            fprintf(f, "|* Corporation.  Any use, reproduction, or disclosure without the written    *|\n");
            fprintf(f, "|* permission of LWPU Corporation is prohibited.                           *|\n");
            fprintf(f, "|*                                                                           *|\n");
            fprintf(f, "\\************************** END COPYRIGHT NOTICE ***************************/\n");
            fprintf(f, "\n");

            fprintf(f, "// auto generated using //sw/dev/gpu_drv/bugfix_main/apps/cyclestats/gpu/GenerateSignalDataBase/GenerateSignalDataBase.sln \n");
            if (g_compress)
            {
                fprintf(f, "// lzma compressed (%.3fMB -> %.3fMB)\n", writeSize/(1024.0f*1024.0f), orgSize/(1024.0f*1024.0f));
            }

            for(LwU32 pmTableIndex=0; ppPmSignalTable[pmTableIndex]; pmTableIndex++)
            {
                const LwSignalDataBase::GPUSignalTable *pPMsignalTable = ppPmSignalTable[pmTableIndex];
                for(const LwSignalDataBase::GPUSignalTable *pGPU = pPMsignalTable; pGPU->gpuId; pGPU++)
                {
                    fprintf(f, "// supports gpuId=0x%08X\n", pGPU->gpuId);
                }
            }

            fprintf(f, "const LwU32 %s[] =\n{\n", g_pDataBaseSymbolName);
            for(LwU32 offset=0; offset<writeSize;)
            {
                if ((offset%0x100) == 0)
                {
                    if (g_minimizeCppFileSize)
                    {
                        fprintf(f, "    ");
                    }
                    else
                    {
                        fprintf(f, "  ");
                    }
                }
                else
                {
                    if (g_minimizeCppFileSize)
                    {
                        fprintf(f, ",");
                    }
                    else
                    {
                        fprintf(f, ", ");
                    }
                }

                LwU32 *pData = (LwU32*) ((char*)pWriteData + offset);
                LwU32 v = *pData;
                if (g_minimizeCppFileSize)
                {
                    if (v < 10)
                    {
                        fprintf(f, "%X", v);
                    }
                    else
                    {
                        fprintf(f, "0x%X", v);
                    }
                }
                else
                {
                    fprintf(f, "0x%08X", v);
                }

                offset += sizeof(LwU32);

                if (((offset%0x100) == 0) && (offset != writeSize))
                {
                    fprintf(f, ",\n");
                }
            }
            if ((writeSize%0x100) != 0)
            {
                fprintf(f, "\n");
            }
            fprintf(f, "};\n");
            fclose(f);
        }
        delete[] pmSignalDataBaseData;
    }
    else
    {
        printf("failed to create compressed database!\n");
    }
    return 0;
}
