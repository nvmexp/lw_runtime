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
#include <stdarg.h>
#include <string.h>
#include <windows.h>

#include <lw32.h>
#include <lwos.h>

#include "lwtypes.h"

#include "../../../drivers/common/cyclestats/gpu/lwPerfMonHW.h"
#include "../../../drivers/common/cyclestats/gpu/lwPerfMonCompressedTable.h"
#include "lwCompressedHeap.h"
#include "lwPerfmonTableCompress.h"

static bool compressCoreSignal(LwCompressedHeap *pHeap, LwCompressedSignalDataBase::CompressedSignalName2Numeric *pDstCoreSignal, const LwSignalDataBase::SignalName2Numeric *pSrcCoreSignal)
{
    // copy the numericId as is
    pDstCoreSignal->numericId = pSrcCoreSignal->numericId;

    // copy the signal name
    char *pDstCoreSignalName = (char*) pHeap->addRedundantString(pSrcCoreSignal->pName);
    if (pDstCoreSignalName)
    {
        pDstCoreSignal->pNameDIV4 = pHeap->getOffsetDIV4(pDstCoreSignalName);

        // how many instances?
        LwU32 instCount = 0;
        for(LwU32 i=0; i<LwSignalDataBase::MAX_INST_SELECT_COUNT; i++)
        {
            const LwSignalDataBase::InstanceSelectInfo *pSrcInst = &pSrcCoreSignal->inst[i];
            if (pSrcInst->offset)
            {
                instCount++;
            }
        }
        pDstCoreSignal->instCount = instCount;
        if (!instCount)
        {
            return true;
        }

        // copy each inst
        LwU32 dstInstTableDIV4[LwSignalDataBase::MAX_INST_SELECT_COUNT];
        memset(dstInstTableDIV4, 0, sizeof(dstInstTableDIV4));
        for(LwU32 i=0; i<instCount; i++)
        {
            const LwSignalDataBase::InstanceSelectInfo *pSrcInst = &pSrcCoreSignal->inst[i];
            assert(pSrcInst->offset);

            LwSignalDataBase::InstanceSelectInfo *dstInst = (LwSignalDataBase::InstanceSelectInfo*) pHeap->addRedundantData(pSrcInst, sizeof(*pSrcInst));
            if (dstInst)
            {
                dstInstTableDIV4[i] = pHeap->getOffsetDIV4(dstInst);
            }
            else
            {
                assert(0);
            }
        }
            
        LwU32 *pFinalDstInstTable = (LwU32*) pHeap->addRedundantData(dstInstTableDIV4, sizeof(LwU32) * instCount);
        if (pFinalDstInstTable)
        {
            pDstCoreSignal->pInstArrayDIV4 = pHeap->getOffsetDIV4(pFinalDstInstTable);
            return true;
        }
        else
        {
            assert(0);
        }
    }
    return false;
}

static int __cdecl sortByCoreName(void *context, const void *pA, const void *pB)
{
    const LwCompressedSignalDataBase::CompressedSignalName2Numeric *pS0 = (const LwCompressedSignalDataBase::CompressedSignalName2Numeric *)pA;
    const LwCompressedSignalDataBase::CompressedSignalName2Numeric *pS1 = (const LwCompressedSignalDataBase::CompressedSignalName2Numeric *)pB;
    char *s0 = (char*)context + (pS0->pNameDIV4 * 4);
    char *s1 = (char*)context + (pS1->pNameDIV4 * 4);
    return strcmp(s0, s1);
}

// helper

static bool compressMuxSetup(LwCompressedHeap *pHeap, LwCompressedSignalDataBase::CompressedMUXedSignalSetup *pDstSetup, const LwSignalDataBase::MUXedSignalSetup *pSrcSetup)
{
    // copy the signal name
    char *pDstMuxSignalSetupName = (char*) pHeap->addRedundantString(pSrcSetup->pName);
    if (pDstMuxSignalSetupName)
    {
        // how many mux are used?
        LwU32 muxCount = 0;
        for(LwU32 i=0; i<LwSignalDataBase::MAX_MUX_SELECT_COUNT; i++)
        {
            const LwSignalDataBase::MultiplexerSelectInfo *pSrcMux = &pSrcSetup->mux[i];
            if (pSrcMux->offset)
            {
                muxCount++;
            }
        }

        // allocate space for the setup pointers
        pDstSetup->muxCount.muxCount = muxCount;
        if (!muxCount)
        {
            return true;
        }

        // many signals just have a single setup. Avoids extra indirection which will be 100% unique/not compress at all
        if (muxCount == 1)
        {
            const LwSignalDataBase::MultiplexerSelectInfo *pSrcMux0 = &pSrcSetup->mux[0];
            assert(pSrcMux0->offset);
            LwSignalDataBase::MultiplexerSelectInfo *dstMux0 = (LwSignalDataBase::MultiplexerSelectInfo*) pHeap->addRedundantData(pSrcMux0, sizeof(LwSignalDataBase::MultiplexerSelectInfo));
            if (dstMux0)
            {
                pDstSetup->muxCountEQ1.muxCount = 1;
                pDstSetup->muxCountEQ1.pMuxDIV4 = pHeap->getOffsetDIV4(dstMux0);
                return true;
            }
            else
            {
                assert(0);
            }

        }
        else
        {
            // copy each mux
            LwU32 dstMuxTableDIV4[LwSignalDataBase::MAX_MUX_SELECT_COUNT];
            memset(dstMuxTableDIV4, 0, sizeof(dstMuxTableDIV4));
            for(LwU32 i=0; i<muxCount; i++)
            {
                const LwSignalDataBase::MultiplexerSelectInfo *pSrcMux = &pSrcSetup->mux[i];
                assert(pSrcMux->offset);

                LwSignalDataBase::MultiplexerSelectInfo *dstMux = (LwSignalDataBase::MultiplexerSelectInfo*) pHeap->addRedundantData(pSrcMux, sizeof(LwSignalDataBase::MultiplexerSelectInfo));
                if (dstMux)
                {
                    dstMuxTableDIV4[i] = pHeap->getOffsetDIV4(dstMux);
                }
                else
                {
                    assert(0);
                }
            }
            
            LwU32 *pFinalDstMuxTable = (LwU32*) pHeap->addRedundantData(dstMuxTableDIV4, sizeof(LwU32) * muxCount);
            if (pFinalDstMuxTable)
            {
                pDstSetup->muxCountGE2.muxCount      = muxCount;
                pDstSetup->muxCountGE2.pMuxArrayDIV4 = pHeap->getOffsetDIV4(pFinalDstMuxTable);
                return true;
            }
            else
            {
                assert(0);
            }
        }
    }
    else
    {
        assert(0);
    }
    return false;
}

static LwU32 findCoreSignalDIV4(LwCompressedHeap *pHeap, LwCompressedSignalDataBase::CompressedSignalName2Numeric *pCompressedCoreTable, LwU32 coreSignalCount, const char *pCoreName)
{
    LwS32 l = 0;
    LwS32 h = coreSignalCount-1;
    for(;l<=h;)
    {
        // check the new middle
        LwU32 m = (l+h)/2;
        LwCompressedSignalDataBase::CompressedSignalName2Numeric *pSrcInst = &pCompressedCoreTable[m];
        char *pName = (char*) ((char*)pHeap->getBase() + (pSrcInst->pNameDIV4 * 4));
        LwS32 compResult = strcmp(pCoreName, pName);
        if (compResult == 0)
        {
            // found name!
            return pHeap->getOffsetDIV4(pSrcInst);
        }
        else if (compResult < 0)
        {
            h = m-1;
        }
        else
        {
            assert(compResult > 0);
            l = m+1;
        }
    }

    // not found
    assert(0);
    return 0;
}

static bool compressMuxSignal(LwCompressedHeap *pHeap, LwCompressedSignalDataBase::CompressedMUXedSignal *pDstMuxSignal, const LwSignalDataBase::MUXedSignal *pSrcMuxSignal, LwCompressedSignalDataBase::CompressedSignalName2Numeric *pCompressedCoreTable, LwU32 coreSignalCount)
{
    // copy the signal name
    void *pDstMuxSignalNamePart0 = NULL;
    void *pDstMuxSignalNamePart1 = NULL;
    LwU32 nameLenPart0 = 0;
    LwU32 nameLenPart1 = 0;
    if (pHeap->addRedundantTwoPartString(&pDstMuxSignalNamePart0, &nameLenPart0, &pDstMuxSignalNamePart1, &nameLenPart1, pSrcMuxSignal->name))
    {
        pDstMuxSignal->name.pNamePart0DIV4 = pHeap->getOffsetDIV4(pDstMuxSignalNamePart0);
        pDstMuxSignal->name.nameLen0       = nameLenPart0;
        if (nameLenPart1)
        {
            pDstMuxSignal->name.pNamePart1DIV4 = pHeap->getOffsetDIV4(pDstMuxSignalNamePart1);
            pDstMuxSignal->name.nameLen1       = nameLenPart1;
        }

        // how many setups?
        LwU32 setupCount = 0;
        for(LwU32 i=0; i<LwSignalDataBase::MAX_SETUP_COUNT; i++)
        {
            const LwSignalDataBase::MUXedSignalSetup *pSrcSetup = &pSrcMuxSignal->setup[i];
            if (pSrcSetup->pName)
            {
                setupCount++;
            }
        }

        // compress each setup
        LwCompressedSignalDataBase::CompressedMUXedSignalSetup dstSetup[LwSignalDataBase::MAX_SETUP_COUNT];
        memset(dstSetup, 0, sizeof(dstSetup));
        for(LwU32 i=0; i<setupCount; i++)
        {
            const LwSignalDataBase::MUXedSignalSetup *pSrcSetup = &pSrcMuxSignal->setup[i];
            assert(pSrcSetup->pName);
            if (!compressMuxSetup(pHeap, &dstSetup[i], pSrcSetup))
            {
                assert(0);
            }

            // compute direct link to core signals
            if (coreSignalCount)
            {
                dstSetup[i].pSignalName2NumericDIV4 = findCoreSignalDIV4(pHeap, pCompressedCoreTable, coreSignalCount, pSrcSetup->pName);
            }
        }

        // many signals just have a single setup. Avoids extra indirection which will be 100% unique/not compress at all
        assert(setupCount>0);
        if (setupCount == 1)
        {
            LwCompressedSignalDataBase::CompressedMUXedSignalSetup *finalDstSetup = (LwCompressedSignalDataBase::CompressedMUXedSignalSetup*) pHeap->addRedundantData(&dstSetup[0], sizeof(LwCompressedSignalDataBase::CompressedMUXedSignalSetup));
            if (finalDstSetup)
            {
                pDstMuxSignal->setupCountEQ1.setupCount = 1;
                pDstMuxSignal->setupCountEQ1.pSetupDIV4 = pHeap->getOffsetDIV4(finalDstSetup);
                return true;
            }
            else
            {
                assert(0);
            }
        }
        else
        {
            // copy each setup
            LwU32 dstSetupTableDIV4[LwSignalDataBase::MAX_SETUP_COUNT];
            memset(dstSetupTableDIV4, 0, sizeof(dstSetupTableDIV4));
            for(LwU32 i=0; i<setupCount; i++)
            {
                LwCompressedSignalDataBase::CompressedMUXedSignalSetup *finalDstSetup = (LwCompressedSignalDataBase::CompressedMUXedSignalSetup*) pHeap->addRedundantData(&dstSetup[i], sizeof(LwCompressedSignalDataBase::CompressedMUXedSignalSetup));
                if (finalDstSetup)
                {
                    dstSetupTableDIV4[i] = pHeap->getOffsetDIV4(finalDstSetup);
                }
                else
                {
                    assert(0);
                }
            }

            LwU32 *pFinalDstSetupTable = (LwU32*) pHeap->addRedundantData(dstSetupTableDIV4, sizeof(LwU32) * setupCount);
            if (pFinalDstSetupTable)
            {
                pDstMuxSignal->setupCountGE2.setupCount      = setupCount;
                pDstMuxSignal->setupCountGE2.pSetupArrayDIV4 = pHeap->getOffsetDIV4(pFinalDstSetupTable);
                return true;
            }
            else
            {
                assert(0);
            }
        }
    }
    else
    {
        assert(0);
    }

    return false;
}

// str[i]cmp() equivalent for "two part strings"

static int strcmp(void *pBase, const LwCompressedSignalDataBase::CompressedTwoPartName *pName0, const LwCompressedSignalDataBase::CompressedTwoPartName *pName1, bool ignoreCase)
{
    char *pA0   = (char*)pBase + (pName0->pNamePart0DIV4 * 4);
    char *pA1   = (char*)pBase + (pName0->pNamePart1DIV4 * 4);
    LwU32 lenA0 = pName0->nameLen0;
    LwU32 lenA1 = pName0->nameLen1;

    char *pB0   = (char*)pBase + (pName1->pNamePart0DIV4 * 4);
    char *pB1   = (char*)pBase + (pName1->pNamePart1DIV4 * 4);
    LwU32 lenB0 = pName1->nameLen0;
    LwU32 lenB1 = pName1->nameLen1;

    // find first mismatching character
    char a;
    char b;
    for(;;)
    {
        // read next char
        if (lenA0)
        {
           a = *pA0++;
           lenA0--;
        }
        else if (lenA1)
        {
           a = *pA1++;
           lenA1--;
        }
        else
        {
            a = 0;
        }

        if (lenB0)
        {
           b = *pB0++;
           lenB0--;
        }
        else if (lenB1)
        {
           b = *pB1++;
           lenB1--;
        }
        else
        {
            b = 0;
        }

        if (ignoreCase)
        {
            a = tolower(a);
            b = tolower(b);
        }

        // match (and didn't reach end)?
        if ((a == b) && a)
        {
            continue;
        }
        else
        {
            // compare first mismatching char
            if (a < b)
            {
                return -1;
            }
            else if (a > b)
            {
                return 1;
            }
            else
            {
                assert(!lenA0 && !lenA1 && !lenB0 && !lenB1);
                return 0;
            }
        }
    }
}

static int __cdecl sortByMuxName(void *context, const void *pA, const void *pB)
{
    const LwCompressedSignalDataBase::CompressedMUXedSignal *pS0 = (const LwCompressedSignalDataBase::CompressedMUXedSignal *)pA;
    const LwCompressedSignalDataBase::CompressedMUXedSignal *pS1 = (const LwCompressedSignalDataBase::CompressedMUXedSignal *)pB;
    return strcmp(context, &pS0->name, &pS1->name, false);
}

static bool compressPerfmonTables(void **pPMSignalDataBaseData, LwU32 *pPMSignalDataBaseSize, const LwSignalDataBase::GPUSignalTable **ppPmSignalTable)
{
    bool succeeded = false;

    // how many gpus do we have in our list?
    LwU32 gpuCount = 0;
    for(LwU32 pmTableIndex=0; ppPmSignalTable[pmTableIndex]; pmTableIndex++)
    {
        const LwSignalDataBase::GPUSignalTable *pGPUSignalTable = ppPmSignalTable[pmTableIndex];
        for(const LwSignalDataBase::GPUSignalTable *pGPU = pGPUSignalTable; pGPU->gpuId; pGPU++)
        {
            gpuCount++;
        }
    }

    // create inital block with one  (all "offsets" need to fit into 24bit -- DIV4 encoding adds two extra bits -- 64MB total)
    LwCompressedHeap *pHeap = LwCompressedHeap::create((1<<24) * 4, false);
    if (pHeap)
    {
        assert(LwCompressedSignalDataBase::MAX_CHIPLET_TYPE_COUNT == LwSignalDataBase::MAX_CHIPLET_TYPE_COUNT);
        LwCompressedSignalDataBase::CompressedGPUSignalTable *pDstGpuTable = (LwCompressedSignalDataBase::CompressedGPUSignalTable*) pHeap->addUniqueData(NULL, sizeof(LwCompressedSignalDataBase::CompressedGPUSignalTable) * (gpuCount + 1));
        if (pDstGpuTable)
        {
            for(LwU32 pmTableIndex=0; ppPmSignalTable[pmTableIndex]; pmTableIndex++)
            {
                const LwSignalDataBase::GPUSignalTable *pGPUSignalTable = ppPmSignalTable[pmTableIndex];
                for(const LwSignalDataBase::GPUSignalTable *pSrcGpuTable = pGPUSignalTable; pSrcGpuTable->gpuId; pDstGpuTable++, pSrcGpuTable++)
                {
                    printf("processing gpu 0x%X\n", pSrcGpuTable->gpuId);

                    // copy data
                    pDstGpuTable->gpuId = pSrcGpuTable->gpuId;

                    // copy over chiplet tables
                    for(LwU32 i=0; i<LwSignalDataBase::MAX_CHIPLET_TYPE_COUNT; i++)
                    {
                        const LwSignalDataBase::GPUChipletSignalTable *srcChipletTable = &pSrcGpuTable->chipletTable[i];
                        if (srcChipletTable->pCoreSignals || srcChipletTable->pMuxSignals)
                        {
                            LwCompressedSignalDataBase::CompressedGPUChipletSignalTable *dstChipletTable = (LwCompressedSignalDataBase::CompressedGPUChipletSignalTable*) pHeap->addUniqueData(NULL, sizeof(LwCompressedSignalDataBase::CompressedGPUChipletSignalTable));
                            if (dstChipletTable)
                            {
                                dstChipletTable->chipletId = srcChipletTable->chipletId;
                                dstChipletTable->domainId = srcChipletTable->domainId;
                                pDstGpuTable->chipletTableOffsetDIV4[i] = pHeap->getOffsetDIV4(dstChipletTable);

                                // how many mux signals do we have?
                                LwU32 coreSignalCount = 0;
                                if (srcChipletTable->pCoreSignals)
                                {
                                    for(LwU32 j=0; srcChipletTable->pCoreSignals[j].pName; j++)
                                    {
                                        coreSignalCount++;
                                    }
                                }

                                // one extra for the null end
                                LwCompressedSignalDataBase::CompressedSignalName2Numeric *pCoreTable = new LwCompressedSignalDataBase::CompressedSignalName2Numeric[coreSignalCount+1];
                                if (pCoreTable)
                                {
                                    memset(pCoreTable, 0, sizeof(pCoreTable[0]) * (coreSignalCount+1));

                                    // copy each core signal
                                    for(LwU32 k=0; k<coreSignalCount; k++)
                                    {
                                        const LwSignalDataBase::SignalName2Numeric *pSrcCoreSignal = &srcChipletTable->pCoreSignals[k];
                                        LwCompressedSignalDataBase::CompressedSignalName2Numeric *pDstCoreSignal = &pCoreTable[k];
                                        if (!compressCoreSignal(pHeap, pDstCoreSignal, pSrcCoreSignal))
                                        {
                                            assert(0);
                                        }
                                    }

                                    // sort all core signals
                                    qsort_s(pCoreTable, coreSignalCount, sizeof(LwCompressedSignalDataBase::CompressedSignalName2Numeric), sortByCoreName, pHeap->getBase());

                                    // verify that all signals are properly sorted (because we do a binary search depends on it)
                                    for(LwU32 k=1; k<coreSignalCount; k++)
                                    {
                                        LwCompressedSignalDataBase::CompressedSignalName2Numeric *pS0 = &pCoreTable[k-1];
                                        LwCompressedSignalDataBase::CompressedSignalName2Numeric *pS1 = &pCoreTable[k];
                                        char *s0 = (char*)pHeap->getBase() + (pS0->pNameDIV4 * 4);
                                        char *s1 = (char*)pHeap->getBase() + (pS1->pNameDIV4 * 4);
                                        if (strcmp(s0, s1) > 0)
                                        {
                                            assert(0);
                                        }
                                        if (_stricmp(s0, s1) > 0)
                                        {
                                            assert(0);
                                        }
                                    }

                                    LwCompressedSignalDataBase::CompressedSignalName2Numeric *pCompressedCoreTable = (LwCompressedSignalDataBase::CompressedSignalName2Numeric*) pHeap->addRedundantData(pCoreTable, sizeof(pCoreTable[0]) * (coreSignalCount+1));

                                    delete[] pCoreTable;
                                    pCoreTable = NULL;

                                    if (pCompressedCoreTable)
                                    {
                                        dstChipletTable->pCoreSignalsOffsetDIV4 = pHeap->getOffsetDIV4(pCompressedCoreTable);
                                        dstChipletTable->coreSignalCount        = coreSignalCount;

                                        // how many mux signals do we have?
                                        LwU32 muxSignalCount = 0;
                                        if (srcChipletTable->pMuxSignals)
                                        {
                                            for(LwU32 j=0; srcChipletTable->pMuxSignals[j].name[0]; j++)
                                            {
                                                muxSignalCount++;
                                            }
                                        }

                                        // one extra for the null end
                                        LwCompressedSignalDataBase::CompressedMUXedSignal *pMuxTable = new LwCompressedSignalDataBase::CompressedMUXedSignal[muxSignalCount+1];
                                        if (pMuxTable)
                                        {
                                            memset(pMuxTable, 0, sizeof(pMuxTable[0]) * (muxSignalCount+1));

                                            // copy each Mux signal
                                            for(LwU32 k=0; k<muxSignalCount; k++)
                                            {
                                                const LwSignalDataBase::MUXedSignal *pSrcMuxSignal = &srcChipletTable->pMuxSignals[k];
                                                LwCompressedSignalDataBase::CompressedMUXedSignal *pDstMuxSignal = &pMuxTable[k];
                                                if (!compressMuxSignal(pHeap, pDstMuxSignal, pSrcMuxSignal, pCompressedCoreTable, coreSignalCount))
                                                {
                                                    assert(0);
                                                }
                                            }

                                            // sort all mux signals
                                            qsort_s(pMuxTable, muxSignalCount, sizeof(LwCompressedSignalDataBase::CompressedMUXedSignal), sortByMuxName, pHeap->getBase());

                                            // verify that all signals are properly sorted (because we do a binary search depends on it)
                                            for(LwU32 k=1; k<muxSignalCount; k++)
                                            {
                                                LwCompressedSignalDataBase::CompressedMUXedSignal *pS0 = &pMuxTable[k-1];
                                                LwCompressedSignalDataBase::CompressedMUXedSignal *pS1 = &pMuxTable[k];
                                                if (strcmp(pHeap->getBase(), &pS0->name, &pS1->name, false) > 0)
                                                {
                                                    assert(0);
                                                }
                                                if (strcmp(pHeap->getBase(), &pS0->name, &pS1->name, true) > 0)
                                                {
                                                    assert(0);
                                                }
                                            }

                                            LwCompressedSignalDataBase::CompressedMUXedSignal *pCompressedMuxTable = (LwCompressedSignalDataBase::CompressedMUXedSignal*) pHeap->addRedundantData(pMuxTable, sizeof(pMuxTable[0]) * (muxSignalCount+1));

                                            delete[] pMuxTable;
                                            pMuxTable = NULL;

                                            if (pCompressedMuxTable)
                                            {
                                                dstChipletTable->pMuxSignalsOffsetDIV4 = pHeap->getOffsetDIV4(pCompressedMuxTable);
                                                dstChipletTable->muxSignalCount        = muxSignalCount;
                                            }
                                            else
                                            {
                                                assert(0);
                                            }
                                        }
                                        else
                                        {
                                            assert(0);
                                        }
                                    }
                                    else
                                    {
                                        assert(0);
                                    }
                                }
                                else
                                {
                                    assert(0);
                                }
                            }
                            else
                            {
                                assert(0);
                            }
                        }
                    }
                }
            }

            // all copied.. return as one big allocation
            void *pResult = (LwCompressedSignalDataBase::CompressedGPUSignalTable*) new char[pHeap->getTotalSize()];
            if (pResult)
            {
                memcpy(pResult, pHeap->getBase(), pHeap->getTotalSize());
                *pPMSignalDataBaseData = pResult;
                *pPMSignalDataBaseSize = pHeap->getTotalSize();
                succeeded = true;
            }
            else
            {
                assert(0);
            }
        }
        else
        {
            assert(0);
        }
        pHeap->destroy();
    }
    else
    {
        assert(0);
    }

    return succeeded;
}


// helper function

bool lwCompressPerfmonTables(void **pPMSignalDataBaseData, LwU32 *pPMSignalDataBaseSize, const LwSignalDataBase::GPUSignalTable **ppPmSignalTable)
{
    *pPMSignalDataBaseData = NULL;
    *pPMSignalDataBaseSize = 0;
    if (compressPerfmonTables(pPMSignalDataBaseData, pPMSignalDataBaseSize, ppPmSignalTable))
    {
        return true;
    }
    else
    {
        assert(0);
        return false;
    }
}
