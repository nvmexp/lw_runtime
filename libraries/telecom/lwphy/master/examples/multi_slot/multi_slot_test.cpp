/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */
#include <sched.h>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "lwda_profiler_api.h"
#include "lwphy.h"
#include "lwphy.hpp"
#include "lwphy_hdf5.hpp"
#include "util.hpp"

#include "multi_slot_test.hpp"
#include "hdf5hpp.hpp"

#define OUTPUT_TB_FNAME ("outputBits")
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

MultiSlotTest::MultiSlotTest(std::string const& name, uint32_t nInstances, lwdaStream_t& delayLwStrm, std::vector<int>& lwStrmPrios, bool& startSyncPt, std::mutex& cvStartSyncPtMutex, std::condition_variable& cvStartSyncPt, std::atomic<std::uint32_t>& atmSyncPtWaitCnt) :
    m_nInstances(nInstances),
    m_name(name),
    m_puschRxList(nInstances),
    m_lwStrms(nInstances, 0),
    m_wrkrThrds(nInstances),
    m_wrkrThrdMutexes(nInstances),
    m_startSyncPt(startSyncPt),
    m_cvStartSyncPtMutex(cvStartSyncPtMutex),
    m_cvStartSyncPt(cvStartSyncPt),
    m_atmSyncPtWaitCnt(atmSyncPtWaitCnt),
    m_eStartProcRecorded(false),
    m_totalBits(nInstances, 0),
    m_enableLwProf(false),
    m_nIterations(0),
    m_descramblingOn(true),
    m_delayMs(0),
    m_sleepDurationUs(0),
    m_delayLwStrm(delayLwStrm),
    m_eStartProcs(nInstances),
    m_eStopProcs(nInstances),
    m_atmEndProcCnt(0),
    m_startTimePts(nInstances),
    m_stopTimePts(nInstances),
    // m_dbgStartTimePt,
    m_dbgTimePts0(nInstances),
    m_dbgTimePts1(nInstances),
    m_dbgTimePts2(nInstances),
    m_dbgTimePts3(nInstances),
    m_perSlotCSum(nInstances),
    m_perSlotnTb(nInstances),
    m_perSlotTBSize(nInstances),
    m_perSlotCBcrcs(nInstances),
    m_perSlotTBcrcs(nInstances),
    m_perSlotOutputBytes(nInstances),
    m_lwrrentTbsPrmsArrays(nInstances),
    m_BBUPrmsArray(nInstances),
    m_slotDataBuf(nInstances),
    m_debugFiles(nInstances),
    m_nSlots(nInstances)
{
    // LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProc, lwdaEventDisableTiming));
    LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProc, lwdaEventBlockingSync));
    LWDA_CHECK(lwdaEventCreateWithFlags(&m_eEndProc, lwdaEventBlockingSync));

    for(uint32_t instIdx = 0; instIdx < nInstances; ++instIdx)
    {
        m_nSlots[instIdx] = 0;
        m_debugFiles[instIdx].resize(MAX_N_SLOTS);
        m_perSlotCSum[instIdx].resize(MAX_N_SLOTS);
        m_perSlotnTb[instIdx].resize(MAX_N_SLOTS);
        m_perSlotTBSize[instIdx].resize(MAX_N_SLOTS);
        m_perSlotCBcrcs[instIdx].resize(MAX_N_SLOTS);
        m_perSlotTBcrcs[instIdx].resize(MAX_N_SLOTS);
        m_perSlotOutputBytes[instIdx].resize(MAX_N_SLOTS);
        m_lwrrentTbsPrmsArrays[instIdx].resize(MAX_N_SLOTS);
        m_BBUPrmsArray[instIdx].resize(MAX_N_SLOTS);
        LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProcs[instIdx], lwdaEventBlockingSync));
        LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStopProcs[instIdx], lwdaEventBlockingSync));

        // LWDA_CHECK(lwdaStreamCreateWithFlags(&lwStrm, lwdaStreamNonBlocking));
        LWDA_CHECK(lwdaStreamCreateWithPriority(&m_lwStrms[instIdx], lwdaStreamNonBlocking, lwStrmPrios[instIdx]));
    }
}

MultiSlotTest::~MultiSlotTest()
{
    LWDA_CHECK(lwdaEventDestroy(m_eStartProc));
    LWDA_CHECK(lwdaEventDestroy(m_eEndProc));

    for(uint32_t instIdx = 0; instIdx < m_puschRxList.size(); ++instIdx)
    {
        LWDA_CHECK(lwdaEventDestroy(m_eStartProcs[instIdx]));
        LWDA_CHECK(lwdaEventDestroy(m_eStopProcs[instIdx]));

        LWDA_CHECK(lwdaStreamDestroy(m_lwStrms[instIdx]));
    }
}

void MultiSlotTest::Setup(uint32_t                nIterations,
                          bool                    enableLwProf,
                          int                     descramblingOn,
                          uint32_t                delayMs,
                          uint32_t                sleepDurationUs,
                          std::vector<int> const& cpuIds,
                          std::vector<int> const& thrdSchdPolicies,
                          std::vector<int> const& thrdPrios,
                          std::vector<int> const& gpuIds,
                          const std::string&      inputFilename,
                          std::string&            outputfilename,
                          uint32_t                fp16Mode)
{
    m_nIterations     = nIterations;
    m_enableLwProf    = enableLwProf;
    m_descramblingOn  = descramblingOn;
    m_delayMs         = delayMs;
    m_sleepDurationUs = sleepDurationUs;

    m_wrkrThrdSchdPolicies = thrdSchdPolicies;
    m_wrkrThrdPrios        = thrdPrios;

    m_cpuIds = cpuIds;
    m_gpuIds = gpuIds;

    m_eStartProcRecorded = false;
    m_atmEndProcCnt      = 0;

    for(auto& totalBits : m_totalBits) { totalBits = 0; }

    //------------------------------------------------------------------
    // Open the input file and required datasets
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

    //-------------------------------------------------------------
    // Check for configuration information in the input file. Newer
    // input files will have configuration values in the file, so
    // that they don't need to be specified on the command line.
    lwphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

    uint32_t slotNumber = 0;
    uint32_t slotType   = 0;

    //-------------------------------------------------------------
    // Check FP16 mode of operation
    bool isDataRxFp16  = true;
    bool isChannelFp16 = false;
    switch(fp16Mode)
    {
    case 0:
        isDataRxFp16  = false;
        isChannelFp16 = false;
        break;
    case 1:
        isDataRxFp16  = true;
        isChannelFp16 = false;
        break;
    case 2:
        isDataRxFp16  = true;
        isChannelFp16 = true;
        break;
    default:
        isDataRxFp16  = false;
        isChannelFp16 = false;
        break;
    }
    lwphyDataType_t typeDataRx     = isDataRxFp16 ? LWPHY_R_16F : LWPHY_R_32F;
    lwphyDataType_t cplxTypeDataRx = isDataRxFp16 ? LWPHY_C_16F : LWPHY_C_32F;

    lwphyDataType_t typeFeChannel     = isChannelFp16 ? LWPHY_R_16F : LWPHY_R_32F;
    lwphyDataType_t cplxTypeFeChannel = isChannelFp16 ? LWPHY_C_16F : LWPHY_C_32F;

    for(int i = 0; i < m_nInstances; i++)
        m_nSlots[i] = 0;

    for(int i = 0; i < MAX_N_SLOTS; i++)
    {
        try
        {
            std::string             slotPostfix = (i > 0) ? "_" + std::to_string(i) : "";
            lwphy::lwphyHDF5_struct gnbConfig   = lwphy::get_HDF5_struct(fInput, std::string("gnb_pars" + slotPostfix).c_str());
            try
            {                                                            //FIXME: keep compatibility with test vectors with no slotType
                slotType = gnbConfig.get_value_as<uint32_t>("slotType"); // 0 UL, 1 DL
            }
            catch(const std::exception& exc)
            {
                slotType = 0; //assume default is uplink for now
            }
            slotNumber                                       = gnbConfig.get_value_as<uint32_t>("slotNumber");
            m_BBUPrmsArray[slotType][i].fc                   = gnbConfig.get_value_as<uint32_t>("fc");
            m_BBUPrmsArray[slotType][i].mu                   = gnbConfig.get_value_as<uint32_t>("mu");
            m_BBUPrmsArray[slotType][i].nRx                  = gnbConfig.get_value_as<uint32_t>("nRx");
            m_BBUPrmsArray[slotType][i].nPrb                 = gnbConfig.get_value_as<uint32_t>("nPrb");
            m_BBUPrmsArray[slotType][i].cellId               = gnbConfig.get_value_as<uint32_t>("cellId");
            m_BBUPrmsArray[slotType][i].slotNumber           = gnbConfig.get_value_as<uint32_t>("slotNumber");
            m_BBUPrmsArray[slotType][i].Nf                   = gnbConfig.get_value_as<uint32_t>("Nf");
            m_BBUPrmsArray[slotType][i].Nt                   = gnbConfig.get_value_as<uint32_t>("Nt");
            m_BBUPrmsArray[slotType][i].df                   = gnbConfig.get_value_as<uint32_t>("df");
            m_BBUPrmsArray[slotType][i].dt                   = gnbConfig.get_value_as<uint32_t>("dt");
            m_BBUPrmsArray[slotType][i].numBsAnt             = gnbConfig.get_value_as<uint32_t>("numBsAnt");
            m_BBUPrmsArray[slotType][i].numBbuLayers         = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
            m_BBUPrmsArray[slotType][i].numTb                = gnbConfig.get_value_as<uint32_t>("numTb");
            m_BBUPrmsArray[slotType][i].ldpcnIterations      = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
            m_BBUPrmsArray[slotType][i].ldpcEarlyTermination = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
            m_BBUPrmsArray[slotType][i].ldpcAlgoIndex        = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
            m_BBUPrmsArray[slotType][i].ldpcFlags            = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
            m_BBUPrmsArray[slotType][i].ldplwseHalf          = gnbConfig.get_value_as<uint32_t>("ldplwseHalf");

            m_lwrrentTbsPrmsArrays[slotType][i].resize(m_BBUPrmsArray[slotType][i].numTb);

            // parse array of tb_pars structs

            hdf5hpp::hdf5_dataset tbpDset = fInput.open_dataset(std::string("tb_pars" + slotPostfix).c_str());

            m_perSlotnTb[slotType][i] = m_BBUPrmsArray[slotType][i].numTb;

            for(int j = 0; j < m_BBUPrmsArray[slotType][i].numTb; j++)
            {
                lwphy::lwphyHDF5_struct tbConfig                        = lwphy::get_HDF5_struct_index(tbpDset, j);
                m_lwrrentTbsPrmsArrays[slotType][i][j].numLayers        = tbConfig.get_value_as<uint32_t>("numLayers");
                m_lwrrentTbsPrmsArrays[slotType][i][j].layerMap         = tbConfig.get_value_as<uint32_t>("layerMap");
                m_lwrrentTbsPrmsArrays[slotType][i][j].startPrb         = tbConfig.get_value_as<uint32_t>("startPrb");
                m_lwrrentTbsPrmsArrays[slotType][i][j].numPrb           = tbConfig.get_value_as<uint32_t>("numPRb");
                m_lwrrentTbsPrmsArrays[slotType][i][j].startSym         = tbConfig.get_value_as<uint32_t>("startSym");
                m_lwrrentTbsPrmsArrays[slotType][i][j].numSym           = tbConfig.get_value_as<uint32_t>("numSym");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dataScramId      = tbConfig.get_value_as<uint32_t>("dataScramId");
                m_lwrrentTbsPrmsArrays[slotType][i][j].mcsTableIndex    = tbConfig.get_value_as<uint32_t>("mcsTableIndex");
                m_lwrrentTbsPrmsArrays[slotType][i][j].mcsIndex         = tbConfig.get_value_as<uint32_t>("mcsIndex");
                m_lwrrentTbsPrmsArrays[slotType][i][j].rv               = tbConfig.get_value_as<uint32_t>("rv");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsType         = tbConfig.get_value_as<uint32_t>("dmrsType");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsAddlPosition = tbConfig.get_value_as<uint32_t>("dmrsAddlPosition");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsScramId      = tbConfig.get_value_as<uint32_t>("dmrsScramId");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsEnergy       = tbConfig.get_value_as<uint32_t>("dmrsEnergy");
                m_lwrrentTbsPrmsArrays[slotType][i][j].nRnti            = tbConfig.get_value_as<uint32_t>("nRnti");
                m_lwrrentTbsPrmsArrays[slotType][i][j].dmrsCfg          = tbConfig.get_value_as<uint32_t>("dmrsCfg");
            }

            // END NEW PARAM STRUCT PARSING
            m_slotDataBuf[slotType].emplace_back(PuschRxDataset(fInput, slotNumber, slotPostfix, cplxTypeDataRx));
            ++m_nSlots[slotType];

            if(!outputfilename.empty())
            {
                m_debugFiles[slotType][i].reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(std::string(outputfilename + slotPostfix).c_str())));
            }
        }
        catch(const std::exception& exc)
        {
            if(i == 0) // FIXME: define and implement proper exception handling
            {
                printf("%s\n", exc.what());
                throw exc;
            }
            else
                break;
            // Continue using command line arguments if the input file does not
            // have a config struct.
        }
    }
    lwphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

    // All transfers in PuschRxDataset (specifically tensor_from_dataset calls) are done via stream 0
    LWDA_CHECK(lwdaStreamSynchronize(0));

    SpawnPipelineWrkrThrds();
}

void MultiSlotTest::SpawnPipelineWrkrThrds()
{
    // Launch CPU worker threads which will wait for run condition
    uint32_t instIdx = 0;
    for(auto& wrkrThrd : m_wrkrThrds)
    {
        wrkrThrd = std::thread(&MultiSlotTest::PipelineWrkrEntry,
                               this,
                               instIdx);
        instIdx++;
    }
}

void MultiSlotTest::PipelineWrkrEntry(uint32_t instIdx)
{
    int cpuId  = m_cpuIds[instIdx];
    int gpuId  = m_gpuIds[instIdx];
    int policy = m_wrkrThrdSchdPolicies[instIdx]; // SCHED_FIFO // SCHED_RR
    int prio   = m_wrkrThrdPrios[instIdx];

    //------------------------------------------------------------------
    // Bump up prio
    pid_t       pid = (pid_t)syscall(SYS_gettid);
    sched_param schdPrm;
    schdPrm.sched_priority = prio;

    // pid_t pid = (pid_t) syscall(SYS_gettid);
    int schdSetRet = sched_setscheduler(pid, policy, &schdPrm);
    if(0 == schdSetRet)
    {
        printf("%s Pipeline[%d]: pid %d policy %d prio %d\n", m_name.c_str(), instIdx, pid, policy, prio);
    }
    else
    {
        printf("%s Pipeline[%d]: Failed to set scheduling algo pid %d, prio %d, return code %d: err %s\n",
               m_name.c_str(),
               instIdx,
               pid,
               prio,
               schdSetRet,
               strerror(errno));
    }

    //------------------------------------------------------------------
    // Set thread affinity to specified CPU
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    CPU_SET(cpuId, &cpuSet);
    // printf("%s Pipeline[%d]: setting affinity of pipeline %d (pid %d) to CPU Id %d\n", m_name.c_str(), instIdx, pid, cpuId);

    int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);
    if(0 == affinitySetRet)
    {
        printf("%s Pipeline[%d]: pid %d set affinity to CPU %d\n", m_name.c_str(), instIdx, pid, cpuId);
    }
    else
    {
        printf("%s Pipeline[%d]: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
               m_name.c_str(),
               instIdx,
               pid,
               cpuId,
               affinitySetRet,
               strerror(errno));
    }

    lwdaSetDevice(gpuId);

    PuschRx&      puschRx = m_puschRxList[instIdx];
    lwdaStream_t& lwStrm  = m_lwStrms[instIdx];
    lwdaEvent_t&  eStart  = m_eStartProcs[instIdx];
    lwdaEvent_t&  eStop   = m_eStopProcs[instIdx];

    //------------------------------------------------------------------
    // Wait for syncpoint
    {
        printf("%s Pipeline[%d]: Wait for start-sync point\n", m_name.c_str(), instIdx);
        m_atmSyncPtWaitCnt++;
        std::unique_lock<std::mutex> cvStartSyncPtMutexLock(m_cvStartSyncPtMutex);
        m_cvStartSyncPt.wait(cvStartSyncPtMutexLock, [this] { return m_startSyncPt; });
    }
    printf("%s Pipeline[%d]: start-syncpoint hit\n", m_name.c_str(), instIdx);

    //------------------------------------------------------------------
    // First pipeline instance special handling to synchronize multiple pipelines
    if(0 == instIdx)
    {
        // Launch a delay kernel to keep GPU busy until the CPU bursts out kernel launches (for all pipelines)
        if(m_delayMs)
        {
            gpu_ms_delay(m_delayMs, gpuId, lwStrm);
            m_delayLwStrm = lwStrm;
        }

        // Place a start event on the first stream which all other streams will wait on. This is used to
        // model conm_lwrrent workload submission from all streams (i.e. pipelines) as close to each other
        // in time as possible
        LWDA_CHECK(lwdaEventRecord(m_eStartProc, m_delayLwStrm));

        // Notify all other worker threads that m_eStartProc is recorded
        {
            std::lock_guard<std::mutex> cvStartProcRecMutexGuard(m_cvStartProcRecMutex);
            m_eStartProcRecorded = true;
        }

        m_dbgStartTimePt = Clock::now();
        m_cvStartProcRec.notify_all();
    }

    //------------------------------------------------------------------
    // Ensure pipeline 0 has recorded the start event before ungating other threads
    if(0 != instIdx)
    {
        std::unique_lock<std::mutex> cvStartProcRecMutexLock(m_cvStartProcRecMutex);
        m_cvStartProcRec.wait(cvStartProcRecMutexLock, [this] { return m_eStartProcRecorded; });
    }

    m_dbgTimePts0[instIdx] = Clock::now();

    // To launch the pipelines as close to each other as possible, all other pipeline streams wait for
    // a lwca event from the first pipeline stream. Since a lwca event can be waited on after it is
    // recorded, the first pipeline waits for run signal from main thread after recording the lwca
    // event and all other pipelines wait for lwca event wait after receiving the run signal from
    // main thread.
    LWDA_CHECK(lwdaStreamWaitEvent(lwStrm, m_eStartProc, 0));

    m_dbgTimePts1[instIdx] = Clock::now();

    if(m_sleepDurationUs)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(m_sleepDurationUs));
    }

    printf("%s Pipeline[%d]: Running\n", m_name.c_str(), instIdx);

    if(m_enableLwProf)
    {
        lwdaProfilerStart();
    }

    //------------------------------------------------------------------
    // Run the pipeline worker
    LWDA_CHECK(lwdaEventRecord(eStart, lwStrm));
    m_startTimePts[instIdx] = Clock::now();

    {
        for(uint32_t i = 0; i < m_nIterations; ++i)
        {
            for(int32_t slot = 0; slot < m_nSlots[instIdx]; slot++)
            {
                //------------------------------------------------------------------
                uint32_t slotBits = puschRx.expandParameters(m_slotDataBuf[instIdx][slot].tWFreq, m_lwrrentTbsPrmsArrays[instIdx][slot], m_BBUPrmsArray[instIdx][slot], m_lwStrms[instIdx]);
                //    LWDA_CHECK(lwdaStreamSynchronize(m_lwStrms[instIdx]));
                // Initialize pipelines
                if(i == m_nIterations - 1)
                {
                    printf("******* Running Slot %d *********\n", slot);
                    m_totalBits[instIdx] += slotBits;
                    m_puschRxList[instIdx].printInfo();
                }

                PuschRxDataset& slotDataBuf = m_slotDataBuf[instIdx][slot];
                puschRx.Run(lwStrm,
                            slotDataBuf.slotNumber,
                            slotDataBuf.tDataRx,
                            slotDataBuf.tShiftSeq,
                            slotDataBuf.tUnShiftSeq,
                            slotDataBuf.tDataSymLoc,
                            slotDataBuf.tQamInfo,
                            slotDataBuf.tNoisePwr,
                            m_descramblingOn,
                            ((m_nIterations - 1 == i)) ? m_debugFiles[instIdx][slot].get() : nullptr);

                if(i == m_nIterations - 1)
                {
                    puschRx.copyOutputToCPU(lwStrm);
                    LWDA_CHECK(lwdaStreamSynchronize(lwStrm));
                    m_perSlotCBcrcs[instIdx][slot].assign(puschRx.getCRCs(), puschRx.getCRCs() + puschRx.getCfgPrms().bePrms.CSum);
                    m_perSlotTBcrcs[instIdx][slot].assign(puschRx.getTbCRCs(), puschRx.getTbCRCs() + puschRx.getCfgPrms().bePrms.nTb);
                    m_perSlotOutputBytes[instIdx][slot].assign(puschRx.getTransportBlocks(), puschRx.getTransportBlocks() + puschRx.getCfgPrms().bePrms.totalTBByteSize);
                    m_perSlotCSum[instIdx][slot]   = puschRx.getCfgPrms().bePrms.CSum;
                    m_perSlotTBSize[instIdx][slot] = puschRx.getCfgPrms().bePrms.totalTBByteSize;
                }
            }
        }
    }
    // Some updates including potentially expensive atomic increment before GPU completes
    uint32_t tstCnt = m_puschRxList.size();
    uint32_t newCnt = 0;
    m_atmEndProcCnt++;

    // printf("%s Pipeline[%d]: End processing\n", m_name.c_str(), instIdx);
    LWDA_CHECK(lwdaEventRecord(eStop, lwStrm));
    LWDA_CHECK(lwdaStreamSynchronize(lwStrm));
    LWDA_CHECK(lwdaEventSynchronize(eStop));

    m_stopTimePts[instIdx] = Clock::now();

    //------------------------------------------------------------------
    // Record an event at the end of last pipeline completion

    // m_atmEndProcCnt.compare_exchange_strong(tstCnt, newCnt)) returns true if (tstCnt == m_atmEndProcCnt)
    // if (tstCnt != m_atmEndProcCnt) => tstCnt = m_atmEndProcCnt
    // if (tstCnt == m_atmEndProcCnt) => m_atmEndProcCnt = newCnt
    if(m_atmEndProcCnt.compare_exchange_strong(tstCnt, newCnt))
    {
        LWDA_CHECK(lwdaEventRecord(m_eEndProc, lwStrm));
        printf("%s Pipeline[%d]: End processing recorded, profile end time (wall clock) %lu\n",
               m_name.c_str(),
               instIdx,
               std::chrono::duration_cast<std::chrono::microseconds>(m_stopTimePts[instIdx].time_since_epoch()).count());
    }
    uint32_t atmEndProcCnt = m_atmEndProcCnt.load();
    // printf("%s Pipeline[%d]: End processing count %d\n", m_name.c_str(), instIdx, atmEndProcCnt);

    if(m_enableLwProf)
    {
        lwdaProfilerStop();
    }

    printf("%s Pipeline[%d]: Profile start time (wall clock) %lu\n",
           m_name.c_str(),
           instIdx,
           std::chrono::duration_cast<std::chrono::microseconds>(m_startTimePts[instIdx].time_since_epoch()).count());
}

void MultiSlotTest::WaitForCompletion()
{
    // Join all threads
    uint32_t instIdx = 0;
    for(auto& wrkrThrd : m_wrkrThrds)
    {
        wrkrThrd.join();
        printf("%s Pipeline[%d]: Joining worker thread\n", m_name.c_str(), instIdx);
        instIdx++;
    }
}

void MultiSlotTest::DisplayMetrics(bool debug, bool throughput, bool bler, bool execTime)
{
    for(uint32_t instIdx = 0; instIdx < m_puschRxList.size(); ++instIdx)
    {
        DisplayInstMetrics(instIdx, debug, throughput, bler, execTime);
    }

    if(execTime && debug)
    {
        float elapsedMs = 0.0f;
        lwdaEventElapsedTime(&elapsedMs, m_eStartProc, m_eEndProc);

        printf("---------------------------------------------------------------\n");
        printf("%s all pipelines: Total average exelwtion time: %4.4f usec (over %d runs, using LWCA event)\n",
               m_name.c_str(),
               elapsedMs * 1000 / m_nIterations,
               m_nIterations);
    }
}

void MultiSlotTest::DisplayInstMetrics(uint32_t instIdx, bool debug, bool throughput, bool bler, bool execTime)
{
    PuschRx&      puschRx   = m_puschRxList[instIdx];
    lwdaStream_t& lwStrm    = m_lwStrms[instIdx];
    lwdaEvent_t&  eStopProc = m_eStopProcs[instIdx];

    float elapsedMs = 0.0f;
    lwdaEventElapsedTime(&elapsedMs, m_eStartProc, eStopProc);

    printf("---------------------------------------------------------------\n");
    if(throughput)
    {
        printf("%s Pipeline[%d]: Metric - Throughput            : %4.4f Gbps (encoded input bits %d) \n",
               m_name.c_str(),
               instIdx,
               (static_cast<float>(m_totalBits[instIdx]) / ((elapsedMs * 1e-3) / m_nIterations)) / 1e9,
               m_totalBits[instIdx]);
    }

    if(bler)
    {
        // Tally CRC errors

        uint32_t nCRCErrors = 0;
        uint32_t CSum       = 0;
        for(int slot = 0; slot < m_nSlots[instIdx]; slot++)
        {
            const uint32_t* crcs = m_perSlotCBcrcs[instIdx][slot].data();
            CSum += m_perSlotCSum[instIdx][slot];
            for(int i = 0; i < m_perSlotCSum[instIdx][slot]; i++)
            {
                if(crcs[i] != 0)
                {
                    nCRCErrors++;
                }
            }
        }
        printf("%s Pipeline[%d]: Metric - Block Error Rate      : %4.4f (Error CBs %d, Total CBs %d)\n",
               m_name.c_str(),
               instIdx,
               static_cast<float>(nCRCErrors) / static_cast<float>(CSum),
               nCRCErrors,
               CSum);
    }

    if(execTime)
    {
        printf("%s Pipeline[%d]: Metric - Average exelwtion time: %4.4f usec (over %d runs, using LWCA event)\n",
               m_name.c_str(),
               instIdx,
               elapsedMs * 1000 / m_nIterations,
               m_nIterations);

        lwdaEventElapsedTime(&elapsedMs, m_eStartProcs[instIdx], eStopProc);
        printf("%s Pipeline[%d]: Metric - Average exelwtion time (excluding start delay): %4.4f usec (over %d runs, using LWCA event)\n",
               m_name.c_str(),
               instIdx,
               elapsedMs * 1000 / m_nIterations,
               m_nIterations);

        lwdaEventElapsedTime(&elapsedMs, m_eStartProc, m_eStartProcs[instIdx]);
        printf("%s Pipeline[%d]: Kernel launch start delay: %4.4f usec, amortized per run %4.4f usec (over %d runs, using LWCA event)\n",
               m_name.c_str(),
               instIdx,
               elapsedMs * 1000,
               elapsedMs * 1000 / m_nIterations,
               m_nIterations);

        if(debug)
        {
            duration<float, std::micro> diff = m_stopTimePts[instIdx] - m_startTimePts[instIdx];
            printf("%s Pipeline[%d]: Average (%d runs) elapsed time in usec (wall clock w/ %u ms delay kernel) = %4.4f\n",
                   m_name.c_str(),
                   instIdx,
                   m_nIterations,
                   m_delayMs,
                   diff.count() / m_nIterations);

            diff = m_dbgTimePts0[instIdx] - m_dbgStartTimePt;
            printf("%s Pipeline[%d]: Debug - start-event record to notify delay in usec (wall clock) = %4.4f\n",
                   m_name.c_str(),
                   instIdx,
                   diff.count());

            diff = m_startTimePts[instIdx] - m_dbgTimePts0[instIdx];
            printf("%s Pipeline[%d]: Debug - start-event notify to pipelne launch start delay in usec (wall clock) = %4.4f\n",
                   m_name.c_str(),
                   instIdx,
                   diff.count());
        }
    }

    //------------------------------------------------------------------
    // Display CRC errors if any

    for(int slot = 0; slot < m_nSlots[instIdx]; slot++)
    {
        const uint32_t* crcs            = m_perSlotCBcrcs[instIdx][slot].data();
        const uint32_t* tbCRCs          = m_perSlotTBcrcs[instIdx][slot].data();
        const uint8_t*  transportBlocks = m_perSlotOutputBytes[instIdx][slot].data();

        // write output bits to file
        std::ofstream of(OUTPUT_TB_FNAME + std::string("_") + std::to_string(instIdx) + "_" + std::to_string(slot), std::ofstream::binary);
        of.write(reinterpret_cast<const char*>(transportBlocks), (m_totalBits[instIdx] / 8) + ((m_totalBits[instIdx] % 8) != 0));
        of.close();

        for(int i = 0; i < m_perSlotCSum[instIdx][slot]; i++)
        {
            if(crcs[i] != 0)
            {
                printf("ERROR: %s Pipeline[%d] slot[%d]: CRC of code block [%d] failed!\n", m_name.c_str(), instIdx, slot, i);
            }
        }
        for(int i = 0; i < m_perSlotnTb[instIdx][slot]; i++)
        {
            if(tbCRCs[i] != 0)
            {
                printf("ERROR: %s Pipeline[%d] slot[%d]: CRC of transport block [%d] failed!\n", m_name.c_str(), instIdx, slot, i);
            }
        }
    }
}
