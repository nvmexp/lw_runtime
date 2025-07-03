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

#include "pusch_rx_test.hpp"
#include "hdf5hpp.hpp"

#define OUTPUT_TB_FNAME ("outputBits")

template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

PuschRxTest::PuschRxTest(std::string const& name, uint32_t nPuschRxInst, lwdaStream_t& delayLwStrm, std::vector<int>& lwStrmPrios, bool& startSyncPt, std::mutex& cvStartSyncPtMutex, std::condition_variable& cvStartSyncPt, std::atomic<std::uint32_t>& atmSyncPtWaitCnt) :
    m_name(name),
    m_puschRxList(nPuschRxInst),
    m_lwStrms(nPuschRxInst, 0),
    m_wrkrThrds(nPuschRxInst),
    m_wrkrThrdMutexes(nPuschRxInst),
    m_startSyncPt(startSyncPt),
    m_cvStartSyncPtMutex(cvStartSyncPtMutex),
    m_cvStartSyncPt(cvStartSyncPt),
    m_atmSyncPtWaitCnt(atmSyncPtWaitCnt),
    m_eStartProcRecorded(false),
    m_totalBits(nPuschRxInst, 0),
    m_enableLwProf(false),
    m_nIterations(0),
    m_descramblingOn(true),
    m_delayMs(0),
    m_sleepDurationUs(0),
    m_delayLwStrm(delayLwStrm),
    m_eStartProcs(nPuschRxInst),
    m_eStopProcs(nPuschRxInst),
    m_atmEndProcCnt(0),
    m_startTimePts(nPuschRxInst),
    m_stopTimePts(nPuschRxInst),
    // m_dbgStartTimePt,
    m_dbgTimePts0(nPuschRxInst),
    m_dbgTimePts1(nPuschRxInst),
    m_dbgTimePts2(nPuschRxInst),
    m_dbgTimePts3(nPuschRxInst)
{
    // LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProc, lwdaEventDisableTiming));
    LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProc, lwdaEventBlockingSync));
    LWDA_CHECK(lwdaEventCreateWithFlags(&m_eEndProc, lwdaEventBlockingSync));

    for(uint32_t instIdx = 0; instIdx < nPuschRxInst; ++instIdx)
    {
        LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStartProcs[instIdx], lwdaEventBlockingSync));
        LWDA_CHECK(lwdaEventCreateWithFlags(&m_eStopProcs[instIdx], lwdaEventBlockingSync));

        // LWDA_CHECK(lwdaStreamCreateWithFlags(&lwStrm, lwdaStreamNonBlocking));
        LWDA_CHECK(lwdaStreamCreateWithPriority(&m_lwStrms[instIdx], lwdaStreamNonBlocking, lwStrmPrios[instIdx]));
    }
}

PuschRxTest::~PuschRxTest()
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

void PuschRxTest::Setup(uint32_t                nIterations,
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

    if(!outputfilename.empty())
    {
        m_debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputfilename.c_str())));
    }

    //-------------------------------------------------------------
    // Check for configuration information in the input file. Newer
    // input files will have configuration values in the file, so
    // that they don't need to be specified on the command line.
    lwphy::disable_hdf5_error_print(); // Temporarily disable HDF5 stderr printing

    std::vector<tb_pars> lwrrentTbsPrmsArray;
    gnb_pars             BBUPrms;
    uint32_t             slotNumber = 0;

    try
    {
        lwphy::lwphyHDF5_struct gnbConfig = lwphy::get_HDF5_struct(fInput, "gnb_pars");
        slotNumber                        = gnbConfig.get_value_as<uint32_t>("slotNumber");
        BBUPrms.fc                        = gnbConfig.get_value_as<uint32_t>("fc");
        BBUPrms.mu                        = gnbConfig.get_value_as<uint32_t>("mu");
        BBUPrms.nRx                       = gnbConfig.get_value_as<uint32_t>("nRx");
        BBUPrms.nPrb                      = gnbConfig.get_value_as<uint32_t>("nPrb");
        BBUPrms.cellId                    = gnbConfig.get_value_as<uint32_t>("cellId");
        BBUPrms.slotNumber                = gnbConfig.get_value_as<uint32_t>("slotNumber");
        BBUPrms.Nf                        = gnbConfig.get_value_as<uint32_t>("Nf");
        BBUPrms.Nt                        = gnbConfig.get_value_as<uint32_t>("Nt");
        BBUPrms.df                        = gnbConfig.get_value_as<uint32_t>("df");
        BBUPrms.dt                        = gnbConfig.get_value_as<uint32_t>("dt");
        BBUPrms.numBsAnt                  = gnbConfig.get_value_as<uint32_t>("numBsAnt");
        BBUPrms.numBbuLayers              = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
        BBUPrms.numTb                     = gnbConfig.get_value_as<uint32_t>("numTb");
        BBUPrms.ldpcnIterations           = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
        BBUPrms.ldpcEarlyTermination      = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
        BBUPrms.ldpcAlgoIndex             = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
        BBUPrms.ldpcFlags                 = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
        if(getelw("LDPC_USE_HALF"))
        {
            BBUPrms.ldplwseHalf = 1;
        }
        else
        {
            BBUPrms.ldplwseHalf = gnbConfig.get_value_as<uint32_t>("ldplwseHalf");
        }

        lwrrentTbsPrmsArray.resize(BBUPrms.numTb);

        // parse array of tb_pars structs

        hdf5hpp::hdf5_dataset tbpDset = fInput.open_dataset("tb_pars");

        for(int i = 0; i < BBUPrms.numTb; i++)
        {
            lwphy::lwphyHDF5_struct tbConfig        = lwphy::get_HDF5_struct_index(tbpDset, i);
            lwrrentTbsPrmsArray[i].numLayers        = tbConfig.get_value_as<uint32_t>("numLayers");
            lwrrentTbsPrmsArray[i].layerMap         = tbConfig.get_value_as<uint32_t>("layerMap");
            lwrrentTbsPrmsArray[i].startPrb         = tbConfig.get_value_as<uint32_t>("startPrb");
            lwrrentTbsPrmsArray[i].numPrb           = tbConfig.get_value_as<uint32_t>("numPRb");
            lwrrentTbsPrmsArray[i].startSym         = tbConfig.get_value_as<uint32_t>("startSym");
            lwrrentTbsPrmsArray[i].numSym           = tbConfig.get_value_as<uint32_t>("numSym");
            lwrrentTbsPrmsArray[i].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
            lwrrentTbsPrmsArray[i].dataScramId      = tbConfig.get_value_as<uint32_t>("dataScramId");
            lwrrentTbsPrmsArray[i].mcsTableIndex    = tbConfig.get_value_as<uint32_t>("mcsTableIndex");
            lwrrentTbsPrmsArray[i].mcsIndex         = tbConfig.get_value_as<uint32_t>("mcsIndex");
            lwrrentTbsPrmsArray[i].rv               = tbConfig.get_value_as<uint32_t>("rv");
            lwrrentTbsPrmsArray[i].dmrsType         = tbConfig.get_value_as<uint32_t>("dmrsType");
            lwrrentTbsPrmsArray[i].dmrsAddlPosition = tbConfig.get_value_as<uint32_t>("dmrsAddlPosition");
            lwrrentTbsPrmsArray[i].dmrsMaxLength    = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
            lwrrentTbsPrmsArray[i].dmrsScramId      = tbConfig.get_value_as<uint32_t>("dmrsScramId");
            lwrrentTbsPrmsArray[i].dmrsEnergy       = tbConfig.get_value_as<uint32_t>("dmrsEnergy");
            lwrrentTbsPrmsArray[i].nRnti            = tbConfig.get_value_as<uint32_t>("nRnti");
            lwrrentTbsPrmsArray[i].dmrsCfg          = tbConfig.get_value_as<uint32_t>("dmrsCfg");
        }
        // END NEW PARAM STRUCT PARSING
    }
    catch(const std::exception& exc)
    {
        printf("%s\n", exc.what());
        throw exc;
        // Continue using command line arguments if the input file does not
        // have a config struct.
    }
    lwphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

    //-------------------------------------------------------------
    // Check FP16 mode of operation
    bool isDataRxFp16  = true;
    //bool isChannelFp16 = false;
    switch(fp16Mode)
    {
    case 0:
        isDataRxFp16  = false;
        //isChannelFp16 = false;
        break;
    case 1:
        isDataRxFp16  = true;
        //isChannelFp16 = false;
        break;
    case 2:
        isDataRxFp16  = true;
        //isChannelFp16 = true;
        break;
    default:
        isDataRxFp16  = false;
        //isChannelFp16 = false;
        break;
    }
    //lwphyDataType_t typeDataRx     = isDataRxFp16 ? LWPHY_R_16F : LWPHY_R_32F;
    lwphyDataType_t cplxTypeDataRx = isDataRxFp16 ? LWPHY_C_16F : LWPHY_C_32F;

    //lwphyDataType_t typeFeChannel     = isChannelFp16 ? LWPHY_R_16F : LWPHY_R_32F;
    //lwphyDataType_t cplxTypeFeChannel = isChannelFp16 ? LWPHY_C_16F : LWPHY_C_32F;

    for(int i = 0; i < PuschRxTest::N_SLOT_DATA_BUF; i++)
    {
        m_slotDataBuf.emplace_back(PuschRxDataset(fInput, slotNumber, "", cplxTypeDataRx));
    }

    m_slotDataBuf[0].printInfo(0);

    //------------------------------------------------------------------
    // Initialize pipelines
    uint32_t instIdx = 0;
    for(auto& puschRx : m_puschRxList)
    {
        m_totalBits[instIdx] = puschRx.expandParameters(m_slotDataBuf[0].tWFreq, lwrrentTbsPrmsArray, BBUPrms, m_lwStrms[instIdx]);
        LWDA_CHECK(lwdaStreamSynchronize(m_lwStrms[instIdx]));
        instIdx++;
    }

    // Display info for one instance
    m_puschRxList[0].printInfo();

    // All transfers in PuschRxDataset (specifically tensor_from_dataset calls) are done via stream 0
    LWDA_CHECK(lwdaStreamSynchronize(0));

    SpawnPipelineWrkrThrds();
}

void PuschRxTest::SpawnPipelineWrkrThrds()
{
    // Launch CPU worker threads which will wait for run condition
    uint32_t instIdx = 0;
    for(auto& wrkrThrd : m_wrkrThrds)
    {
        wrkrThrd = std::thread(&PuschRxTest::PipelineWrkrEntry,
                               this,
                               instIdx);
        instIdx++;
    }
}

void PuschRxTest::PipelineWrkrEntry(uint32_t instIdx)
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
        // model conlwrrent workload submission from all streams (i.e. pipelines) as close to each other
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
            uint32_t slotDim = m_slotDataBuf[0].tDataRx.rank() - 1;
            uint32_t nslots  = (m_slotDataBuf[0].tDataRx.rank() < 4) ? 1 : m_slotDataBuf[0].tDataRx.layout().dimensions()[slotDim];
            for(uint32_t slot = 0; slot < nslots; slot++)
            {
                PuschRxDataset& slotDataBuf = m_slotDataBuf[i % PuschRxTest::N_SLOT_DATA_BUF];
                puschRx.Run(lwStrm,
                            slotDataBuf.slotNumber,
                            slotDataBuf.tDataRx,
                            slotDataBuf.tShiftSeq,
                            slotDataBuf.tUnShiftSeq,
                            slotDataBuf.tDataSymLoc,
                            slotDataBuf.tQamInfo,
                            slotDataBuf.tNoisePwr,
                            m_descramblingOn,
                            ((0 == i) && (0 == instIdx)) ? m_debugFile.get() : nullptr);
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

void PuschRxTest::WaitForCompletion()
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

void PuschRxTest::DisplayMetrics(bool debug, bool throughput, bool bler, bool execTime)
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

void PuschRxTest::DisplayInstMetrics(uint32_t instIdx, bool debug, bool throughput, bool bler, bool execTime)
{
    PuschRx&      puschRx   = m_puschRxList[instIdx];
    lwdaStream_t& lwStrm    = m_lwStrms[instIdx];
    lwdaEvent_t&  eStopProc = m_eStopProcs[instIdx];

    float elapsedMs = 0.0f;
    lwdaEventElapsedTime(&elapsedMs, m_eStartProc, eStopProc);

    const uint32_t* crcs = puschRx.getCRCs();

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
        puschRx.copyOutputToCPU(lwStrm);
        LWDA_CHECK(lwdaStreamSynchronize(lwStrm));

        uint32_t nCRCErrors = 0;
        for(int i = 0; i < puschRx.getCfgPrms().bePrms.CSum; i++)
        {
            if(crcs[i] != 0)
            {
                nCRCErrors++;
            }
        }
        printf("%s Pipeline[%d]: Metric - Block Error Rate      : %4.4f (Error CBs %d, Total CBs %d)\n",
               m_name.c_str(),
               instIdx,
               static_cast<float>(nCRCErrors) / static_cast<float>(puschRx.getCfgPrms().bePrms.CSum),
               nCRCErrors,
               puschRx.getCfgPrms().bePrms.CSum);
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
    const uint32_t* tbCRCs          = puschRx.getTbCRCs();
    const uint8_t*  transportBlocks = puschRx.getTransportBlocks();

    // write output bits to file
    std::ofstream of(OUTPUT_TB_FNAME, std::ofstream::binary);
    of.write(reinterpret_cast<const char*>(transportBlocks), (m_totalBits[instIdx] / 8) + ((m_totalBits[instIdx] % 8) != 0));
    of.close();

    for(int i = 0; i < puschRx.getCfgPrms().bePrms.CSum; i++)
    {
        if(crcs[i] != 0)
        {
            printf("ERROR: %s Pipeline[%d]: CRC of code block [%d] failed!\n", m_name.c_str(), instIdx, i);
        }
    }
    for(int i = 0; i < puschRx.getCfgPrms().bePrms.nTb; i++)
    {
        if(tbCRCs[i] != 0)
        {
            printf("ERROR: %s Pipeline[%d]: CRC of transport block [%d] failed!\n", m_name.c_str(), instIdx, i);
        }
    }
}
