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
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>
#include "pusch_rx.hpp"

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

#if 0
class PipelineTest
{

};
#endif

class PuschRxTest {
public:
    void Setup(uint32_t                nIterations,
               bool                    enableLwprof,
               int                     descramblingOn,
               uint32_t                delayMs,
               uint32_t                sleepDurationUs,
               std::vector<int> const& cpuIds,
               std::vector<int> const& thrdSchdPolicies,
               std::vector<int> const& thrdPrios,
               std::vector<int> const& gpuIds,
               const std::string&      inputFilename,
               std::string&            outputfilename,
               uint32_t                fp16Mode);

    void PipelineWrkrEntry(uint32_t instIdx);
    void WaitForCompletion();
    void DisplayMetrics(bool debug = false, bool throughput = true, bool bler = true, bool execTime = true);

    PuschRxTest(std::string const& name, uint32_t nPuschRxInst, lwdaStream_t& delayLwStrm, std::vector<int>& lwStrmPrios, bool& startSyncPt, std::mutex& cvStartSyncPtMutex, std::condition_variable& cvStartSyncPt, std::atomic<std::uint32_t>& atmSyncPtWaitCnt);
    PuschRxTest(PuschRxTest const&) = delete;
    PuschRxTest& operator=(PuschRxTest const&) = delete;
    ~PuschRxTest();

    static constexpr uint32_t N_SLOT_DATA_BUF = 64;

private:
    void SpawnPipelineWrkrThrds();
    void DisplayInstMetrics(uint32_t instIdx, bool debug, bool throughput, bool bler, bool execTime);

    std::string                 m_name;
    std::vector<PuschRx>        m_puschRxList; // no copy construction in vector member tensor
    std::vector<lwdaStream_t>   m_lwStrms;
    std::vector<PuschRxDataset> m_slotDataBuf;
    std::vector<std::thread>    m_wrkrThrds;
    std::vector<std::mutex>     m_wrkrThrdMutexes;

    // Condition variable and supporting mutex to issue sync point from signal from main thread to all threads
    // and pipelines
    bool&                       m_startSyncPt;
    std::mutex&                 m_cvStartSyncPtMutex;
    std::condition_variable&    m_cvStartSyncPt;
    std::atomic<std::uint32_t>& m_atmSyncPtWaitCnt;

    // Condition variable and supporting mutex to signal from pipeline 0 back to main thread
    bool                    m_eStartProcRecorded = false;
    std::mutex              m_cvStartProcRecMutex;
    std::condition_variable m_cvStartProcRec;

    std::vector<int> m_wrkrThrdSchdPolicies;
    std::vector<int> m_wrkrThrdPrios;

    std::vector<int> m_cpuIds;
    std::vector<int> m_gpuIds;

    std::unique_ptr<hdf5hpp::hdf5_file> m_debugFile;
    std::vector<uint32_t>               m_totalBits;

    bool     m_enableLwProf    = false;
    uint32_t m_nIterations     = 0;
    int      m_descramblingOn  = true;
    uint32_t m_delayMs         = 0;
    uint32_t m_sleepDurationUs = 0;

    lwdaStream_t               m_delayLwStrm = 0; // LWCA stream on which the delay kernel is launched
    lwdaEvent_t                m_eStartProc;
    lwdaEvent_t                m_eEndProc;
    std::vector<lwdaEvent_t>   m_eStartProcs;
    std::vector<lwdaEvent_t>   m_eStopProcs;
    std::atomic<std::uint32_t> m_atmEndProcCnt;

    std::vector<TimePoint> m_startTimePts, m_stopTimePts;

    // For debug
    TimePoint              m_dbgStartTimePt, m_dbgTimePt0, m_dbgTimePt1, m_dbgTimePt2, m_dbgTimePt3;
    std::vector<TimePoint> m_dbgTimePts0, m_dbgTimePts1, m_dbgTimePts2, m_dbgTimePts3;
};
