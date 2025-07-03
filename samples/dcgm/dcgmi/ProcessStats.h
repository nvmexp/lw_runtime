/*
 * ProcessStats.h
 *
 *  Created on: Oct 16, 2015
 *      Author: chris
 */

#ifndef PROCESSSTATS_H_
#define PROCESSSTATS_H_

#include "Command.h"

class ProcessStats {
public:
    ProcessStats();
    virtual ~ProcessStats();

    /*****************************************************************************
     * This method is used to enable watches on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t EnableWatches(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t groupId);

    /*****************************************************************************
     * This method is used to disable watches on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t DisableWatches(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t groupId);
    /*****************************************************************************
     * This method is used to display process stats on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t ViewProcessStats(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t groupId, unsigned int pid, bool verbose);

    /*****************************************************************************
     * This method is used to start a job on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t StartJob(dcgmHandle_t mLwcmHandle, dcgmGpuGrp_t groupId, std::string jobId);

    /*****************************************************************************
     * This method is used to stop a job on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t StopJob(dcgmHandle_t mLwcmHandle, std::string jobId);

    /*****************************************************************************
     * This method is used to display job stats on the host-engine represented by the
     * DCGM handle
     *****************************************************************************/
    dcgmReturn_t ViewJobStats(dcgmHandle_t mLwcmHandle, std::string jobId, bool verbose);

    /*****************************************************************************
     * This method is used to remove job stats for a job on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RemoveJob(dcgmHandle_t mLwcmHandle, std::string jobId);

    /*****************************************************************************
     * This method is used to remove all job stats on the host-engine represented
     * by the DCGM handle
     *****************************************************************************/
    dcgmReturn_t RemoveAllJobs(dcgmHandle_t mLwcmHandle);

private:
    /*****************************************************************************
     * Function to display all exelwtion stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidExelwtionStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
     *   Function to display all event stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidEventStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
    Function to display all performance stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidPerformanceStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);
    /*****************************************************************************
     * Function to display all violation stats from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidViolationStats(dcgmPidSingleInfo_t *pidInfo, bool verbose);

    /*****************************************************************************
     * Function to display process utilization from given pidInfo
     *****************************************************************************/
    void HelperDisplayPidProcessUtilization(dcgmPidSingleInfo_t* pidInfo, bool verbose);

    /*****************************************************************************
     * Function to display over all Health of the system while the process is running
     *****************************************************************************/
    void HelperDisplayOverAllHealth(dcgmPidSingleInfo_t* pidInfo, bool verbose);


    /*****************************************************************************
     * Function to display all exelwtion stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobExelwtionStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
     *   Function to display all event stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobEventStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
        Function to display all performance stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobPerformanceStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    /*****************************************************************************
     * Function to display all violation stats from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobViolationStats(dcgmGpuUsageInfo_t *jobInfo, bool verbose);
    
    /*****************************************************************************
     * Function to display process utilization from given jobInfo
     *****************************************************************************/
    void HelperDisplayJobProcessUtilization(dcgmGpuUsageInfo_t* jobInfo, bool verbose);

    
    /*****************************************************************************
     * Function to display over all Health of the system while the job was running
     *****************************************************************************/
    void HelperDisplayOverAllHealth(dcgmGpuUsageInfo_t* jobInfo, bool verbose);
    
    /*****************************************************************************
     *  Colwerts double count of seconds to hr:min:sec format
     *****************************************************************************/
    std::string HelperFormatTotalTime(double time);
    /*****************************************************************************
     * Colwerts long long timeseries into a human readable date and time
     *****************************************************************************/
    std::string HelperFormatTimestamp(long long ts, bool isStartTime);

    /*****************************************************************************
     * Colwerts a time difference into a string. If one or both are blank, prints
     * the appropriate error text.
     *****************************************************************************/
    std::string HelperFormatTimeDifference(long long timestamp1, long long timestamp2);

    /*****************************************************************************
     * Colwerts long long times into percentage, checking for DCGM_INT64_BLANK prior
     *****************************************************************************/
    double HelperFormatPercentTime(long long denominator, long long numerator);

    /*****************************************************************************
     * Colwerts a 32 bit interger stat summary into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStat32Summary(dcgmStatSummaryInt32_t &summary);

    /*****************************************************************************
     * Colwerts a double stat summary into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStatFp64Summary(dcgmStatSummaryFp64_t &summary);

    /*****************************************************************************
     *Colwerts a 64 bit interger stat sumamry into avg: min: max: string
     *****************************************************************************/
    std::string HelperFormatStatPCISummary(dcgmStatSummaryInt64_t &summary);

    /*****************************************************************************
     * Checks if integer is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string Helper32IntToString(int num);

    /*****************************************************************************
     * Checks if double is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string HelperDoubleToString(double num);

    /*****************************************************************************
     * Checks if long long is blank and returns N/A if so. Number as string otherwise
     *****************************************************************************/
    std::string HelperPCILongToString(long long num);

    std::string HelperHealthSystemToString(dcgmHealthSystems_enum system);

    std::string HelperHealthResultToString(dcgmHealthWatchResults_t health);
};


/**
 * Enable Watches Ilwoker
 */
class EnableWatches : public Command
{
public:
    EnableWatches(std::string hostname, unsigned int groupId);
    virtual ~EnableWatches();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Enable Watches Ilwoker
 */
class DisableWatches : public Command
{
public:
    DisableWatches(std::string hostname, unsigned int groupId);
    virtual ~DisableWatches();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Get Process Information Watches Ilwoker
 */
class ViewProcessStats : public Command
{
public:
    ViewProcessStats(std::string hostname, unsigned int mGroupId, unsigned int pid, bool verbose);
    virtual ~ViewProcessStats();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    unsigned int mPid;
    bool mVerbose;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Start Job Ilwoker
 */
class StartJob : public Command
{
public:
    StartJob(std::string hostname, unsigned int groupId, std::string jobId);
    virtual ~StartJob();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
    dcgmGpuGrp_t mGroupId;
};

/**
 * Stop Job Ilwoker
 */
class StopJob : public Command
{
public:
    StopJob(std::string hostname, std::string jobId);
    virtual ~StopJob();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
};

/**
 * Remove Job Ilwoker
 */
class RemoveJob : public Command
{
public:
    RemoveJob(std::string hostname, std::string jobId);
    virtual ~RemoveJob();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
};

/**
 * Remove All Jobs Ilwoker
 */
class RemoveAllJobs : public Command
{
public:
    RemoveAllJobs(std::string hostname);
    virtual ~RemoveAllJobs();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
};


/**
 * Get Process Information Watches Ilwoker
 */
class ViewJobStats : public Command
{
public:
    ViewJobStats(std::string hostname, std::string jobId, bool verbose);
    virtual ~ViewJobStats();

    int Execute();

private:
    ProcessStats mProcessStatsObj;
    std::string jobId;
    bool verbose;
};


#endif /* PROCESSSTATS_H_ */
