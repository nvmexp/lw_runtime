
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "LwcmHostEngineHandler.h"
#include "DcgmModuleLwSwitch.h"
#include "dcgm_lwswitch_structs.h"
#include "LwcmProtobuf.h"
#include "dcgm_structs.h"
#include "DcgmLocalStatsReporter.h"
#include "LwcmServerConnection.h"
#include "fabricmanager.pb.h"
#include "LwcmSettings.h"

#include "DcgmLogging.h"
#include "logging.h"

/*****************************************************************************/
DcgmLocalStatsReporter::DcgmLocalStatsReporter(DcgmLocalFabricManagerControl *pLfm,
                                               DcgmLWSwitchPhyIdToFdInfoMap &switchIdToFdInfoMap,
                                               DcgmLFMLWLinkDevRepo *linkDevRepo)
{
    dcgmReturn_t retVal;

    mLWLinkDevRepo = linkDevRepo;
    mpLfm = pLfm;
    mSwitchIdToFdInfoMap = switchIdToFdInfoMap;

    // initialize the GPU index and CacheManager LWLink Watch information
    // only in non shared fabric mode,
    // because the GPUs are not with service VM in shared fabric mode
    if ( !mpLfm->isSharedFabricMode() )
    {
        retVal = registerGpuLWLinkErrorWatch();
        if ( DCGM_ST_OK != retVal )
        {
            PRINT_ERROR("", "failed to register all the LWLink error watch events for GPUs");
            throw std::runtime_error("failed to register all the LWLink error watch events for GPUs");
        }
    }
};

DcgmLocalStatsReporter::~DcgmLocalStatsReporter()
{
    // worst case wait for 1 second more than the time used in poll() timeout
    StopAndWait((FATAL_ERROR_POLLING_TIMEOUT + 1) * 1000);
    // make sure that the error polling thread is exited before unregistering
    // any DCGM GPU watch to avoid any potential get latest sample request to DCGM.

    if ( !mpLfm->isSharedFabricMode() )
    {
        unRegisterGpuLWLinkErrorWatch();
    }
};

dcgmReturn_t
DcgmLocalStatsReporter::registerGpuLWLinkErrorWatch(void)
{
    dcgmReturn_t retVal;
    DcgmWatcher watcher(DcgmWatcherTypeFabricManager);

    // Register with LWCMCacheManager for watching LWLink Error events from RM.
    // The events are per GPU, so register for all the available GPUs.
    // TODO: fix the GPU ID if this is ever used in any dynamic gpu attach/detach context.
    retVal = LwcmHostEngineHandler::Instance()->GetLwcmGpuIds( mDcgmGpuIds, 1 );
    if ( DCGM_ST_OK != retVal )
    {
        PRINT_ERROR( "%d", "GetLwcmGpuIds failed with %d", retVal );
        return retVal;
    }

    std::vector<unsigned int>::iterator it = mDcgmGpuIds.begin();
    for ( ; it != mDcgmGpuIds.end(); it++ )
    {
        unsigned int gpuIndex = (*it);
        // FM is interested in volta based GPUs only for now.
        lwmlChipArchitecture_t arch;
        LwcmHostEngineHandler::Instance()->GetLwcmGpuArch( gpuIndex, arch );
        if ( arch != LWML_CHIP_ARCH_VOLTA )
            continue;
        
        /* 1-hour polling. the fields below are handled continuously by their own thread. 
           there's no reason to poll them, so we're just using a very large polling value 
           to prevent the polling loop from waking up for them */
        long long watchFreq = 3600000000; 

        retVal = LwcmHostEngineHandler::Instance()->WatchFieldValue( DCGM_FE_GPU, gpuIndex, 
                                            DCGM_FI_DEV_GPU_LWLINK_ERRORS, watchFreq, 0, 100, watcher );
        if ( DCGM_ST_OK != retVal )
        {
            PRINT_ERROR( "%d", "GPU LWLink error watch field request failed with %d", retVal );
            return retVal;
        }

        // build the map of GPUs to be polled with initial samples as empty
        dcgmcm_sample_t dcgmSample = {0};
        mGpuLWLinkErrSample.insert( std::make_pair(gpuIndex, dcgmSample) );
    }

    return retVal;
}

dcgmReturn_t
DcgmLocalStatsReporter::unRegisterGpuLWLinkErrorWatch(void)
{
    dcgmReturn_t retVal = DCGM_ST_OK;
    DcgmWatcher watcher(DcgmWatcherTypeFabricManager);

    std::vector<unsigned int>::iterator it = mDcgmGpuIds.begin();
    for ( ; it != mDcgmGpuIds.end(); it++ )
    {
        unsigned int gpuIndex = (*it);
        // LWLink recovery error watch is supported for volta based GPUs only        
        lwmlChipArchitecture_t arch;
        LwcmHostEngineHandler::Instance()->GetLwcmGpuArch( gpuIndex, arch );
        if ( arch != LWML_CHIP_ARCH_VOLTA )
            continue;

        retVal = LwcmHostEngineHandler::Instance()->UnwatchFieldValue( DCGM_FE_GPU, gpuIndex, 
                                            DCGM_FI_DEV_GPU_LWLINK_ERRORS, 1, watcher );
        if (DCGM_ST_OK != retVal)
        {
            PRINT_ERROR( "%d", "GPU LWLink error unwatch field request failed with %d", retVal );
            return retVal;
        }
    }

    mDcgmGpuIds.clear();
    mGpuLWLinkErrSample.clear();
    return retVal;
}

void
DcgmLocalStatsReporter::run()
{
    struct pollfd *pfds;
    nfds_t nfds;
    sigset_t mask;
    sigset_t orig_mask;
    int rc;
    DcgmLWSwitchPhyIdToFdInfoMap::iterator it;

    nfds = mSwitchIdToFdInfoMap.size();
    pfds = (struct pollfd *) calloc( (sizeof(struct pollfd) * nfds), 1 );
    if ( !pfds )
    {
        PRINT_ERROR(" ", "Failed to allocate memory for all the polling switch file descriptors");
        return;
    }

    // set sigmask
    sigemptyset (&mask);
    if ( sigprocmask( SIG_BLOCK, &mask, &orig_mask ) < 0 )
    {
        PRINT_ERROR("%d", "Failed to mask all the signals for error/stats polling, errno %d", errno);
        free( pfds );
        return;
    }

    // reset stats/non-fatal error reporting interval counters.
    mNonFatalErrorReportCnt = 0;
    mStatsReportCnt = 0;

    while( !ShouldStop() )
    {
        // set up fatal switch error polling
        uint32_t i = 0;
        for ( it = mSwitchIdToFdInfoMap.begin(); it != mSwitchIdToFdInfoMap.end(); it++ )
        {
            pfds[i].fd = it->second;
            pfds[i].events  = POLLPRI;
            pfds[i].revents = 0;
            i++;
        }

        rc = poll( pfds, nfds, ( FATAL_ERROR_POLLING_TIMEOUT * 1000 ));
        // poll returned, break if we have an outstanding request to exit the thread
        if ( ShouldStop() )
        {
            break;
        }
        // continue processing the fd events
        if ( rc > 0 )
        {
            uint32_t j = 0;
            for ( it = mSwitchIdToFdInfoMap.begin(); it != mSwitchIdToFdInfoMap.end(); it++ )
            {
                if ( pfds[j].revents  == POLLPRI )
                {
                    // report fatal errors
                    reportFatalErrors( it->first );
                }
                j++;
            }
        }

        mNonFatalErrorReportCnt++;
        if ( mNonFatalErrorReportCnt >= NON_FATAL_ERROR_REPORTING_INTERVAL_CNT )
        {
            // time to get non fatal errors
            reportNonFatalErrors( );
            mNonFatalErrorReportCnt = 0;
        }

        mStatsReportCnt++;
        if ( mStatsReportCnt >= STATS_REPORTING_INTERVAL_CNT )
        {
            // time to get stats
            reportStats( );
            mStatsReportCnt = 0;
        }

        // no need to monitor GPU lwlinks non shared fabric mode,
        // because the GPUs are not with service VM in shared fabric mode
        if ( !mpLfm->isSharedFabricMode() )
            checkForGpuLWLinkError();
    }

    free( pfds );
}

/*
 * Get switch errors from the switch interface
 *
 * Allocate SwitchError_structs on the errQ
 * The caller of the function should free the memory
 */
void
DcgmLocalStatsReporter::getSwitchErrors(uint32_t physicalId,
                                        uint32_t errorMask,
                                        std::queue < SwitchError_struct * > *errQ)
{
    DcgmSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );

    if ( !pInterface || !errQ )
    {
        PRINT_ERROR(" ", "Invalid LWSwitch driver interface or error message queue");
        return;
    }

    switchIoctl_t       ioctlStruct;
    LWSWITCH_GET_ERRORS ioctlParams;
    uint32_t i;

    ioctlStruct.type        = IOCTL_LWSWITCH_GET_ERRORS;
    ioctlStruct.ioctlParams = &ioctlParams;

    memset( &ioctlParams, 0, sizeof(LWSWITCH_GET_ERRORS) );
    ioctlParams.errorMask = errorMask;

    do {
        ioctlParams.errorCount = 0;
        if ( pInterface->doIoctl( &ioctlStruct ) != FM_SUCCESS )
        {
            return;
        }

        for ( i = 0; i < ioctlParams.errorCount; i++)
        {
            // LWLink Recovery errors are reported as non-fatal and requires
            // some treatment at GlobalFM. So report the additional data as
            // seperate recovery message. But continue reporting the switch error
            // to GlobalFM, so that the same will be published into cache manager
            if (ioctlParams.error[i].error_type == LWSWITCH_ERR_HW_DLPL_TX_RECOVERY_LONG)
            {
                reportSwitchLWLinkRecoveryError( physicalId, ioctlParams.error[i] );
            }
            SwitchError_struct *switchErr = new SwitchError_struct;
            switchErr->switchPhysicalId = physicalId;
            memcpy( &(switchErr->switchError), &(ioctlParams.error[i]), sizeof(LWSWITCH_ERROR) );
            errQ->push( switchErr );
        }

    } while (ioctlParams.errorCount != 0);
}

/*
 * build switch error message to global fabric manager
 * Caller need to free the FM message
 */
lwswitch::fmMessage*
DcgmLocalStatsReporter::buildSwitchErrorMsg(uint32_t errorMask,
                                            std::queue < SwitchError_struct * > *errQ)
{
    if ( !errQ )
        return NULL;

    PRINT_DEBUG("0x%0x, %d", "error mask 0x%0x, error count %d.",
                errorMask, (int)errQ->size());

    SwitchError_struct          *swErr;
    lwswitch::switchError       *lwSwithErr = NULL;
    lwswitch::switchErrorInfo   *info = NULL;

    lwswitch::switchErrorReport *report = new lwswitch::switchErrorReport();

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();

    if ( errorMask == LWSWITCH_GET_ERRORS_FATAL ) {
        pFmMessage->set_type( lwswitch::FM_LWSWITCH_ERROR_FATAL );
    } else {
        pFmMessage->set_type( lwswitch::FM_LWSWITCH_ERROR_NON_FATAL );
    }

    pFmMessage->set_allocated_errorreport( report );

    while ( !errQ->empty() ) {

        swErr = errQ->front();

        lwSwithErr = report->add_switcherror();
        lwSwithErr->set_switchphysicalid( swErr->switchPhysicalId );

        info = lwSwithErr->add_errorinfo();
        info->set_errortype( swErr->switchError.error_type );
        info->set_severity( (enum lwswitch::switchErrorSeverity)swErr->switchError.severity );
        info->set_errorsrc( (enum lwswitch::switchErrorSrc) swErr->switchError.error_src );
        info->set_instance( swErr->switchError.instance );
        info->set_subinstance( swErr->switchError.subinstance );
        info->set_time( swErr->switchError.time );
        info->set_resolved( (bool)swErr->switchError.error_resolved );

        errQ->pop();
        delete swErr;
    }

    return pFmMessage;
}

/*
 * Send fatal error synchronously to GFM
 */
void
DcgmLocalStatsReporter::reportFatalErrors(uint32_t physicalId)
{
    std::queue < SwitchError_struct * > errQ;
    uint32_t errorMask = LWSWITCH_GET_ERRORS_FATAL;
    lwswitch::fmMessage   *pFmMessage;
    lwswitch::fmMessage   *pFmResponse = NULL;
    dcgmReturn_t          ret;
    DcgmSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );

    if ( !pInterface )
    {
        PRINT_ERROR(" ", "Invalid LWSwitch driver interface");
        return;
    }

    getSwitchErrors( physicalId, errorMask, &errQ );
    if ( errQ.size() == 0 ) return;

    pFmMessage = buildSwitchErrorMsg( errorMask, &errQ );
    if ( !pFmMessage ) return;


    // send Fatal errors to global fabric manager
    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send fatal error message to globalFM, return %d.", ret);
        // TODO: retry sending fatal errors
    }

    if ( pFmMessage )  delete pFmMessage;
    if ( pFmResponse ) delete pFmResponse;
}

/*
 * Send non fatal error asynchronously to GFM
 */
void
DcgmLocalStatsReporter::reportNonFatalErrors(void)
{
    std::queue < SwitchError_struct * > errQ;
    uint32_t errorMask = LWSWITCH_GET_ERRORS_ERROR;
    lwswitch::fmMessage   *pFmMessage;
    dcgmReturn_t          ret;
    DcgmLWSwitchPhyIdToFdInfoMap::iterator it;

    for ( it = mSwitchIdToFdInfoMap.begin(); it != mSwitchIdToFdInfoMap.end(); it++ )
    {
        getSwitchErrors(it->first, errorMask, &errQ );
    }

    if ( errQ.size() == 0 ) return;

    pFmMessage = buildSwitchErrorMsg( errorMask, &errQ );
    if ( !pFmMessage ) return;

    // send non fatal errors to global fabric manager
    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send non fatal error message to globalFM, return %d.", ret);
    }

    if ( pFmMessage )  delete pFmMessage;
}

/*
 * Get switch latency from the switch interface
 *
 * Allocate SwitchLatency_struct on the latencyQ
 * The calling function should free the memory
 */
void
DcgmLocalStatsReporter::getSwitchInternalLatency(uint32_t physicalId,
                                                 std::queue < SwitchLatency_struct * > *latencyQ)
{
    DcgmSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );
    if ( !pInterface || !latencyQ )
    {
        PRINT_ERROR(" ", "Invalid LWSwitch driver interface or latency message queue");
        return;
    }

    switchIoctl_t ioctlStruct;
    SwitchLatency_struct *latency = new SwitchLatency_struct;

    memset(&ioctlStruct, 0, sizeof(ioctlStruct));
    memset(latency, 0, sizeof(*latency));

    latency->latencies.vc_selector = 0;                         //$$$PJS TODO debate if we want the sum of 0 and 5  

    ioctlStruct.type = IOCTL_LWSWITCH_GET_INTERNAL_LATENCY;
    ioctlStruct.ioctlParams = &(latency->latencies);
    latency->switchPhysicalId = physicalId;

    if ( pInterface->doIoctl( &ioctlStruct ) != FM_SUCCESS )
    {
        delete latency;
        return;
    }

    latencyQ->push( latency );
}

/*
 * Get switch lwlink counter from the switch interface
 *
 * Allocate LwlinkCounter_struct on the counterQ
 * The calling function should free the memory
 */
void
DcgmLocalStatsReporter::getSwitchLwlinkCounter(uint32_t physicalId,
                                               std::queue < LwlinkCounter_struct * > *counterQ)
{
    DcgmSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );
    if ( !pInterface || !counterQ )
    {
        PRINT_ERROR(" ", "Invalid LWSwitch driver interface or counter message queue");
        return;
    }

    switchIoctl_t ioctlStruct;
    LwlinkCounter_struct *counter = new LwlinkCounter_struct;

    memset(&ioctlStruct, 0, sizeof(ioctlStruct));
    ioctlStruct.type = IOCTL_LWSWITCH_GET_LWLIPT_COUNTERS;
    ioctlStruct.ioctlParams = &(counter->counters);
    counter->switchPhysicalId = physicalId;

    if ( pInterface->doIoctl( &ioctlStruct ) != FM_SUCCESS )
    {
        delete counter;
        return;
    }

    counterQ->push( counter );
}

/*
 * build switch stats to global fabric manager
 *
 * Caller need to free the FM message
 */
lwswitch::fmMessage*
DcgmLocalStatsReporter::buildSwitchStatsMsg(std::queue < SwitchLatency_struct * > *latencyQ,
                                            std::queue < LwlinkCounter_struct * > *counterQ)
{
    uint32_t port;

    SwitchLatency_struct        *swLatencyStruct;
    lwswitch::switchLatencyHist *swLatency = NULL;
    lwswitch::portLatencyHist   *portLatency = NULL;

    LwlinkCounter_struct          *swCounterStruct;
    lwswitch::switchLwlinkCounter *swCounter = NULL;
    lwswitch::lwlinkCounter       *linkCounter = NULL;

    lwswitch::nodeStats *report = new lwswitch::nodeStats();

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_type( lwswitch::FM_NODE_STATS_REPORT );

    pFmMessage->set_allocated_statsreport( report );

    while ( latencyQ && ( !latencyQ->empty() ) ) {

        swLatencyStruct = latencyQ->front();
        // get the number of ports and port mask from specified LWSwitch device
        DcgmSwitchInterface *pSwitchInterface = mpLfm->switchInterfaceAt( swLatencyStruct->switchPhysicalId );
        if ( !pSwitchInterface )
        {
            PRINT_ERROR(" ", "Invalid LWSwitch driver interface while building latency stats message");
            break;
        }
        uint32_t numPorts = pSwitchInterface->getNumPorts();                
        uint64_t enabledPortMask = pSwitchInterface->getEnabledPortMask();

        swLatency = report->add_latestlatency();
        swLatency->set_switchphysicalid( swLatencyStruct->switchPhysicalId );

        for ( port = 0; port < numPorts; port++ )
        {
            if ( !(enabledPortMask & ((uint64_t)1 << port)) )
                continue;

            portLatency = swLatency->add_latencyhist();
            portLatency->set_portnum( port );
            portLatency->set_elapsedtimemsec( swLatencyStruct->latencies.elapsed_time_msec );
            portLatency->set_low( swLatencyStruct->latencies.egressHistogram[port].low );
            portLatency->set_med( swLatencyStruct->latencies.egressHistogram[port].medium );
            portLatency->set_high( swLatencyStruct->latencies.egressHistogram[port].high );
            portLatency->set_panic( swLatencyStruct->latencies.egressHistogram[port].panic );
        }

        latencyQ->pop();
        delete swLatencyStruct;
    }

    while ( counterQ && ( !counterQ->empty() ) ) {

        swCounterStruct = counterQ->front();
        // get the number of ports and port mask from specified LWSwitch device
        DcgmSwitchInterface *pSwitchInterface = mpLfm->switchInterfaceAt( swCounterStruct->switchPhysicalId );
        if ( !pSwitchInterface )
        {
            PRINT_ERROR(" ", "Invalid LWSwitch driver interface while building counter stats message");
            break;
        }
        uint32_t numPorts = pSwitchInterface->getNumPorts();                
        uint64_t enabledPortMask = pSwitchInterface->getEnabledPortMask();

        swCounter = report->add_lwlinkcounter();
        swCounter->set_switchphysicalid( swCounterStruct->switchPhysicalId );

        for ( port = 0; port < numPorts; port ++ )
        {
            if ( !(enabledPortMask & ((uint64_t)1 << port)) )
                continue;

            linkCounter = swCounter->add_linkcounter();
            linkCounter->set_portnum( port );
            linkCounter->set_txcounter0( swCounterStruct->counters.liptCounter[port].txCounter0 );
            linkCounter->set_rxcounter0( swCounterStruct->counters.liptCounter[port].rxCounter0 );
            linkCounter->set_txcounter1( swCounterStruct->counters.liptCounter[port].txCounter1 );
            linkCounter->set_rxcounter1( swCounterStruct->counters.liptCounter[port].rxCounter1 );
        }

        counterQ->pop();
        delete swCounterStruct;
    }

    return pFmMessage;
}

void
DcgmLocalStatsReporter::reportStats(void)
{
    std::queue < SwitchLatency_struct * > latencyQ;
    std::queue < LwlinkCounter_struct * > counterQ;
    lwswitch::fmMessage   *pFmMessage;
    dcgmReturn_t          ret;
    DcgmLWSwitchPhyIdToFdInfoMap::iterator it;

    for ( it = mSwitchIdToFdInfoMap.begin(); it != mSwitchIdToFdInfoMap.end(); it++ )
    {
        getSwitchInternalLatency(it->first, &latencyQ );
        getSwitchLwlinkCounter(it->first, &counterQ );
    }

    // send stats to global fabric manager
    pFmMessage = buildSwitchStatsMsg( &latencyQ, &counterQ );
    if ( !pFmMessage ) return;

    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send stats report message to globalFM, return %d.", ret);
    }

    if ( pFmMessage )  delete pFmMessage;
}

void
DcgmLocalStatsReporter::checkForGpuLWLinkError(void)
{
    dcgmReturn_t retVal;
    // iterate all the GPUs for LWLink error
    GpuIDToDcgmSampleMap::iterator it = mGpuLWLinkErrSample.begin();
    for ( ; it != mGpuLWLinkErrSample.end(); it++ )
    {
        dcgmcm_sample_t lwrrentSample = {0};
        unsigned int gpuIndex = it->first;
        retVal = LwcmHostEngineHandler::Instance()->GetLatestSample( DCGM_FE_GPU, gpuIndex,
                                                           DCGM_FI_DEV_GPU_LWLINK_ERRORS, &lwrrentSample );
        if ( retVal == DCGM_ST_OK )
        {
            // we received a sample for LWLink error. Check the timestamp of this event.
            dcgmcm_sample_t oldSample = it->second;
            if ( lwrrentSample.timestamp != oldSample.timestamp )
            {
                // time stamp is different, new event from RM
                // replace our cached sample with this latest one
                mGpuLWLinkErrSample[gpuIndex] = lwrrentSample;
                // check what kind of LWLink error it is and process accordingly
                processGpuLWLinkError( gpuIndex, lwrrentSample);
            }
        }
    }
}

void
DcgmLocalStatsReporter::processGpuLWLinkError(unsigned int gpuIndex, dcgmcm_sample_t &lwrrentSample)
{
    switch( lwrrentSample.val.i64 )
    {
        case DCGM_GPU_LWLINK_ERROR_FATAL:
        {
            reportGpuLWLinkFatalError( gpuIndex );
            break;
        }
        case DCGM_GPU_LWLINK_ERROR_RECOVERY_REQUIRED:
        {
            reportGpuLWLinkRecoveryError( gpuIndex );
            break;
        }
        default:
        {
            PRINT_ERROR( "", "Unknown GPU LWLink error sample received from CacheManager" );
            break;
        }
    }
}

void
DcgmLocalStatsReporter::reportGpuLWLinkRecoveryError(unsigned int gpuIndex)
{
    int ret;
    lwmlReturn_t lwmlResult;
    lwmlDevice_t lwmlDevice;

    // report the device and link id information to GFM. Based on the
    // that, GFM will retrain the specific connection

    // GPU don't report which link failed and moved to SWCFG. 
    // query RM for all the link status for this device and find the link in SWCFG.
    lwmlResult = lwmlDeviceGetHandleByIndex( gpuIndex, &lwmlDevice );
    if ( LWML_SUCCESS != lwmlResult )
    {
        PRINT_ERROR( "%d %d", "lwmlDeviceGetHandleByIndex failed for index %d with %d", 
                    gpuIndex, lwmlResult );
        return;
    }

    // LWML uses gpuIndex and LWLinkDevRepo uses PCI information.
    lwmlPciInfo_t lwmlGpuPciInfo;
    lwmlResult = lwmlDeviceGetPciInfo( lwmlDevice, &lwmlGpuPciInfo );
    if ( LWML_SUCCESS != lwmlResult )
    {
        PRINT_ERROR( "%d %u", "lwmlDeviceGetPciInfo returned %d for lwmlIndex %u",
                    (int)lwmlResult, gpuIndex );
        return;
    }

    // fill the recovery error message details
    lwswitch::lwlinkErrorRecoveryMsg *recoveryMsg = new lwswitch::lwlinkErrorRecoveryMsg();

    // get the device's lwlink link status information
    for ( int linkIdx = 0; linkIdx < LWML_LWLINK_MAX_LINKS; linkIdx++ )
    {
        lwmlEnableState_t isActive = LWML_FEATURE_DISABLED;
        lwmlResult = lwmlDeviceGetLwLinkState( lwmlDevice, linkIdx, &isActive );
        if ( LWML_SUCCESS != lwmlResult )
        {
            PRINT_ERROR( "%d %d", "lwmlDeviceGetLwLinkState failed for link index %d with %d", 
                         linkIdx, lwmlResult );
            continue;
        }
        // check for links in SWCFG, which can be trained to Active.
        if ( LWML_FEATURE_DISABLED == isActive)
        {
            //
            // Note: This LWML API will return the link state as LWML_FEATURE_DISABLED for disabled
            // links and links which are not in HIGH SPEED (i.e. ACTIVE). When a recovery error happens,
            // the link will fall back to SAFE (i.e. SWCFG) mode. So there is no way to distinguish between
            // links which are disabled or in SAFE mode. However, GlobalFM will not have a connection
            // entry for disabled links and there for will not attempt a re-train for those links.
            //
            recoveryMsg->add_linkindex( linkIdx );
        }
    }

    // populate the device information
    uint64 deviceId = 0;
    lwlink_pci_dev_info pciInfo;
    pciInfo.domain = lwmlGpuPciInfo.domain;
    pciInfo.bus = lwmlGpuPciInfo.bus;
    pciInfo.device = lwmlGpuPciInfo.device;
    pciInfo.function = 0; // not filled by LWML now.
    deviceId = mLWLinkDevRepo->getDeviceId( pciInfo );
    recoveryMsg->set_gpuorswitchid( deviceId );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
    errorMsg->set_allocated_recoverymsg( recoveryMsg );

    // fill the outer FM Message
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );

    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send GPU LWLink recovery error message to globalFM, return %d.", ret);
    }

    delete pFmMessage;
}

void
DcgmLocalStatsReporter::reportGpuLWLinkFatalError(unsigned int gpuIndex)
{
    int ret;
    lwmlReturn_t lwmlResult;
    lwmlDevice_t lwmlDevice;

    // query RM for all the link status for this device
    lwmlResult = lwmlDeviceGetHandleByIndex( gpuIndex, &lwmlDevice );
    if ( LWML_SUCCESS != lwmlResult )
    {
         PRINT_ERROR( "%d %d", "lwmlDeviceGetHandleByIndex failed for index %d with %d", 
                     gpuIndex, lwmlResult );
         return;
    } 

    // LWML uses gpuIndex and LWLinkDevRepo uses PCI information.
    lwmlPciInfo_t lwmlGpuPciInfo;
    lwmlResult = lwmlDeviceGetPciInfo(lwmlDevice, &lwmlGpuPciInfo);
    if(lwmlResult != LWML_SUCCESS)
    {
        PRINT_ERROR("%d %u", "lwmlDeviceGetPciInfo returned %d for lwmlIndex %u",
                    (int)lwmlResult, gpuIndex);
        return;
    }

    lwlink_pci_dev_info lwLinkPciInfo;
    uint64 deviceId = 0;    
    lwLinkPciInfo.domain = lwmlGpuPciInfo.domain;
    lwLinkPciInfo.bus = lwmlGpuPciInfo.bus;
    lwLinkPciInfo.device = lwmlGpuPciInfo.device;
    //TODO: fill PCI function information. DCGM don't have this info available straightforward
    lwLinkPciInfo.function = 0;
    deviceId = mLWLinkDevRepo->getDeviceId( lwLinkPciInfo );

    // fill the lwlink fatal error message and device information
    lwswitch::lwlinkErrorGpuFatalMsg *fatalMsg = new lwswitch::lwlinkErrorGpuFatalMsg();
    fatalMsg->set_gpuorswitchid( deviceId );

    // fill the lwlink error message    
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
    errorMsg->set_allocated_gpufatalmsg( fatalMsg );

    // fill the outer FM Message
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_GPU_FATAL );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );

    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send GPU LWLink fatal error message to globalFM, return %d.", ret);
    }

    delete pFmMessage;
}

void
DcgmLocalStatsReporter::reportSwitchLWLinkRecoveryError(uint32_t physicalId,
                                                        LWSWITCH_ERROR &switchError)
{
    int ret;

    // report the device and link id information to GFM. Based on the
    // that, GFM will retrain the specific connection

    // get the device id information using PCI information
    DcgmSwitchInterface *pInterface = mpLfm->switchInterfaceAt( physicalId );
    if ( !pInterface )
    {
        PRINT_ERROR(" ", "Invalid LWSwitch driver interface");
        return;
    }

    DcgmFMPciInfo switchPciInfo = pInterface->getSwtichPciInfo();
    lwlink_pci_dev_info pciInfo;
    pciInfo.domain = switchPciInfo.domain;
    pciInfo.bus = switchPciInfo.bus;
    pciInfo.device = switchPciInfo.device;
    pciInfo.function = switchPciInfo.function;
    uint64 deviceId = mLWLinkDevRepo->getDeviceId( pciInfo );

    // fill the recovery error message details
    lwswitch::lwlinkErrorRecoveryMsg *recoveryMsg = new lwswitch::lwlinkErrorRecoveryMsg();
    recoveryMsg->set_gpuorswitchid( deviceId );
    recoveryMsg->add_linkindex ( switchError.instance );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkErrorMsg *errorMsg = new lwswitch::lwlinkErrorMsg();
    errorMsg->set_allocated_recoverymsg( recoveryMsg );

    // fill the outer FM Message
    pFmMessage->set_type( lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY );
    pFmMessage->set_allocated_lwlinkerrormsg( errorMsg );

    ret = mpLfm->SendMessageToGfm( pFmMessage, true );
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d", "Failed to send LWSwitch LWLink recovery error message to globalFM, return %d.", ret);
    }

    delete pFmMessage;
}
