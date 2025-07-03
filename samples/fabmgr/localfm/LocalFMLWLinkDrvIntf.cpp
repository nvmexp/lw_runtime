/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "fm_log.h"
#include "lwlink_errors.h"
#include "lwlink_lib_ioctl.h"
#include "FMLWLinkError.h"
#include "g_lwconfig.h"
#include "LocalFMLWLinkDrvIntf.h"

LocalFMLWLinkDrvIntf::LocalFMLWLinkDrvIntf()
{
    LW_STATUS retVal;

    // initialize the LWLinkCoreLib driver interface through the LWLink shim layer
    retVal = lwlink_api_init();
    if (retVal != LW_OK) {
        std::ostringstream ss;
        ss << "request to initialize LWLink driver API interface failed with error:" << lwstatusToString(retVal);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // open the LWLinkCoreLib driver interface
    mpLWLinkDrvSession = NULL;
    retVal = lwlink_api_create_session(&mpLWLinkDrvSession);
    if (retVal != LW_OK) {
        std::ostringstream ss;
        ss << "request to open LWLink driver handle failed with error:" << lwstatusToString(retVal);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    //
    // before issuing any IOCTL to the driver, acquire our fabric management capability which indicate
    // fabric manager has enough permission to issue privileged IOCTLs to LWLinkCoreLib Driver.
    // This is required to run FM from root and non-root context.
    //

    // this function will throw exception if we fail to acquire the required capability
    acquireFabricManagementCapability();
}

LocalFMLWLinkDrvIntf::~LocalFMLWLinkDrvIntf()
{
    // close our driver handle
    if (mpLWLinkDrvSession) {
        lwlink_api_free_session(&mpLWLinkDrvSession);
        mpLWLinkDrvSession = NULL;
    }
}

int
LocalFMLWLinkDrvIntf::doIoctl(int ioctlCmd, void *ioctlParam, int paramSize)
{
    LW_STATUS retVal;

    // validate input param and driver file handle
    if ((ioctlParam == NULL) || (mpLWLinkDrvSession == NULL)) {
        return -LWL_BAD_ARGS;
    }

    // check for ioctl command validity
    if (!isIoctlCmdSupported(ioctlCmd)) {
        FM_LOG_ERROR("LWLink driver ioctl command 0x%x is not supported", ioctlCmd);
        return -LWL_BAD_ARGS;
    }

    // issue the ioctl to LWLink driver
    //FM_LOG_DEBUG("start cmd=0x%x", ioctlCmd);
    retVal = lwlink_api_control(mpLWLinkDrvSession, ioctlCmd, ioctlParam, paramSize);
    //FM_LOG_DEBUG("end cmd=0x%x", ioctlCmd);
    if (retVal != LW_OK) {
        // this failure is from kernel or due to some bad params.
        // the actual ioctl specific error is reported in the individual ioctl params.
        FM_LOG_ERROR( "LWLink driver ioctl command 0x%x failed with error %s", ioctlCmd, lwstatusToString(retVal) );
        return retVal;
    }

    // the status member in each ioctl type indicate the driver
    // returned final status. check the same and log any errors
    logIoctlError(ioctlCmd, ioctlParam);
    return retVal;
}

int
LocalFMLWLinkDrvIntf::isIoctlCmdSupported(int ioctlCmd)
{
    switch(ioctlCmd) {
        case IOCTL_LWLINK_CHECK_VERSION:
        case IOCTL_LWLINK_SET_NODE_ID:
        case IOCTL_LWLINK_SET_TX_COMMON_MODE:
        case IOCTL_LWLINK_OPTICAL_INIT_LINKS:
        case IOCTL_LWLINK_OPTICAL_ENABLE_IOBIST:
        case IOCTL_LWLINK_OPTICAL_START_PRETRAIN:
        case IOCTL_LWLINK_OPTICAL_STOP_PRETRAIN:
        case IOCTL_LWLINK_OPTICAL_CHECK_PRETRAIN:
        case IOCTL_LWLINK_OPTICAL_DISABLE_IOBIST:
        case IOCTL_LWLINK_INITPHASE1:
        case IOCTL_LWLINK_INITNEGOTIATE:
        case IOCTL_LWLINK_RX_INIT_TERM:
        case IOCTL_LWLINK_SET_RX_DETECT:
        case IOCTL_LWLINK_GET_RX_DETECT:
        case IOCTL_LWLINK_DEVICE_READ_SIDS:
        case IOCTL_LWLINK_CALIBRATE:
        case IOCTL_LWLINK_ENABLE_DATA:
        case IOCTL_LWLINK_INITPHASE5:
        case IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC:
        case IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS:
        case IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS:
        case IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_ADD_INTERNODE_CONN:
        case IOCTL_LWLINK_REMOVE_INTERNODE_CONN:
#endif
        case IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS:
        case IOCTL_LWLINK_READ_DISCOVERY_TOKENS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE:
        case IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE:
#endif
        case IOCTL_LWLINK_OPTICAL_ENABLE_INFINITE_MODE:
        case IOCTL_LWLINK_OPTICAL_ENABLE_MAINTENANCE:
        case IOCTL_LWLINK_OPTICAL_DISABLE_INFINITE_MODE:
        case IOCTL_LWLINK_OPTICAL_ENABLE_FORCE_EQ:
        case IOCTL_LWLINK_OPTICAL_DISABLE_FORCE_EQ:
        case IOCTL_LWLINK_OPTICAL_CHECK_EOM_STATUS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL:
        case IOCTL_LWLINK_GET_LINK_STATE:
#endif
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONN:
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK:
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK:
#endif
        case IOCTL_LWLINK_GET_DEVICE_LINK_STATES:
        case IOCTL_LWLINK_GET_DEVICES_INFO: {
            // valid type
            return true;
        }
    }

    // default case, not supported ioctl type
    return false;
}

void
LocalFMLWLinkDrvIntf::logIoctlError(int ioctlCmd, void *ioctlParam)
{
    LwlStatus ioctlStatus = LWL_SUCCESS;
    const char *ioctlStr;

    // find the return status based on ioctl command
    switch(ioctlCmd) {
        case IOCTL_LWLINK_CHECK_VERSION: {
            lwlink_check_version *iocReq = (lwlink_check_version*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_CHECK_VERSION";
            break;
        }
        case IOCTL_LWLINK_SET_NODE_ID: {
            lwlink_set_node_id *iocReq = (lwlink_set_node_id*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_SET_NODE_ID";
            break;
        }
        case IOCTL_LWLINK_SET_TX_COMMON_MODE: {
            lwlink_set_tx_common_mode *iocReq = (lwlink_set_tx_common_mode*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_SET_TX_COMMON_MODE";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_INIT_LINKS: {
            lwlink_optical_init_links *iocReq = (lwlink_optical_init_links*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_INIT_LINKS";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_ENABLE_IOBIST: {
            lwlink_optical_set_iobist *iocReq = (lwlink_optical_set_iobist*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_ENABLE_IOBIST";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_START_PRETRAIN: {
            lwlink_optical_set_pretrain *iocReq = (lwlink_optical_set_pretrain*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_START_PRETRAIN";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_STOP_PRETRAIN: {
            lwlink_optical_set_pretrain *iocReq = (lwlink_optical_set_pretrain*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_STOP_PRETRAIN";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_CHECK_PRETRAIN: {
            lwlink_optical_check_pretrain *iocReq = (lwlink_optical_check_pretrain*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_CHECK_PRETRAIN";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_DISABLE_IOBIST: {
            lwlink_optical_set_iobist *iocReq = (lwlink_optical_set_iobist*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_DISABLE_IOBIST";
            break;
        }
        case IOCTL_LWLINK_INITPHASE1: {
            lwlink_initphase1 *iocReq = (lwlink_initphase1*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_INITPHASE1";
            break;
        }
        case IOCTL_LWLINK_INITNEGOTIATE: {
            lwlink_initnegotiate *iocReq = (lwlink_initnegotiate*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_INITNEGOTIATE";
            break;
        }
        case IOCTL_LWLINK_DEVICE_READ_SIDS: {
            lwlink_device_read_sids *iocReq = (lwlink_device_read_sids*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_DEVICE_READ_SIDS";
            break;
        }
        case IOCTL_LWLINK_RX_INIT_TERM: {
            lwlink_rx_init_term *iocReq = (lwlink_rx_init_term*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_RX_INIT_TERM";
            break;
        }
        case IOCTL_LWLINK_SET_RX_DETECT: {
            lwlink_set_rx_detect *iocReq = (lwlink_set_rx_detect*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_SET_RX_DETECT";
            break;
        }
        case IOCTL_LWLINK_GET_RX_DETECT: {
            lwlink_get_rx_detect *iocReq = (lwlink_get_rx_detect*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_GET_RX_DETECT";
            break;
        }
        case IOCTL_LWLINK_CALIBRATE: {
            lwlink_calibrate *iocReq = (lwlink_calibrate*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_CALIBRATE";
            break;
        }
        case IOCTL_LWLINK_ENABLE_DATA: {
            lwlink_enable_data *iocReq = (lwlink_enable_data*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_ENABLE_DATA";
            break;
        }
        case IOCTL_LWLINK_INITPHASE5: {
            lwlink_initphase5 *iocReq = (lwlink_initphase5*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_INITPHASE5";
            break;
        }
        case IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC: {
            lwlink_link_init_async *iocReq = (lwlink_link_init_async*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC";
            break;
        }
        case IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS: {
            lwlink_device_link_init_status *iocReq = (lwlink_device_link_init_status*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS";
            break;
        }
        case IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS: {
            lwlink_discover_intranode_conns *iocReq = (lwlink_discover_intranode_conns*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS";
            break;
        }
        case IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS: {
            lwlink_device_get_intranode_conns *iocReq = (lwlink_device_get_intranode_conns*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS";
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_ADD_INTERNODE_CONN: {
            lwlink_add_internode_conn *iocReq = (lwlink_add_internode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_ADD_INTERNODE_CONN";
            break;
        }
        case IOCTL_LWLINK_REMOVE_INTERNODE_CONN: {
            lwlink_remove_internode_conn *iocReq = (lwlink_remove_internode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_REMOVE_INTERNODE_CONN";
            break;
        }
#endif
        case IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS: {
            lwlink_device_write_discovery_tokens *iocReq = (lwlink_device_write_discovery_tokens*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS";
            break;
        }
        case IOCTL_LWLINK_READ_DISCOVERY_TOKENS: {
            lwlink_device_read_discovery_tokens *iocReq = (lwlink_device_read_discovery_tokens*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_READ_DISCOVERY_TOKENS";
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONN: {
            lwlink_train_intranode_conn *iocReq = (lwlink_train_intranode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTRANODE_CONN";
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL: {
            lwlink_train_intranode_conns_parallel *iocReq = (lwlink_train_intranode_conns_parallel *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL";
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE: {
            lwlink_train_internode_links_initoptimize *iocReq = (lwlink_train_internode_links_initoptimize *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE";
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE: {
            lwlink_train_internode_links_post_initoptimize *iocReq = (lwlink_train_internode_links_post_initoptimize *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE";
            break;
        }
#endif
        case IOCTL_LWLINK_OPTICAL_ENABLE_INFINITE_MODE: {
            lwlink_optical_set_infinite_mode *iocReq = (lwlink_optical_set_infinite_mode *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_ENABLE_INFINITE_MODE";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_ENABLE_MAINTENANCE: {
            lwlink_optical_enable_maintenance *iocReq = (lwlink_optical_enable_maintenance *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_ENABLE_MAINTENANCE";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_DISABLE_INFINITE_MODE: {
            lwlink_optical_set_infinite_mode *iocReq = (lwlink_optical_set_infinite_mode *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_DISABLE_INFINITE_MODE";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_ENABLE_FORCE_EQ: {
            lwlink_optical_set_force_eq *iocReq = (lwlink_optical_set_force_eq *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_ENABLE_FORCE_EQ";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_DISABLE_FORCE_EQ: {
            lwlink_optical_set_force_eq *iocReq = (lwlink_optical_set_force_eq *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_DISABLE_FORCE_EQ";
            break;
        }
        case IOCTL_LWLINK_OPTICAL_CHECK_EOM_STATUS: {
            lwlink_optical_check_eom_status *iocReq = (lwlink_optical_check_eom_status *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_OPTICAL_CHECK_EOM_STATUS";
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL: {
            lwlink_train_internode_conns_parallel *iocReq = (lwlink_train_internode_conns_parallel *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL";
            break;
        }
        case IOCTL_LWLINK_GET_LINK_STATE: {
            lwlink_get_link_state *iocReq = (lwlink_get_link_state *)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_GET_LINK_STATE";
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK: {
            lwlink_train_internode_conn_link *iocReq = (lwlink_train_internode_conn_link*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK";
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK: {
            lwlink_train_internode_conn_sublink *iocReq = (lwlink_train_internode_conn_sublink*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK";
            break;
        }
#endif
        case IOCTL_LWLINK_GET_DEVICES_INFO: {
            lwlink_get_devices_info *iocReq = (lwlink_get_devices_info*)ioctlParam;
            ioctlStatus = iocReq->status;
            ioctlStr = "IOCTL_LWLINK_GET_DEVICES_INFO";
            break;
        }
        case IOCTL_LWLINK_GET_DEVICE_LINK_STATES: {
            lwlink_get_device_link_states *iocReq = (lwlink_get_device_link_states*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
    }

    // check for error case
    if (ioctlStatus != LWL_SUCCESS) {
        LWLinkErrorCodes errorCode = (LWLinkErrorCodes)FMLWLinkError::getLinkErrorCode(ioctlStatus);
        FM_LOG_ERROR("LWLink driver ioctl command 0x%x failed with internal status: %s",
                     ioctlCmd, FMLWLinkError::getLinkErrorString(errorCode));
    }
    //FM_LOG_DEBUG("LWLink driver ioctl command 0x%x (%s) called with internal status: %d",
    //             ioctlCmd, ioctlStr, ioctlStatus);
}

void
LocalFMLWLinkDrvIntf::acquireFabricManagementCapability(void)
{
    LW_STATUS retVal;

    //
    // by default, the LWLinkCoreLib device node (/dev/lwpu-lwlink in Linux ) has permission
    // to all users. All the privileged access is controlled through special fabric management node.
    // The actual management node dependents on driver mechanism. For devfs based support, it will be
    // /dev/lwpu-caps/lwpu-capX and for procfs based, it will be /proc/driver/lwpu-lwlink/capabilities/fabric-mgmt
    // This entry is created by driver and default access is for root/admin. The system administrator then must
    // change access to desired user. The below API is verifying whether FM has access to the path, if so open it
    // and associate/link the corresponding file descriptor with the file descriptor associated with
    // LWLinkCoreLib device node file descriptor (ie fd of /dev/lwpu-lwlink)
    //

    retVal = lwlink_api_session_acquire_capability(mpLWLinkDrvSession, LWLINK_CAP_FABRIC_MANAGEMENT);
    if ( retVal != LW_OK ) {
        // failed to get capability. throw error based on common return values
        switch (retVal) {
            case LW_ERR_INSUFFICIENT_PERMISSIONS: {
                std::ostringstream ss;
                ss << "failed to acquire required privileges to access LWLink driver." <<
                      " make sure fabric manager has access permissions to required device node files";
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
                break;
            }
            case LW_ERR_NOT_SUPPORTED: {
                //
                // driver doesn't have fabric management capability support on Windows for now and will 
                // return LW_ERR_NOT_SUPPORTED. So, treat this as not an error for now to let Windows FM to
                // continue. In Linux, the assumption is that Driver will not return LW_ERR_NOT_SUPPORTED
                // and even if, FM will eventually fail as privileged control calls will start erroring out.
                //
                break;
            }
            default: {
                std::ostringstream ss;
                ss << "request to acquire required privileges to access LWLink driver failed with error:" << lwstatusToString(retVal);
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }

    // successfully acquired required fabric management capability
}

