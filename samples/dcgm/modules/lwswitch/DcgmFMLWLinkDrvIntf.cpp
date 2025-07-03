
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "logging.h"
#include "lwlink_errors.h"
#include "lwlink_lib_ioctl.h"
#include "DcgmFMLWLinkError.h"
#include <g_lwconfig.h>

extern "C"
{
    #include "lwpu-modprobe-utils.h"
}

#include "DcgmFMLWLinkDrvIntf.h"

DcgmFMLWLinkDrvIntf::DcgmFMLWLinkDrvIntf()
{
    // attempt to create the device node before opening. if the device file already
    // exists with correct properties, the call will return success
    if (!lwidia_lwlink_mknod()) {
        PRINT_ERROR("", "failed to create LWLinkCoreLib driver device node file");
        throw std::runtime_error("failed to create LWLinkCoreLib driver device node file");
    }

    mDrvHandle = open(LW_LWLINK_DEVICE_NAME, O_RDWR);
    if (mDrvHandle < 0) {
        int errNum = errno;
        std::ostringstream ss;
        ss << "failed to open LWLinkCoreLib driver " << LW_LWLINK_DEVICE_NAME << "error: " << errNum;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }
}

DcgmFMLWLinkDrvIntf::~DcgmFMLWLinkDrvIntf()
{
    close(mDrvHandle);
}

int
DcgmFMLWLinkDrvIntf::doIoctl(int ioctlCmd, void *ioctlParam)
{
    int status;

    // validate input param and driver file handle
    if ((ioctlParam == NULL) || (mDrvHandle < 0)) {
        return -LWL_BAD_ARGS;
    }

    // check for ioctl command validity
    if (!isIoctlCmdSupported(ioctlCmd)) {
        PRINT_ERROR("%x", "LWLinkCoreLib driver ioctl type %x is not supported", ioctlCmd);
        return -LWL_BAD_ARGS;
    }

    status = ioctl(mDrvHandle, ioctlCmd, ioctlParam);
    if (status < 0) {
        // this failure is from kernel or due to some bad params.
        // the actual ioctl specific error is reported in the individual ioctl params.
        PRINT_ERROR( "%d", "LWLinkCoreLib driver ioctl failed with error = %d\n", status );
        return status;
    }

    // the status member in each ioctl type indicate the driver
    // returned final status. check the same and log any errors
    logIoctlError(ioctlCmd, ioctlParam);
    return status;
}

int
DcgmFMLWLinkDrvIntf::isIoctlCmdSupported(int ioctlCmd)
{
    switch(ioctlCmd) {
        case IOCTL_LWLINK_CHECK_VERSION:
        case IOCTL_LWLINK_SET_NODE_ID:
        case IOCTL_LWLINK_SET_TX_COMMON_MODE:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case IOCTL_LWLINK_INITPHASE1:
        case IOCTL_LWLINK_INITNEGOTIATE:
        case IOCTL_LWLINK_RX_INIT_TERM:
        case IOCTL_LWLINK_SET_RX_DETECT:
        case IOCTL_LWLINK_GET_RX_DETECT:
#endif
        case IOCTL_LWLINK_CALIBRATE:
        case IOCTL_LWLINK_ENABLE_DATA:
        case IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC:
        case IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS:
        case IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS:
        case IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS:
        case IOCTL_LWLINK_ADD_INTERNODE_CONN:
        case IOCTL_LWLINK_REMOVE_INTERNODE_CONN:
        case IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS:
        case IOCTL_LWLINK_READ_DISCOVERY_TOKENS:
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONN:
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK:
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK:
        case IOCTL_LWLINK_GET_DEVICES_INFO: {
            // valid type
            return true;
        }
    }

    //default case, not supported ioctl type
    return false;
}

void
DcgmFMLWLinkDrvIntf::logIoctlError(int ioctlCmd, void *ioctlParam)
{
    LwlStatus ioctlStatus = LWL_SUCCESS;

    // find the return status based on ioctl command
    switch(ioctlCmd) {
        case IOCTL_LWLINK_CHECK_VERSION: {
            lwlink_check_version *iocReq = (lwlink_check_version*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_SET_NODE_ID: {
            lwlink_set_node_id *iocReq = (lwlink_set_node_id*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_SET_TX_COMMON_MODE: {
            lwlink_set_tx_common_mode *iocReq = (lwlink_set_tx_common_mode*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case IOCTL_LWLINK_INITPHASE1: {
            lwlink_initphase1 *iocReq = (lwlink_initphase1*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_INITNEGOTIATE: {
            lwlink_initnegotiate *iocReq = (lwlink_initnegotiate*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_RX_INIT_TERM: {
            lwlink_rx_init_term *iocReq = (lwlink_rx_init_term*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_SET_RX_DETECT: {
            lwlink_set_rx_detect *iocReq = (lwlink_set_rx_detect*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_GET_RX_DETECT: {
            lwlink_get_rx_detect *iocReq = (lwlink_get_rx_detect*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
#endif
        case IOCTL_LWLINK_CALIBRATE: {
            lwlink_calibrate *iocReq = (lwlink_calibrate*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_ENABLE_DATA: {
            lwlink_enable_data *iocReq = (lwlink_enable_data*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC: {
            lwlink_link_init_async *iocReq = (lwlink_link_init_async*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS: {
            lwlink_device_link_init_status *iocReq = (lwlink_device_link_init_status*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS: {
            lwlink_discover_intranode_conns *iocReq = (lwlink_discover_intranode_conns*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS: {
            lwlink_device_get_intranode_conns *iocReq = (lwlink_device_get_intranode_conns*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_ADD_INTERNODE_CONN: {
            lwlink_add_internode_conn *iocReq = (lwlink_add_internode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_REMOVE_INTERNODE_CONN: {
            lwlink_remove_internode_conn *iocReq = (lwlink_remove_internode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS: {
            lwlink_device_write_discovery_tokens *iocReq = (lwlink_device_write_discovery_tokens*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_READ_DISCOVERY_TOKENS: {
            lwlink_device_read_discovery_tokens *iocReq = (lwlink_device_read_discovery_tokens*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONN: {
            lwlink_train_intranode_conn *iocReq = (lwlink_train_intranode_conn*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK: {
            lwlink_train_internode_conn_link *iocReq = (lwlink_train_internode_conn_link*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK: {
            lwlink_train_internode_conn_sublink *iocReq = (lwlink_train_internode_conn_sublink*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
        case IOCTL_LWLINK_GET_DEVICES_INFO: {
            lwlink_get_devices_info *iocReq = (lwlink_get_devices_info*)ioctlParam;
            ioctlStatus = iocReq->status;
            break;
        }
    }

    // check for error case
    if (ioctlStatus != LWL_SUCCESS) {
        DcgmLWLinkErrorCodes errorCode = (DcgmLWLinkErrorCodes)DcgmFMLWLinkError::getLinkErrorCode(ioctlStatus);
        PRINT_ERROR("%x %s", "LWLinkCoreLib driver ioctl %x failed with status: %s",
                    ioctlCmd,
                    DcgmFMLWLinkError::getLinkErrorString(errorCode));
    }
}

const char*
DcgmFMLWLinkDrvIntf::getIoctlCmdString(int ioctlCmd)
{
    switch(ioctlCmd) {
        case IOCTL_LWLINK_CHECK_VERSION:
            return "IOCTL_LWLINK_CHECK_VERSION";
        case IOCTL_LWLINK_SET_NODE_ID:
            return "IOCTL_LWLINK_SET_NODE_ID";
        case IOCTL_LWLINK_SET_TX_COMMON_MODE:
            return "IOCTL_LWLINK_SET_TX_COMMON_MODE";
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case IOCTL_LWLINK_INITPHASE1:
            return "IOCTL_LWLINK_INITPHASE1";
        case IOCTL_LWLINK_INITNEGOTIATE:
            return "IOCTL_LWLINK_INITNEGOTIATE";
        case IOCTL_LWLINK_RX_INIT_TERM:
            return "IOCTL_LWLINK_RX_INIT_TERM";
        case IOCTL_LWLINK_SET_RX_DETECT:
            return "IOCTL_LWLINK_SET_RX_DETECT";
        case IOCTL_LWLINK_GET_RX_DETECT:
            return "IOCTL_LWLINK_GET_RX_DETECT";
#endif
        case IOCTL_LWLINK_CALIBRATE:
            return "IOCTL_LWLINK_CALIBRATE";
        case IOCTL_LWLINK_ENABLE_DATA:
            return "IOCTL_LWLINK_ENABLE_DATA";
        case IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC:
            return "IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC";
        case IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS:
            return "IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS";
        case IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS:
            return "IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS";
        case IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS:
            return "IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS";
        case IOCTL_LWLINK_ADD_INTERNODE_CONN:
            return "IOCTL_LWLINK_ADD_INTERNODE_CONN";
        case IOCTL_LWLINK_REMOVE_INTERNODE_CONN:
            return "IOCTL_LWLINK_REMOVE_INTERNODE_CONN";
        case IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS:
            return "IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS";
        case IOCTL_LWLINK_READ_DISCOVERY_TOKENS:
            return "IOCTL_LWLINK_READ_DISCOVERY_TOKENS";
        case IOCTL_LWLINK_TRAIN_INTRANODE_CONN:
            return "IOCTL_LWLINK_TRAIN_INTRANODE_CONN";
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK:
            return "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK";
        case IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK:
            return "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK";
        case IOCTL_LWLINK_GET_DEVICES_INFO:
            return "IOCTL_LWLINK_GET_DEVICES_INFO";
    }

    //default case, unknown ioctl type
    return "Unknown LWLinkCoreLib driver IOCTL";
}
