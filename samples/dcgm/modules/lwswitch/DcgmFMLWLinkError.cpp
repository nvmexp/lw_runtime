
#include "DcgmFMLWLinkError.h"

int
DcgmFMLWLinkError::getLinkErrorCode(LwlStatus status)
{
    // lwrrently there is a one-to-one exact match for driver returned
    // lwlink errors. Special processing is required if there is no such
    // one-to-one translation.
    return status;
}

const char*
DcgmFMLWLinkError::getLinkErrorString(DcgmLWLinkErrorCodes errorCode)
{
    switch (errorCode) {
        case FM_LWLINK_ST_SUCCESS:
            return "Success";
        case FM_LWLINK_ST_BAD_ARGS:
            return "Invalid Arguments";
        case FM_LWLINK_ST_LWL_NO_MEM:
            return "Failed to allocate memory";
        case FM_LWLINK_ST_LWL_NOT_FOUND:
            return "Object/LWLink Device/Endpoint not found";
        case FM_LWLINK_ST_LWL_INITIALIZATION_PARTIAL_FAILURE:
            return "Initialization failed partially";
        case FM_LWLINK_ST_LWL_INITIALIZATION_TOTAL_FAILURE:
            return "Initialization failed";
        case FM_LWLINK_ST_LWL_PCI_ERROR:
            return "PCI error oclwred";
        case FM_LWLINK_ST_LWL_ERR_GENERIC:
            return "Generic error happened";
        case FM_LWLINK_ST_LWL_ERR_ILWALID_STATE:
            return "Invalid state detected";
        case FM_LWLINK_ST_LWL_UNBOUND_DEVICE:
            return "Unbound device";
        case FM_LWLINK_ST_LWL_MORE_PROCESSING_REQUIRED:
            return "More processing required";
        case FM_LWLINK_ST_LWL_IO_ERROR:
            return "IO error oclwrred";
        case FM_LWLINK_ST_TIMEOUT:
            return "Request timed out";
        case FM_LWLINK_ST_SLAVE_FM_SOCKET_ERR:
            return "Slave FM socket error";
        case FM_LWLINK_ST_MASTER_FM_SOCKET_ERR:
            return "Master FM socket error";
    }

    // no switch case matched. shouldn't happen
    return "Unknown error code";
}
