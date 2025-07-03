#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "logging.h"
#include "DcgmFMError.h"

DcgmFMError::DcgmFMError()
{
    mError.insert(std::make_pair(FM_SUCCESS,               "OK"));
    mError.insert(std::make_pair(FM_FILE_ILWALID,          "Invalid topology file"));
    mError.insert(std::make_pair(FM_FILE_OPEN_ERR,         "Failed to open file"));
    mError.insert(std::make_pair(FM_FILE_PARSING_ERR,      "Failed to parse file"));

    mError.insert(std::make_pair(FM_ILWALID_NODE,            "Invalid node"));
    mError.insert(std::make_pair(FM_ILWALID_GPU,             "Invalid GPU"));
    mError.insert(std::make_pair(FM_ILWALID_WILLOW,          "Invalid Willow"));
    mError.insert(std::make_pair(FM_ILWALID_PORT,            "Invalid port"));
    mError.insert(std::make_pair(FM_ILWALID_INGR_REQ_ENTRY,  "Invalid ingress request entry"));
    mError.insert(std::make_pair(FM_ILWALID_INGR_RESP_ENTRY, "Invalid ingress response entry"));
    mError.insert(std::make_pair(FM_ILWALID_PARTITION,       "Invalid partition"));

    mError.insert(std::make_pair(FM_ILWALID_NODE_CFG,            "Invalid node config"));
    mError.insert(std::make_pair(FM_ILWALID_GPU_CFG,             "Invalid GPU config"));
    mError.insert(std::make_pair(FM_ILWALID_WILLOW_CFG,          "Invalid Willow config"));
    mError.insert(std::make_pair(FM_ILWALID_PORT_CFG,            "Invalid port config"));
    mError.insert(std::make_pair(FM_ILWALID_INGR_REQ_ENTRY_CFG,  "Invalid ingress request entry config"));
    mError.insert(std::make_pair(FM_ILWALID_INGR_RESP_ENTRY_CFG, "Invalid ingress response entry config"));
    mError.insert(std::make_pair(FM_ILWALID_GANGED_LINK_ENTRY_CFG, "Invalid ganged link entry config"));

    mError.insert(std::make_pair(FM_ILWALID_GLOBAL_CONTROL_CONN_TO_LFM, "Invalid connection to local fabric manager"));
    mError.insert(std::make_pair(FM_ILWALID_LOCAL_CONTROL_CONN_TO_GFM,  "Invalid connection to gloabl fabric manager"));

    mError.insert(std::make_pair(FM_IOCTL_ERR,    "IOCTL error"));
    mError.insert(std::make_pair(FM_CFG_TIMEOUT,  "Configuration timeout"));
    mError.insert(std::make_pair(FM_CFG_ERROR,    "Configuration error"));
    mError.insert(std::make_pair(FM_MSG_SEND_ERR, "Failed to send message"));

};

DcgmFMError::~DcgmFMError()
{
    mError.clear();
}

const char * DcgmFMError::getErrorStr(FM_ERROR_CODE errCode)
{
    std::map <FM_ERROR_CODE, const char *>::iterator it;

    it = mError.find( errCode );
    if ( it != mError.end() )
    {
        return it->second;
    }
    else
    {
        return "Unknown error code.";
    }
}
