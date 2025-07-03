#ifndef DCGM_FABRIC_MANAGER_ERROR_H
#define DCGM_FABRIC_MANAGER_ERROR_H

#include <map>
#include <string>

typedef enum {
    FM_SUCCESS  = 0,

    // file errors
    FM_FILE_ILWALID,
    FM_FILE_OPEN_ERR,
    FM_FILE_PARSING_ERR,

    // invalid objects
    FM_ILWALID_NODE,
    FM_ILWALID_GPU,
    FM_ILWALID_WILLOW,
    FM_ILWALID_PORT,
    FM_ILWALID_INGR_REQ_ENTRY,
    FM_ILWALID_INGR_RESP_ENTRY,
    FM_ILWALID_TABLE_ENTRY,
    FM_ILWALID_GANGED_LINK_ENTRY,
    FM_ILWALID_PARTITION,

    // invalid config
    FM_ILWALID_NODE_CFG,
    FM_ILWALID_GPU_CFG,
    FM_ILWALID_WILLOW_CFG,
    FM_ILWALID_PORT_CFG,
    FM_ILWALID_INGR_REQ_ENTRY_CFG,
    FM_ILWALID_INGR_RESP_ENTRY_CFG,
    FM_ILWALID_GANGED_LINK_ENTRY_CFG,

    // invalid connection
    FM_ILWALID_GLOBAL_CONTROL_CONN_TO_LFM,
    FM_ILWALID_LOCAL_CONTROL_CONN_TO_GFM,

    // interface error
    FM_IOCTL_ERR,

    // config timeout and error
    FM_CFG_TIMEOUT,
    FM_CFG_ERROR,

    FM_MSG_SEND_ERR,


} FM_ERROR_CODE;

class DcgmFMError
{
public:
    DcgmFMError();
    virtual ~DcgmFMError();

    const char *getErrorStr(FM_ERROR_CODE errCode);

private:
    std::map <FM_ERROR_CODE, const char *> mError;
};

#endif
