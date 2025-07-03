#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <iostream>
#include <map>
#include <list>
#include <sys/time.h>
#include <errno.h>
#include <mpi.h>

#include "helper.h"
#include "message_types.h"

#include "json/json.h"

#define LWLINK_DRV_PATH "/dev/lwpu-lwlink"
#define MAX_SWTICH_DEVICE 50

int DrvHandle;
int lwSwitchFds[MAX_SWTICH_DEVICE];
lwlink_get_devices_info deviceInfo;
LwU16 myNodeId = 0;
LWLinkConnList connList;
int switchArch;

uint64_t lwrrent_timestamp() 
{
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    uint64_t milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calwlate milliseconds
    return milliseconds;
}

struct devIdTuple {
    int nodeId;
    int busId;
};
static inline bool operator < (const devIdTuple& l, const devIdTuple& r)
{
    if (l.nodeId < r.nodeId)
        return true;
    else if((l.nodeId == r.nodeId) && (l.busId < r.busId))
        return true;
    else
        return false;
}

std::map<devIdTuple, int> dev_id_phy_id;

void setBusToPhyId(int node_id, int bus_id, int phy_id)
{
    devIdTuple dev_id;
    dev_id.nodeId = node_id;
    dev_id.busId = bus_id;
    dev_id_phy_id[dev_id] = phy_id;

}

int getBusToPhyId(int node_id, int bus_id)
{
    devIdTuple dev_id;
    dev_id.nodeId = node_id;
    dev_id.busId = bus_id;
    std::map<devIdTuple, int>::iterator it;
    if ( (it = dev_id_phy_id.find(dev_id)) != dev_id_phy_id.end() )
    {
        return it->second;
    }
    else
    {
        return -1;
    }
}

int open_switch_dev(std::string dev_name)
{
    int fd;
    dev_name.insert(0, "/dev/lwpu-");
    if((fd = open(dev_name.c_str(), O_WRONLY)) == -1)
    {
        fprintf(stderr, "Unable to open device %s errno=%d", dev_name.c_str(), errno);
        return -1;
    }
    else
    {
        return fd;
    }

}

uint64_t getDeviceId(lwlink_detailed_dev_info devInfo)
{
    uint64 deviceId = 0;
    deviceId = (uint64)(devInfo.pciInfo.domain) << 48;
    deviceId = deviceId | (uint64)devInfo.pciInfo.bus << 40;
    deviceId = deviceId | (uint64)devInfo.pciInfo.device << 32;
    deviceId = deviceId | (uint64)(devInfo.pciInfo.function) << 24;
    return deviceId;
}

/*
Get switch physical id by reading the LWSwitch
dev_id: Device instance in /dev/
Return: switch physical id
*/
static int getSwitchPhyId(int fd)
{
    uint32_t phy_id = -1;
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;
    if(ioctl(fd, IOCTL_LWSWITCH_GET_INFO, &ioctlParams) == -1)
    {
        perror("Unable to read switch physical ID");
        exit(0);
    }
    else
    {
        phy_id = ioctlParams.info[0];
    }
    return phy_id;
}

static int getSwitchArch(int fd)
{
    int arch;
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;

    if(ioctl(fd, IOCTL_LWSWITCH_GET_INFO, &ioctlParams) == -1)
    {
        perror("Unable to read switch physical ID");
        exit(0);
    }
    else
    {
        arch = ioctlParams.info[0];
    }

    return arch;
}

//map switch instance /dev/lwpu-lwswitch<dev id> to switch_id (0-12 as used in HWLinks table)
void
save_switch_phy_id(std::string device_name, int bus_id)
{
    int fd;
    devIdTuple dev_id;
    dev_id.nodeId = myNodeId;
    dev_id.busId = bus_id;

    dev_id_phy_id[dev_id] = -1;
    if (device_name.find("lwswitch") != std::string::npos)
    {
        if((fd = open_switch_dev(device_name)) != -1)
        {
            int phy_id = getSwitchPhyId(fd);
            switchArch = getSwitchArch(fd);
            setBusToPhyId(myNodeId, bus_id, phy_id);
            //PRINT_VERBOSE << "bus id = " << bus_id << "phyId = " <<  phy_id << "\n";
            close(fd);
        }
    }
}

int getArch()
{
    return switchArch; 
}

int getDevType(lwlink_get_devices_info deviceInfo, lwlink_pci_dev_info pciInfo)
{
    for (int idx = 0; idx < deviceInfo.numDevice; idx++) {
        if (deviceInfo.devInfo[idx].pciInfo.domain == pciInfo.domain && 
            deviceInfo.devInfo[idx].pciInfo.bus == pciInfo.bus &&
            deviceInfo.devInfo[idx].pciInfo.device == pciInfo.device &&
            deviceInfo.devInfo[idx].pciInfo.function == pciInfo.function) {
                return deviceInfo.devInfo[idx].devType;
        }
    }

    return 2;
}

static int checkVersion()
{
    LWSWITCH_GET_DEVICES_PARAMS params;
    LW_STATUS status;
    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            fprintf(stderr, "lwlink_train version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
            exit(-1);
        }
        // all other errors, log the error code and bail out
        fprintf(stderr, "lwlink_train:failed to query device information from LWSwitch driver, return status: %d\n", status);
        exit(-1);
    }
    if (params.deviceCount <= 0)
    {
        fprintf(stderr, "No LWSwitches found\n");
        exit(-1);
    }
    
    return params.deviceCount;
}

int open_lwlinklib_driver(unsigned short nodeId)
{
    int ret_val;

    DrvHandle = open( LWLINK_DRV_PATH, O_RDWR );
    if ( DrvHandle < 0) {
        PRINT_VERBOSE_ERRORS << "failed to open LWLinklib driver file: " << LWLINK_DRV_PATH << " Error is: "  << strerror(errno) << std::endl;
        return 1;
    }

    // open lwswitch devices. start with 0 and break when there is an error
    // this is required just for the driver to create the links
    for (int i = 0; i < MAX_SWTICH_DEVICE; i++) {
        std::string switchDev = "/dev/lwpu-lwswitch";
        std::ostringstream os; // older compiler don't support std::to_string()
        os << i;
        std::string idx = os.str();
        switchDev = switchDev + idx;
        lwSwitchFds[i] = open( switchDev.c_str(), O_RDWR );
    }

    myNodeId = nodeId;
    // issue set node id IOCTL
    lwlink_set_node_id idParams;
    idParams.nodeId = myNodeId;
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_NODE_ID, &idParams );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_SET_NODE_ID failed with:" << idParams.status << std::endl;
        return -1;
    }

    PRINT_VERBOSE_DEBUG << "setting node id success with status " << idParams.status << std::endl;
    
    return 0;
}

lwlink_get_devices_info get_device_information()
{
    int ret_val;
    int num_switches;

    num_switches = checkVersion();
    
    // query the driver for device information
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_GET_DEVICES_INFO, &deviceInfo );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_GET_DEVICES_INFO failed with:" << ret_val << std::endl;
        return deviceInfo;
    }

    PRINT_VERBOSE_DEBUG << "get_device_information - count " << deviceInfo.numDevice << std::endl;
    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        PRINT_VERBOSE_DEBUG << " \tdevice name " << deviceInfo.devInfo[idx].deviceName;
        PRINT_VERBOSE_DEBUG << " domain " << (int)deviceInfo.devInfo[idx].pciInfo.domain;
        PRINT_VERBOSE_DEBUG << " bus " << (int)deviceInfo.devInfo[idx].pciInfo.bus;
        PRINT_VERBOSE_DEBUG << " device " << (int)deviceInfo.devInfo[idx].pciInfo.device;
        PRINT_VERBOSE_DEBUG << " function " << (int)deviceInfo.devInfo[idx].pciInfo.function;
        PRINT_VERBOSE_DEBUG << std::endl;
        save_switch_phy_id(deviceInfo.devInfo[idx].deviceName, (int)deviceInfo.devInfo[idx].pciInfo.bus);
    }
    return deviceInfo;
}

void print_device_information(lwlink_get_devices_info deviceInfo)
{
    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        PRINT_VERBOSE << " \tdevice name " << deviceInfo.devInfo[idx].deviceName;
        PRINT_VERBOSE << " domain " << (int)deviceInfo.devInfo[idx].pciInfo.domain;
        PRINT_VERBOSE << " bus " << (int)deviceInfo.devInfo[idx].pciInfo.bus;
        PRINT_VERBOSE << " device " << (int)deviceInfo.devInfo[idx].pciInfo.device;
        PRINT_VERBOSE << " function " << (int)deviceInfo.devInfo[idx].pciInfo.function;
        PRINT_VERBOSE << std::endl;
    }
}

int getNodeId()
{
    return myNodeId;
}

bool set_initphase1()
{
    int ret_val;
    lwlink_initphase1 initphase1Param = {0};

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_INITPHASE1, &initphase1Param );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_INITPHASE1 failed with:" << ret_val << " on node "<< myNodeId << std::endl;
        return false;
    }
    
    if (initphase1Param.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "set_initphase1 failed with status " << initphase1Param.status << "on node " << myNodeId << std::endl;
        return false;
    }

    return true;
}

bool rx_init_term()
{
    int ret_val;
    lwlink_rx_init_term rxInitTermParam = {0};
    
    PRINT_VERBOSE_DEBUG << "rx_init_term" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_RX_INIT_TERM, &rxInitTermParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_RX_INIT_TERM failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (rxInitTermParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "rx_init_term failed with status " << rxInitTermParam.status << "on node " << myNodeId << std::endl;
        return false;
    }

    return true;
}


bool set_rx_detect()
{
    int ret_val;
    lwlink_set_rx_detect setRxDetectParam = {0};
    
    PRINT_VERBOSE_DEBUG << "set_rx_detect" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_RX_DETECT, &setRxDetectParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_SET_RX_DETECT failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (setRxDetectParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "set_rx_detect failed with status " << setRxDetectParam.status << "on node " << myNodeId << std::endl;
        return false;
    }

    return true;
}


bool get_rx_detect()
{
    int ret_val;
    lwlink_get_rx_detect getRxDetectParam = {0};
    
    PRINT_VERBOSE_DEBUG << "get_rx_detect" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_GET_RX_DETECT, &getRxDetectParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_GET_RX_DETECT failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (getRxDetectParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "get_rx_detect failed with status " << getRxDetectParam.status << "on node " << myNodeId << std::endl;
        return false;
    }

    return true;
}

bool enable_devices_common_mode()
{
    int ret_val;

    PRINT_VERBOSE_DEBUG << "enable_devices_common_mode" << std::endl;

    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = true;
    
    //PRINT_VERBOSE << "calling ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE" << std::endl;
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (modeParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "enable_devices_common_mode failed" << std::endl;
        return false;
    }

    return true;
}

bool disable_devices_common_mode()
{
    int ret_val;

    PRINT_VERBOSE_DEBUG << "disable_devices_common_mode" << std::endl;
    
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = false;
    
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (modeParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "disable_devices_common_mode failed" << std::endl;
        return false;
    }

    return true;
}

bool calibrate_devices()
{
    int ret_val;
    lwlink_calibrate calibrateParam = {0};
    
    PRINT_VERBOSE_DEBUG << "calibrate_devices" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_CALIBRATE, &calibrateParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_CALIBRATE failed with:" << ret_val << std::endl;
        return false;
    }
    
    if (calibrateParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "calibrate_devices failed" << std::endl;
        return false;
    }

    return true;
}

bool enable_devices_data()
{
    int ret_val;

    PRINT_VERBOSE_DEBUG << "enable_devices_data" << std::endl;

    lwlink_enable_data enableDataParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_ENABLE_DATA, &enableDataParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_ENABLE_DATA failed with:" << ret_val << std::endl;
        return false;
    }

    if (enableDataParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "enable_devices_data failed" << std::endl;
        return false;
    }

    return true;
}

bool set_initphase5()
{
    int ret_val;
    lwlink_initphase5 initphase5Param = {0};

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_INITPHASE5, &initphase5Param );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_INITPHASE5 failed with:" << ret_val << " on node "<< myNodeId << std::endl;
        return false;
    }

    if (initphase5Param.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "set_initphase5 failed with status " << initphase5Param.status << "on node " << myNodeId << std::endl;
        return false;
    }

    return true;
}

bool do_initnegotiate()
{
    int ret_val;

    PRINT_VERBOSE_DEBUG << "do_init_negotiate from " << myNodeId << std::endl;

    lwlink_initnegotiate initNegotiateParam;
    memset(&initNegotiateParam, 0, sizeof(initNegotiateParam));

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_INITNEGOTIATE, &initNegotiateParam );
   
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_INITNEGOTIATE failed with:" << ret_val << std::endl;
        return false;
    }

    if(initNegotiateParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "initnegotiate failed"<< std::endl;
        return false;
    }

    return true;
}

bool do_link_init()
{
    int ret_val;

    PRINT_VERBOSE_DEBUG << "do_link_init" << std::endl;

    lwlink_link_init_async initParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC, &initParam );
    
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC  failed with:" << ret_val << std::endl;
       return false;
    }

    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        lwlink_device_link_init_status statusParam;
        // fill the LWLink Device information
        statusParam.devInfo.pciInfo.domain = deviceInfo.devInfo[idx].pciInfo.domain;
        statusParam.devInfo.pciInfo.bus = deviceInfo.devInfo[idx].pciInfo.bus;
        statusParam.devInfo.pciInfo.device = deviceInfo.devInfo[idx].pciInfo.device;
        statusParam.devInfo.pciInfo.function = deviceInfo.devInfo[idx].pciInfo.function;
        statusParam.devInfo.nodeId =  myNodeId;
        //gdadwal: TODO
        ret_val = ioctl( DrvHandle, IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS, &statusParam );
       
        if (ret_val < 0) {
            PRINT_VERBOSE << "ioctl IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS failed with:" << ret_val << std::endl;
            return false;
        }
        // TODO: make this a verbose print
        PRINT_VERBOSE_DEBUG << "NodeId = " << statusParam.devInfo.nodeId << "\n";
        int link_num;
        PRINT_VERBOSE_DEBUG << " \tdevice name " << deviceInfo.devInfo[idx].deviceName;
        PRINT_VERBOSE_DEBUG << " domain " << (int)deviceInfo.devInfo[idx].pciInfo.domain;
        PRINT_VERBOSE_DEBUG << " bus " << (int)deviceInfo.devInfo[idx].pciInfo.bus;
        PRINT_VERBOSE_DEBUG << " device " << (int)deviceInfo.devInfo[idx].pciInfo.device;
        PRINT_VERBOSE_DEBUG << " function " << (int)deviceInfo.devInfo[idx].pciInfo.function;
        PRINT_VERBOSE_DEBUG << std::endl;
        for(link_num =0 ; link_num < LWLINK_MAX_DEVICE_CONN; link_num++) {
            PRINT_VERBOSE_DEBUG << "\tlinkIndex = " << statusParam.linkStatus[link_num].linkIndex << " initStatus = " << statusParam.linkStatus[link_num].initStatus << "\n";
        }
 
    }

    return true;
}

bool isDuplicateConnection(lwlink_connection_info conn, LWLinkConnList &connectionList)
{
    LWLinkConnList::iterator it;
    for ( it = connectionList.begin(); it != connectionList.end(); it++ ) {
        lwlink_connection_info tempConn = *it;
        if ( ((conn.srcEndPoint.nodeId == tempConn.srcEndPoint.nodeId) &&
             (conn.srcEndPoint.linkIndex == tempConn.srcEndPoint.linkIndex) &&
             (conn.srcEndPoint.pciInfo.domain == tempConn.srcEndPoint.pciInfo.domain) && 
             (conn.srcEndPoint.pciInfo.bus == tempConn.srcEndPoint.pciInfo.bus) &&
             (conn.srcEndPoint.pciInfo.device == tempConn.srcEndPoint.pciInfo.device) &&
             (conn.srcEndPoint.pciInfo.function == tempConn.srcEndPoint.pciInfo.function)) &&
             
             ((conn.dstEndPoint.nodeId == tempConn.dstEndPoint.nodeId) &&
             (conn.dstEndPoint.linkIndex == tempConn.dstEndPoint.linkIndex) &&
             (conn.dstEndPoint.pciInfo.domain == tempConn.dstEndPoint.pciInfo.domain) && 
             (conn.dstEndPoint.pciInfo.bus == tempConn.dstEndPoint.pciInfo.bus) &&
             (conn.dstEndPoint.pciInfo.device == tempConn.dstEndPoint.pciInfo.device) &&
             (conn.dstEndPoint.pciInfo.function == tempConn.dstEndPoint.pciInfo.function) ))
             
        {
            return true;
        }

        if ( ((conn.srcEndPoint.nodeId == tempConn.dstEndPoint.nodeId) &&
             (conn.srcEndPoint.linkIndex == tempConn.dstEndPoint.linkIndex) &&
             (conn.srcEndPoint.pciInfo.domain == tempConn.dstEndPoint.pciInfo.domain) && 
             (conn.srcEndPoint.pciInfo.bus == tempConn.dstEndPoint.pciInfo.bus) &&
             (conn.srcEndPoint.pciInfo.device == tempConn.dstEndPoint.pciInfo.device) &&
             (conn.srcEndPoint.pciInfo.function == tempConn.dstEndPoint.pciInfo.function)) &&
             
             ((conn.dstEndPoint.nodeId == tempConn.srcEndPoint.nodeId) &&
             (conn.dstEndPoint.linkIndex == tempConn.srcEndPoint.linkIndex) &&
             (conn.dstEndPoint.pciInfo.domain == tempConn.srcEndPoint.pciInfo.domain) && 
             (conn.dstEndPoint.pciInfo.bus == tempConn.srcEndPoint.pciInfo.bus) &&
             (conn.dstEndPoint.pciInfo.device == tempConn.srcEndPoint.pciInfo.device) &&
             (conn.dstEndPoint.pciInfo.function == tempConn.srcEndPoint.pciInfo.function) ))
             
        {
            return true;
        }
    }

    return false;
}

bool discover_intra_connections()
{
    int ret_val;

    // first initiate discover connection
    uint64_t startTime = lwrrent_timestamp();
    lwlink_discover_intranode_conns discoverParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discoverParam );
    if (ret_val < 0) {
        PRINT_VERBOSE << "ioctl IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS failed with:" << ret_val << std::endl;
        return false;
    }

    uint64_t finishTime = lwrrent_timestamp();
    
    connList.clear();
    
    // now get the list of connections
    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        lwlink_device_get_intranode_conns getConnParam;
        // fill the LWLink Device information
        getConnParam.devInfo.pciInfo.domain = deviceInfo.devInfo[idx].pciInfo.domain;
        getConnParam.devInfo.pciInfo.bus = deviceInfo.devInfo[idx].pciInfo.bus;
        getConnParam.devInfo.pciInfo.device = deviceInfo.devInfo[idx].pciInfo.device;
        getConnParam.devInfo.pciInfo.function = deviceInfo.devInfo[idx].pciInfo.function;
        getConnParam.devInfo.nodeId =  myNodeId;
        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &getConnParam );
        if (ret_val < 0) {
            PRINT_VERBOSE << "ioctl IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS failed with:" << ret_val << std::endl;
            return false;
        }

        PRINT_VERBOSE_DEBUG << "device=" << idx << "connecttions found =" << getConnParam.numConnections << "\n";
        // move the connections to our list
        for (unsigned int i = 0; i < getConnParam.numConnections; i++) {
            if (!isDuplicateConnection(getConnParam.conn[i], connList)) {
                connList.push_back(getConnParam.conn[i]);
            }
        }
    }

    // display_connections(connList);
    return true;
    // dump the connection information
    PRINT_VERBOSE_DEBUG << "Total number of intra-node connections = " << connList.size() << std::endl;
}

LWLinkConnList getIntraConns()
{
    return connList;
}

void display_connections(LWLinkConnList connList)
{
    LWLinkConnList::iterator it = connList.begin();
    int connIdx = 0;
    PRINT_VERBOSE << "nodeId\t(d::b:d.f)\tphyId\tlinkIndex\tdevType\tnodeIdFar\t(d::b:d.f)Far\tphyIdFar\tlinkIndexFar\tdevTypeFar\n";
    while ( it != connList.end() ) {
        lwlink_connection_info connInfo = *it;
        connIdx++;
        PRINT_VERBOSE << connInfo.srcEndPoint.nodeId;
        PRINT_VERBOSE << "\t(" << (int)connInfo.srcEndPoint.pciInfo.domain;
        PRINT_VERBOSE << ":" << (int)connInfo.srcEndPoint.pciInfo.bus;
        PRINT_VERBOSE << ":" << (int)connInfo.srcEndPoint.pciInfo.device;
        PRINT_VERBOSE << "." << (int)connInfo.srcEndPoint.pciInfo.function <<")";
        int32_t phyId = getBusToPhyId(connInfo.srcEndPoint.nodeId, (int) connInfo.srcEndPoint.pciInfo.bus);
        PRINT_VERBOSE << "\t" << phyId;
        PRINT_VERBOSE << "\t" << connInfo.srcEndPoint.linkIndex;
        if (phyId > 0)
            PRINT_VERBOSE << "\t\t" << 0;
        else
            PRINT_VERBOSE << "\t\t" << 1;

        PRINT_VERBOSE << "\t" << connInfo.dstEndPoint.nodeId;
        PRINT_VERBOSE << "\t\t(" << (int)connInfo.dstEndPoint.pciInfo.domain;
        PRINT_VERBOSE << ":" << (int)connInfo.dstEndPoint.pciInfo.bus;
        PRINT_VERBOSE << ":" << (int)connInfo.dstEndPoint.pciInfo.device;
        PRINT_VERBOSE << "." << (int)connInfo.dstEndPoint.pciInfo.function<<")";
        int32_t phyIdFar = getBusToPhyId(connInfo.srcEndPoint.nodeId, (int) connInfo.dstEndPoint.pciInfo.bus);
        PRINT_VERBOSE << "\t" << phyIdFar;
        PRINT_VERBOSE << "\t\t" << connInfo.dstEndPoint.linkIndex;
        if (phyIdFar > 0)
            PRINT_VERBOSE << "\t\t" << 0;
        else
            PRINT_VERBOSE << "\t\t" << 1;

        PRINT_VERBOSE << std::endl;

        it++;
    }
}

bool write_discovery_tokens(DiscoveryTokenList &writeList)
{
    int ret_val;
    lwlink_device_write_discovery_tokens writeParam;
    
    for(unsigned int i = 0; i < deviceInfo.numDevice; i++)
    {
        writeParam.devInfo.nodeId = myNodeId;
        writeParam.devInfo.pciInfo.domain = deviceInfo.devInfo[i].pciInfo.domain;
        writeParam.devInfo.pciInfo.bus = deviceInfo.devInfo[i].pciInfo.bus;
        writeParam.devInfo.pciInfo.device = deviceInfo.devInfo[i].pciInfo.device;
        writeParam.devInfo.pciInfo.function = deviceInfo.devInfo[i].pciInfo.function;

        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS, &writeParam );
        if (writeParam.status != LWL_SUCCESS) {
            PRINT_VERBOSE << "IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS ioctl failed with:" << writeParam.status << std::endl;
            return false;
        }
        
        for (unsigned int idx = 0; idx < writeParam.numTokens; idx++ ) {
            DiscoveryTokenInfo info;
            info.nodeId = myNodeId;
            info.domain = deviceInfo.devInfo[i].pciInfo.domain;
            info.bus = deviceInfo.devInfo[i].pciInfo.bus;
            info.device = deviceInfo.devInfo[i].pciInfo.device;
            info.function = deviceInfo.devInfo[i].pciInfo.function;
            info.linkIndex = writeParam.tokenInfo[idx].linkIndex;
            info.tokelwalue = writeParam.tokenInfo[idx].tokelwalue;
            // info.devType = deviceInfo.devInfo[i].devType;
            writeList.push_back(info);
        }
    }

    return true;
}

bool read_discovery_tokens(DiscoveryTokenList &readList)
{
    //printf("read_discovery_tokens called\n");

    int ret_val;
    lwlink_device_read_discovery_tokens readParam;
    
    for(int dev_num = 0; dev_num < 6; dev_num++) 
    {
        //printf("reading tokens from dev_num = %d \n", dev_num);
        readParam.devInfo.nodeId = myNodeId;
        readParam.devInfo.pciInfo.domain = deviceInfo.devInfo[dev_num].pciInfo.domain;
        readParam.devInfo.pciInfo.bus = deviceInfo.devInfo[dev_num].pciInfo.bus;
        readParam.devInfo.pciInfo.device = deviceInfo.devInfo[dev_num].pciInfo.device;
        readParam.devInfo.pciInfo.function = deviceInfo.devInfo[dev_num].pciInfo.function;

        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_READ_DISCOVERY_TOKENS, &readParam );
        if (readParam.status != LWL_SUCCESS) {
            PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_READ_DISCOVERY_TOKENS ioctl failed with:" << readParam.status << std::endl;
            return false;
        }
        for (unsigned int idx = 0; idx < readParam.numTokens; idx++ ) {
            DiscoveryTokenInfo info;
            info.nodeId = myNodeId;
            info.domain = deviceInfo.devInfo[dev_num].pciInfo.domain;
            info.bus = deviceInfo.devInfo[dev_num].pciInfo.bus;
            info.device = deviceInfo.devInfo[dev_num].pciInfo.device;
            info.function = deviceInfo.devInfo[dev_num].pciInfo.function;
            info.linkIndex = readParam.tokenInfo[idx].linkIndex;
            info.tokelwalue = readParam.tokenInfo[idx].tokelwalue;
            info.phyId = getBusToPhyId(myNodeId, (int) deviceInfo.devInfo[0].pciInfo.bus);
            // info.devType = deviceInfo.devInfo[i].devType;
            readList.push_back(info);
        }
    }

    return true;
}

bool read_SIDs(SidInfoList &sidList)
{
    int ret_val = 0;
    int linkIdx;

    sidList.clear();

    for(unsigned int i = 0; i < deviceInfo.numDevice; i++)
    {
        lwlink_device_read_sids readParam;
        memset( &readParam, 0, sizeof(readParam));

        // fill the LWLink Device information
        readParam.devInfo.nodeId = myNodeId;
        readParam.devInfo.pciInfo.domain = deviceInfo.devInfo[i].pciInfo.domain;
        readParam.devInfo.pciInfo.bus = deviceInfo.devInfo[i].pciInfo.bus;
        readParam.devInfo.pciInfo.device = deviceInfo.devInfo[i].pciInfo.device;
        readParam.devInfo.pciInfo.function = deviceInfo.devInfo[i].pciInfo.function;

        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_DEVICE_READ_SIDS, &readParam );
        if (readParam.status != LWL_SUCCESS) {
            PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_DEVICE_READ_SIDS ioctl failed with:" << readParam.status << std::endl;
            return false;
        } else {
            for (linkIdx = 0; linkIdx < readParam.numEntries; linkIdx++) {
                SidNodeConnectionInfo sidInfo;

                sidInfo.nodeId = myNodeId;
                sidInfo.gpuOrSwitchId = getDeviceId(deviceInfo.devInfo[i]);
                sidInfo.nearSid = readParam.sidInfo[linkIdx].localLinkSid;
                sidInfo.nearLinkIndex = readParam.sidInfo[linkIdx].localLinkNum;
                sidInfo.farSid = readParam.sidInfo[linkIdx].remoteLinkSid;
                sidInfo.farLinkIndex = readParam.sidInfo[linkIdx].remoteLinkNum;
                sidInfo.domain = deviceInfo.devInfo[i].pciInfo.domain;
                sidInfo.bus = deviceInfo.devInfo[i].pciInfo.bus;
                sidInfo.device = deviceInfo.devInfo[i].pciInfo.device;
                sidInfo.function = deviceInfo.devInfo[i].pciInfo.function;
                sidInfo.devType = deviceInfo.devInfo[i].devType;
                if (sidInfo.farSid == 0 && sidInfo.nearSid == 0) continue;
                PRINT_VERBOSE_DEBUG << "\tFound the following connection \n";
                PRINT_VERBOSE_DEBUG << " \tnodeId  " << sidInfo.nodeId;
                PRINT_VERBOSE_DEBUG << " linkIndex " << sidInfo.nearLinkIndex;
                PRINT_VERBOSE_DEBUG << " Sid " << (int)sidInfo.nearSid;
                PRINT_VERBOSE_DEBUG << "<======>";
                PRINT_VERBOSE_DEBUG << " linkIndex " << sidInfo.farLinkIndex;
                PRINT_VERBOSE_DEBUG << " Sid " << (int)sidInfo.farSid<<std::endl;
                // TODO : remove this workaround for driver returning junk for bad GPU links as though they are for remote node
                // sidInfo.farSid = sidInfo.nodeId ? 0: 1;
                // sidInfo.nearSid = sidInfo.nodeId;
                // // TODO: hard-coded for E4700 based 2-node system as links 0-12 connect to local node GPU.
                // if (sidInfo.nearLinkIndex < 12)
                //     continue;
                sidList.push_back(sidInfo);
            }
        }
    }

    return true;
} 

bool add_internode_connections(lwlink_endpoint &localEndPoint, lwlink_remote_endpoint_info &remoteEndPoint)
{
    int ret_val;
    lwlink_add_internode_conn addParam;
    
    addParam.localEndPoint = localEndPoint;
    addParam.remoteEndPoint = remoteEndPoint;
    
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_ADD_INTERNODE_CONN, &addParam );
    if (addParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_ADD_INTERNODE_CONNECTION ioctl failed with:" << addParam.status << std::endl;
        return false;
    }

    return true;
}

bool train_intra_connection(lwlink_conn_train_type trainTo, int connIdx)
{
    // first get the required connection from index
    if (connList.size() < connIdx) {
        PRINT_VERBOSE_ERRORS <<"Invalid connection index " << std::endl;
        return false;
    }

    LWLinkConnList::iterator it = connList.begin();
    std::advance(it, connIdx);
    lwlink_connection_info connInfo = *it;
    int ret_val;

    PRINT_VERBOSE << "\tTraining the following connection \n";
    PRINT_VERBOSE << " \tnodeId  " << connInfo.srcEndPoint.nodeId;
    PRINT_VERBOSE << " linkIndex " << connInfo.srcEndPoint.linkIndex;
    PRINT_VERBOSE << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
    PRINT_VERBOSE << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
    PRINT_VERBOSE << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
    PRINT_VERBOSE << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
    PRINT_VERBOSE << "<======>";
    PRINT_VERBOSE << " nodeId = " << connInfo.dstEndPoint.nodeId;
    PRINT_VERBOSE << " linkIndex " << connInfo.dstEndPoint.linkIndex;
    PRINT_VERBOSE << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
    PRINT_VERBOSE << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
    PRINT_VERBOSE << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
    PRINT_VERBOSE << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
    PRINT_VERBOSE << std::endl;
    

    lwlink_train_intranode_conn trainParam;
    trainParam.trainTo = trainTo;

    // fill source device information
    trainParam.srcEndPoint.nodeId = connInfo.srcEndPoint.nodeId;
    trainParam.srcEndPoint.linkIndex= connInfo.srcEndPoint.linkIndex;
    trainParam.srcEndPoint.pciInfo.domain = connInfo.srcEndPoint.pciInfo.domain;
    trainParam.srcEndPoint.pciInfo.bus = connInfo.srcEndPoint.pciInfo.bus;
    trainParam.srcEndPoint.pciInfo.device = connInfo.srcEndPoint.pciInfo.device;
    trainParam.srcEndPoint.pciInfo.function= connInfo.srcEndPoint.pciInfo.function;
    // fill destination device information
    trainParam.dstEndPoint.nodeId = connInfo.dstEndPoint.nodeId;    
    trainParam.dstEndPoint.linkIndex= connInfo.dstEndPoint.linkIndex;
    trainParam.dstEndPoint.pciInfo.domain = connInfo.dstEndPoint.pciInfo.domain;
    trainParam.dstEndPoint.pciInfo.bus = connInfo.dstEndPoint.pciInfo.bus;
    trainParam.dstEndPoint.pciInfo.device = connInfo.dstEndPoint.pciInfo.device;
    trainParam.dstEndPoint.pciInfo.function= connInfo.dstEndPoint.pciInfo.function;    

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTRANODE_CONN, &trainParam );
    if (trainParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "Train Connection ioctl IOCTL_LWLINK_TRAIN_INTRANODE_CONN failed with:" << trainParam.status << std::endl;
        return false;
    }

    PRINT_VERBOSE_DEBUG << "\tSrc linkMode " << getMainLinkStateString(trainParam.srcEndState.linkMode);
    PRINT_VERBOSE_DEBUG << " Src txSubLinkMode " << getTxSubLinkStateString(trainParam.srcEndState.txSubLinkMode);
    PRINT_VERBOSE_DEBUG << " Src rxSubLinkMode " << getRxSubLinkStateString(trainParam.srcEndState.rxSubLinkMode);
    PRINT_VERBOSE_DEBUG << std::endl;    
    PRINT_VERBOSE_DEBUG << "\tDst linkMode " << getMainLinkStateString(trainParam.dstEndState.linkMode);
    PRINT_VERBOSE_DEBUG << " Dst txSubLinkMode " << getTxSubLinkStateString(trainParam.dstEndState.txSubLinkMode);
    PRINT_VERBOSE_DEBUG << " Dst rxSubLinkMode " << getRxSubLinkStateString(trainParam.dstEndState.rxSubLinkMode);
    PRINT_VERBOSE_DEBUG << std::endl;

    return true;
}

bool train_all_intra_connections(lwlink_conn_train_type trainTo)
{
    bool anyFail = false;
    LWLinkConnList::iterator it = connList.begin();
    unsigned int connIdx = 0;
    uint64 startTime = lwrrent_timestamp();
    PRINT_VERBOSE << "connList.size() " << connList.size() << std::endl;
    while ( it != connList.end() ) {
        if (!train_intra_connection(trainTo, connIdx)) {
            anyFail = true;
        }

        connIdx++;
        it++;
    }
    uint64 finishTime = lwrrent_timestamp();
    PRINT_VERBOSE << "train_all_connections took " << finishTime - startTime << " milliseconds" << std::endl;
    return !anyFail;
}

bool train_intra_conn_parallel(lwlink_conn_train_type trainTo, LWLinkConnList connInfoList)
{
    int ret_val;
    lwlink_train_intranode_conns_parallel* pTrainParam = NULL;
    pTrainParam = (lwlink_train_intranode_conns_parallel*) calloc(1, sizeof(lwlink_train_intranode_conns_parallel));

    pTrainParam->trainTo = trainTo;
    pTrainParam->endPointPairsCount = connInfoList.size();

    LWLinkConnList::iterator it;
    int i  = 0;
    for (it = connInfoList.begin(); it != connInfoList.end(); it++) {
        lwlink_connection_info connInfo = *it;
        pTrainParam->endPointPairs[i].src.nodeId = connInfo.srcEndPoint.nodeId;
        pTrainParam->endPointPairs[i].src.linkIndex = connInfo.srcEndPoint.linkIndex;
        pTrainParam->endPointPairs[i].src.pciInfo.domain = connInfo.srcEndPoint.pciInfo.domain;
        pTrainParam->endPointPairs[i].src.pciInfo.bus = connInfo.srcEndPoint.pciInfo.bus;
        pTrainParam->endPointPairs[i].src.pciInfo.device = connInfo.srcEndPoint.pciInfo.device;
        pTrainParam->endPointPairs[i].src.pciInfo.function = connInfo.srcEndPoint.pciInfo.function;

        pTrainParam->endPointPairs[i].dst.nodeId = connInfo.dstEndPoint.nodeId;
        pTrainParam->endPointPairs[i].dst.linkIndex = connInfo.dstEndPoint.linkIndex;
        pTrainParam->endPointPairs[i].dst.pciInfo.domain = connInfo.dstEndPoint.pciInfo.domain;
        pTrainParam->endPointPairs[i].dst.pciInfo.bus = connInfo.dstEndPoint.pciInfo.bus;
        pTrainParam->endPointPairs[i].dst.pciInfo.device = connInfo.dstEndPoint.pciInfo.device;
        pTrainParam->endPointPairs[i].dst.pciInfo.function = connInfo.dstEndPoint.pciInfo.function;
        i++;
    }

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL, pTrainParam );
    if (ret_val < 0) {
        PRINT_VERBOSE_ERRORS << "ioctl IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL failed with:" << ret_val << "with err value " << errno << std::endl;
        return false;
    }

    if (pTrainParam->status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "Train Connection ioctl IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL failed with:" << pTrainParam->status << std::endl;
        return false;
    }

    for (int i = 0; i < pTrainParam->endPointPairsCount; i++) {
        PRINT_VERBOSE_DEBUG << "\tSrc linkMode " << getMainLinkStateString(pTrainParam->endpointPairsStates[i].srcEnd.linkMode);
        PRINT_VERBOSE_DEBUG << " Src txSubLinkMode " << getTxSubLinkStateString(pTrainParam->endpointPairsStates[i].srcEnd.txSubLinkMode);
        PRINT_VERBOSE_DEBUG << " Src rxSubLinkMode " << getRxSubLinkStateString(pTrainParam->endpointPairsStates[i].srcEnd.rxSubLinkMode);
        PRINT_VERBOSE_DEBUG << std::endl;    
        PRINT_VERBOSE_DEBUG << "\tDst linkMode " << getMainLinkStateString(pTrainParam->endpointPairsStates[i].dstEnd.linkMode);
        PRINT_VERBOSE_DEBUG << " Dst txSubLinkMode " << getTxSubLinkStateString(pTrainParam->endpointPairsStates[i].dstEnd.txSubLinkMode);
        PRINT_VERBOSE_DEBUG << " Dst rxSubLinkMode " << getRxSubLinkStateString(pTrainParam->endpointPairsStates[i].dstEnd.rxSubLinkMode);
        PRINT_VERBOSE_DEBUG << std::endl;
    }
    return true;
}

bool set_mainlink_state(lwlink_link_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint)
{
    int ret_val;
    lwlink_train_internode_conn_link mainLinkParam;
    std::string masterSlave = isMasterEnd ? "master" : "slave";
    mainLinkParam.trainTo = trainTo;
    mainLinkParam.isMasterEnd = isMasterEnd;
    mainLinkParam.localEndPoint = localEndPoint;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, &mainLinkParam );
    if (mainLinkParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK ioctl failed with:" << mainLinkParam.status << " at " << masterSlave << std::endl;
        return false;
    }

    PRINT_VERBOSE_DEBUG << "Set mainlink state at " << masterSlave << std::endl;
    PRINT_VERBOSE_DEBUG << "\tlinkMode " << getMainLinkStateString(mainLinkParam.localEndState.linkMode);
    PRINT_VERBOSE_DEBUG << " txSubLinkMode " << getTxSubLinkStateString(mainLinkParam.localEndState.txSubLinkMode);
    PRINT_VERBOSE_DEBUG << " rxSubLinkMode " << getRxSubLinkStateString(mainLinkParam.localEndState.rxSubLinkMode);
    PRINT_VERBOSE_DEBUG << std::endl;

    return true;
}

bool set_sublink_state(lwlink_sublink_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint)
{
    int ret_val;
    lwlink_train_internode_conn_sublink subLinkParam;
    std::string masterSlave = isMasterEnd ? "master" : "slave";
    subLinkParam.trainTo = trainTo;
    subLinkParam.isMasterEnd = isMasterEnd;
    subLinkParam.localEndPoint = localEndPoint;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, &subLinkParam );
    if (subLinkParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK ioctl failed with: " << subLinkParam.status << " at " << masterSlave << std::endl;
        return false;
    }
    
    PRINT_VERBOSE_DEBUG << "Set sublink state at " << masterSlave << std::endl;
    PRINT_VERBOSE_DEBUG << "\tlinkMode " << getMainLinkStateString(subLinkParam.localEndState.linkMode);
    PRINT_VERBOSE_DEBUG << " txSubLinkMode " << getTxSubLinkStateString(subLinkParam.localEndState.txSubLinkMode);
    PRINT_VERBOSE_DEBUG << " rxSubLinkMode " << getRxSubLinkStateString(subLinkParam.localEndState.rxSubLinkMode);
    PRINT_VERBOSE_DEBUG << std::endl;

    return true;
}

bool doOptimizeReq(Json::Value message)
{
    Json::Value optimizeLinkTrainEndPoint(Json::objectValue);
    Json::Value optimizeLinkTrainList(Json::arrayValue);

    optimizeLinkTrainList = message["end_point_list"];
    int endPointCount = message["end_point_count"].asInt();
    if (message["optimize_type"].asString() == "INIT_OPTIMIZE") return doInitOptimize(optimizeLinkTrainList, endPointCount);
    else return doPostInitOptimize(optimizeLinkTrainList, endPointCount);
}

bool doInitOptimize(Json::Value optimizeLinkTrainList, int endPointCount)
{
    lwlink_train_internode_links_initoptimize trainParam;
    memset( &trainParam, 0 , sizeof(trainParam) );
    trainParam.endPointCount = endPointCount;
    int i = 0;
    for (auto itr:optimizeLinkTrainList) {
        trainParam.endPoints[i].nodeId = itr["master_node_id"].asInt();
        trainParam.endPoints[i].linkIndex = itr["link_index"].asInt();
        trainParam.endPoints[i].pciInfo = getJsonPciInfo(itr["pci_info"]);
        i++;
    }

    int ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE, &trainParam );
    if (trainParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE ioctl failed with: " << trainParam.status << std::endl;
        return false;
    }

    return true;
}

bool doPostInitOptimize(Json::Value optimizeLinkTrainList, int endPointCount)
{
    lwlink_train_internode_links_post_initoptimize trainParam;
    memset( &trainParam, 0 , sizeof(trainParam) );
    trainParam.endPointCount = endPointCount;
    int i = 0;
    for (auto itr:optimizeLinkTrainList) {
        trainParam.endPoints[i].nodeId = itr["master_node_id"].asInt();
        trainParam.endPoints[i].linkIndex = itr["link_index"].asInt();
        trainParam.endPoints[i].pciInfo = getJsonPciInfo(itr["pci_info"]);
        i++;
    }

    int ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE, &trainParam );
    if (trainParam.status != LWL_SUCCESS) {
        PRINT_VERBOSE_ERRORS << "IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE ioctl failed with: " << trainParam.status << std::endl;
        return false;
    }

    return true;
}

std::string getMainLinkStateString(int linkMode)
{
    // printf("linkMode = %d\n", linkMode);
    switch (linkMode) {
        case lwlink_link_mode_off:
            return "Init/Off";
        case lwlink_link_mode_active:
            return "Active";
        case lwlink_link_mode_swcfg:
            return "Swcfg";
        case lwlink_link_mode_fault:
            return "Faulty";
        case lwlink_link_mode_recovery:
            return "Recovery";
        case lwlink_link_mode_fail:
            return "Fail";
        case lwlink_link_mode_detect:
            return "Detect";
        case lwlink_link_mode_reset:
            return "Reset";
        case lwlink_link_mode_enable_pm:
            return "Enable PM";
        case lwlink_link_mode_disable_pm:
            return "Disable PM";
        case lwlink_link_mode_traffic_setup:
            return "Setup Traffic";
        case lwlink_link_mode_unknown:
            return "Unknown";
        case lwlink_link_mode_contain:
            return "contain";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string getTxSubLinkStateString(int txSubLinkMode)
{
    // printf("txSubLinkMode = %d\n", txSubLinkMode);
    switch (txSubLinkMode) {
        case lwlink_tx_sublink_mode_hs:
            return "High Speed";
        case lwlink_tx_sublink_mode_single_lane:
            return "Single Lane";
        case lwlink_tx_sublink_mode_train:
            return "Training";
        case lwlink_tx_sublink_mode_safe:
            return "Safe";
        case lwlink_tx_sublink_mode_off:
            return "Off";
        case lwlink_tx_sublink_mode_common_mode:
            return "Common Mode Enable";
        case lwlink_tx_sublink_mode_common_mode_disable:
            return "Common Mode Disable";
        case lwlink_tx_sublink_mode_data_ready:
            return "Data Ready";
        case lwlink_tx_sublink_mode_tx_eq:
            return "Equalization";
        case lwlink_tx_sublink_mode_pbrs_en:
            return "PRBS Generator";
        case lwlink_tx_sublink_mode_post_hs:
            return "Post Active HW Retraining";
        case lwlink_tx_sublink_mode_unknown:
            return "Unknown";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string getRxSubLinkStateString(int rxSubLinkMode)
{
    // printf("rxSubLinkMode = %d\n", rxSubLinkMode);
    switch (rxSubLinkMode) {
        case lwlink_rx_sublink_mode_hs:
            return "High Speed";
        case lwlink_rx_sublink_mode_single_lane:
            return "Single Lane";
        case lwlink_rx_sublink_mode_train:
            return "Training";
        case lwlink_rx_sublink_mode_safe:
            return "Safe";
        case lwlink_rx_sublink_mode_off:
            return "Off";
        case lwlink_rx_sublink_mode_rxcal:
            return "Calibration";
        case lwlink_rx_sublink_mode_unknown:
            return "Unknown";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

Json::Value getPciInfoJson(lwlink_endpoint endPoint)
{
    Json::Value pciInfo(Json::objectValue);
    pciInfo["domain"] = endPoint.pciInfo.domain;
    pciInfo["bus"] = endPoint.pciInfo.bus;
    pciInfo["device"] = endPoint.pciInfo.device;
    pciInfo["function"] = endPoint.pciInfo.function;
    return pciInfo;
}

lwlink_pci_dev_info getJsonPciInfo(Json::Value pci_info)
{
    lwlink_pci_dev_info pciInfo; 
    pciInfo.domain = pci_info["domain"].asInt();
    pciInfo.bus = pci_info["bus"].asInt();
    pciInfo.device = pci_info["device"].asInt();
    pciInfo.function = pci_info["function"].asInt();
    return pciInfo;
}

std::string getStringFromUint64(uint64_t val)
{
    std::ostringstream oss;
    oss << val;
    std::string intAsString(oss.str());
    return intAsString;
}

uint64_t getUint64FromString(std::string strVal)
{
    uint64_t value;
    std::istringstream iss(strVal);
    iss >> value;
    return value;
}

void colwertConnInfoToJson(lwlink_endpoint srcEndPoint, lwlink_endpoint dstEndPoint, Json::Value &message)
{
    message["src_nodeId"] = srcEndPoint.nodeId;
    message["src_link_index"] = srcEndPoint.linkIndex;
    message["src_domain"] = srcEndPoint.pciInfo.domain;
    message["src_bus"] = srcEndPoint.pciInfo.bus;
    message["src_device"] = srcEndPoint.pciInfo.device;
    message["src_function"] = srcEndPoint.pciInfo.function;
    message["dst_nodeId"] = dstEndPoint.nodeId;
    message["dst_link_index"] = dstEndPoint.linkIndex;
    message["dst_domain"] = dstEndPoint.pciInfo.domain;
    message["dst_bus"] = dstEndPoint.pciInfo.bus;
    message["dst_device"] = dstEndPoint.pciInfo.device;
    message["dst_function"] = dstEndPoint.pciInfo.function;
}

void colwertJsonToConnInfo(lwlink_endpoint &srcEndPoint, lwlink_endpoint &dstEndPoint, Json::Value message)
{
    srcEndPoint.nodeId = message["src_nodeId"].asInt();
    srcEndPoint.linkIndex = message["src_link_index"].asInt();
    srcEndPoint.pciInfo.domain = message["src_domain"].asInt();
    srcEndPoint.pciInfo.bus = message["src_bus"].asInt();
    srcEndPoint.pciInfo.device = message["src_device"].asInt();
    srcEndPoint.pciInfo.function = message["src_function"].asInt();
    dstEndPoint.nodeId = message["dst_nodeId"].asInt();
    dstEndPoint.linkIndex = message["dst_link_index"].asInt();
    dstEndPoint.pciInfo.domain = message["dst_domain"].asInt();
    dstEndPoint.pciInfo.bus = message["dst_bus"].asInt();
    dstEndPoint.pciInfo.device = message["dst_device"].asInt();
    dstEndPoint.pciInfo.function = message["dst_function"].asInt();
}

void printConnInfo(lwlink_connection_info connInfo)
{
    PRINT_VERBOSE << " \tnodeId  " << connInfo.srcEndPoint.nodeId;
    PRINT_VERBOSE << " linkIndex " << connInfo.srcEndPoint.linkIndex;
    PRINT_VERBOSE << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
    PRINT_VERBOSE << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
    PRINT_VERBOSE << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
    PRINT_VERBOSE << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
    PRINT_VERBOSE << "<======>";
    PRINT_VERBOSE << " nodeId = " << connInfo.dstEndPoint.nodeId;
    PRINT_VERBOSE << " linkIndex " << connInfo.dstEndPoint.linkIndex;
    PRINT_VERBOSE << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
    PRINT_VERBOSE << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
    PRINT_VERBOSE << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
    PRINT_VERBOSE << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
    PRINT_VERBOSE << std::endl;
}

void sendMessage(Json::Value message, int dest)
{
    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(message);
    MPI_Send(sStyled.c_str(), sStyled.size(), MPI_CHAR, dest, 0, MPI_COMM_WORLD);
}

std::string recvMessage(int src)
{
    MPI_Status status;
    int size;
    // Probe for an incoming message from master
    MPI_Probe(src, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &size);
    
    char arr[size];

    MPI_Recv(&arr, size, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::string recvMsg(arr);
    return recvMsg;
}
