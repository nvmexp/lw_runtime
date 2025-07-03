
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

#include "lwlink.h"
#include "lwlink_lib_ioctl.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwos.h"
#include "socket_interface.h"
#include "helper.h"
#include "lwlink_train_cmd_parser.h"

extern "C"
{
#include "lwswitch_user_linux.h"
}

#define LWLINK_DRV_PATH "/dev/lwpu-lwlink"
#define MAX_SWTICH_DEVICE 50

int DrvHandle;
int lwSwitchFds[MAX_SWTICH_DEVICE];
lwlink_get_devices_info deviceInfo;
LWLinkConnList connList;
LwU16 myNodeId = 0;

uint64 lwrrent_timestamp() 
{
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    uint64 milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calwlate milliseconds
    return milliseconds;
}

int open_lwlinklib_driver(unsigned short nodeId)
{
    int ret_val;

    DrvHandle = open( LWLINK_DRV_PATH, O_RDWR );
    if ( DrvHandle < 0) {
        std::cout << "failed to open LWLinklib driver file: " << LWLINK_DRV_PATH << " Error is: "  << strerror(errno) << std::endl;
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
        std::cout << "ioctl IOCTL_LWLINK_SET_NODE_ID failed with:" << idParams.status << std::endl;
        return -1;
    }
    
    return 0;
}


//check version and return number of LWSwitches found
static int checkVersion()
{
    LWSWITCH_GET_DEVICES_PARAMS params;
    LW_STATUS status;
    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            fprintf(stderr, "lwswitch-audit version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
            exit(-1);
        }
        // all other errors, log the error code and bail out
        fprintf(stderr, "lwswitch-audit:failed to query device information from LWSwitch driver, return status: %d\n", status);
        exit(-1);
    }
    if (params.deviceCount <= 0)
    {
        fprintf(stderr, "No LWSwitches found\n");
        exit(-1);
    }
    return params.deviceCount;
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
    }
    else
    {
        phy_id = ioctlParams.info[0];
    }
    return phy_id;
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
            setBusToPhyId(myNodeId, bus_id, phy_id);
            //std::cout << "bus id = " << bus_id << "phyId = " <<  phy_id << "\n";
            close(fd);
        }
    }
}

void get_device_information()
{
    int ret_val;
    int num_switches;

    num_switches = checkVersion();
    std::cout << "get_device_information" << std::endl;
    
    // query the driver for device information
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_GET_DEVICES_INFO, &deviceInfo );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_GET_DEVICES_INFO failed with:" << ret_val << std::endl;
        return;
    }

    std::cout << "get_device_information - count " << deviceInfo.numDevice << std::endl;
    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        std::cout << " \tdevice name " << deviceInfo.devInfo[idx].deviceName;
        std::cout << " domain " << (int)deviceInfo.devInfo[idx].pciInfo.domain;
        std::cout << " bus " << (int)deviceInfo.devInfo[idx].pciInfo.bus;
        std::cout << " device " << (int)deviceInfo.devInfo[idx].pciInfo.device;
        std::cout << " function " << (int)deviceInfo.devInfo[idx].pciInfo.function;
        std::cout << std::endl;
        save_switch_phy_id(deviceInfo.devInfo[idx].deviceName, (int)deviceInfo.devInfo[idx].pciInfo.bus);
    }
}

void enable_devices_common_mode()
{
    int ret_val;

    std::cout << "enable_devices_common_mode" << std::endl;

    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = true;

    
    std::cout << "calling ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE" << std::endl;
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE failed with:" << ret_val << std::endl;
        return;
    }
    std::cout << "returned ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE" << std::endl;
    
    if (modeParam.status != LWL_SUCCESS) {
        std::cout << "enable_devices_common_mode failed" << std::endl;
    }
}

void disable_devices_common_mode()
{
    int ret_val;

    //std::cout << "disable_devices_common_mode" << std::endl;
    
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = false;
    
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_SET_TX_COMMON_MODE failed with:" << ret_val << std::endl;
        return;
    }
    
    if (modeParam.status != LWL_SUCCESS) {
        std::cout << "disable_devices_common_mode failed" << std::endl;
    }
}

void set_initphase1()
{
    int ret_val;
    lwlink_initphase1 initphase1Param = {0};
    
    //std::cout << "set_initphase1" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_INITPHASE1, &initphase1Param );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_INITPHASE1 failed with:" << ret_val << std::endl;
        return;
    }
    
    if (initphase1Param.status != LWL_SUCCESS) {
        std::cout << "set_initphase1 failed" << std::endl;
    }
}
void rx_init_term()
{
    int ret_val;
    lwlink_rx_init_term rxInitTermParam = {0};
    
    //std::cout << "rx_init_term" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_RX_INIT_TERM, &rxInitTermParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_RX_INIT_TERM failed with:" << ret_val << std::endl;
        return;
    }
    
    if (rxInitTermParam.status != LWL_SUCCESS) {
    }
}


void set_rx_detect()
{
    int ret_val;
    lwlink_set_rx_detect setRxDetectParam = {0};
    
    //std::cout << "set_rx_detect" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_SET_RX_DETECT, &setRxDetectParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_SET_RX_DETECT failed with:" << ret_val << std::endl;
        return;
    }
    
    if (setRxDetectParam.status != LWL_SUCCESS) {
        std::cout << "set_rx_detect failed" << std::endl;
    }
}


void get_rx_detect()
{
    int ret_val;
    lwlink_get_rx_detect getRxDetectParam = {0};
    
    //std::cout << "get_rx_detect" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_GET_RX_DETECT, &getRxDetectParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_GET_RX_DETECT failed with:" << ret_val << std::endl;
        return;
    }
    
    if (getRxDetectParam.status != LWL_SUCCESS) {
        std::cout << "get_rx_detect failed" << std::endl;
    }
}


void calibrate_devices()
{
    int ret_val;
    lwlink_calibrate calibrateParam = {0};
    
    //std::cout << "calibrate_devices" << std::endl;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_CALIBRATE, &calibrateParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_CALIBRATE failed with:" << ret_val << std::endl;
        return;
    }
    
    if (calibrateParam.status != LWL_SUCCESS) {
        std::cout << "calibrate_devices failed" << std::endl;
    }
}

void enable_devices_data()
{
    int ret_val;

    //std::cout << "enable_devices_data" << std::endl;

    lwlink_enable_data enableDataParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_ENABLE_DATA, &enableDataParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_ENABLE_DATA failed with:" << ret_val << std::endl;
        return;
    }
    
    if (enableDataParam.status != LWL_SUCCESS) {
        std::cout << "enable_devices_data failed" << std::endl;
    }
}

void do_initnegotiate()
{
    int ret_val;

    //std::cout << "do_init_negotiate" << std::endl;

    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        lwlink_initnegotiate statusParam;
        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_INITNEGOTIATE, &statusParam );
       
        if (ret_val < 0) {
            std::cout << "ioctl IOCTL_LWLINK_INITNEGOTIATE failed with:" << ret_val << std::endl;
            return;
        }
        if(statusParam.status != LWL_SUCCESS) {
            std::cout << "initnegotiat failed for device index "<< idx << std::endl;
        }
    }
}


void do_link_init()
{
    int ret_val;

    //std::cout << "do_link_init" << std::endl;

    lwlink_link_init_async initParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC, &initParam );
    
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC  failed with:" << ret_val << std::endl;
       // return;
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
            std::cout << "ioctl IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS failed with:" << ret_val << std::endl;
            return;
        }
        /*TODO: make this a verbose print
        std::cout << "NodeId = " << statusParam.devInfo.nodeId << "\n";
        int link_num;
        for(link_num =0 ; link_num < LWLINK_MAX_DEVICE_CONN; link_num++) {
            std::cout << "\tlinkIndex = " << statusParam.linkStatus[link_num].linkIndex << " initStatus = " << statusParam.linkStatus[link_num].initStatus << "\n";
        }
        */
 
    }
}

bool isDuplicateConnection(lwlink_connection_info conn)
{
    LWLinkConnList::iterator it;
    for ( it = connList.begin(); it != connList.end(); it++ ) {
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

void discover_intra_connections()
{
    int ret_val;

    // first initiate discover connection
    uint64 startTime = lwrrent_timestamp();
    lwlink_discover_intranode_conns discoverParam = {0};
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discoverParam );
    if (ret_val < 0) {
        std::cout << "ioctl IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS failed with:" << ret_val << std::endl;
        return;
    }
    //std::cout << "devices found =" << deviceInfo.numDevice << "\n";
    uint64 finishTime = lwrrent_timestamp();
    //std::cout << "discovery took " << finishTime - startTime << " milliseconds" << std::endl;
    
    connList.clear();
    
    // now get the list of connections
    for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
        //lwlink_device_get_intranode_conns getConnParam = {0};
        lwlink_device_get_intranode_conns getConnParam;
        // fill the LWLink Device information
        getConnParam.devInfo.pciInfo.domain = deviceInfo.devInfo[idx].pciInfo.domain;
        getConnParam.devInfo.pciInfo.bus = deviceInfo.devInfo[idx].pciInfo.bus;
        getConnParam.devInfo.pciInfo.device = deviceInfo.devInfo[idx].pciInfo.device;
        getConnParam.devInfo.pciInfo.function = deviceInfo.devInfo[idx].pciInfo.function;
        getConnParam.devInfo.nodeId =  myNodeId;
        ret_val = ioctl( DrvHandle, IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &getConnParam );
        if (ret_val < 0) {
            std::cout << "ioctl IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS failed with:" << ret_val << std::endl;
            return;
        }

        //std::cout << "device=" << idx << "connecttions found =" << getConnParam.numConnections << "\n";
        // move the connections to our list
        for (unsigned int i = 0; i < getConnParam.numConnections; i++) {
            if (!isDuplicateConnection(getConnParam.conn[i])) {
                connList.push_back(getConnParam.conn[i]);
            }
        }
    }

    // dump the connection information
    //std::cout << "Total number of intra-node connections = " << connList.size() << std::endl;
    LWLinkConnList::iterator it = connList.begin();
    int connIdx = 0;
    std::cout << "nodeId\t(d::b:d.f)\tphyId\tlinkIndex\tnodeIdFar\t(d::b:d.f)Far\tphyIdFar\tlinkIndexFar\n";
    while ( it != connList.end() ) {
        lwlink_connection_info connInfo = *it;
        connIdx++;
        std::cout << connInfo.srcEndPoint.nodeId;
        std::cout << "\t(" << (int)connInfo.srcEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.srcEndPoint.pciInfo.function <<")";
        std::cout << "\t" << getBusToPhyId(connInfo.srcEndPoint.nodeId, (int) connInfo.srcEndPoint.pciInfo.bus);
        std::cout << "\t" << connInfo.srcEndPoint.linkIndex;
        std::cout << "\t" << connInfo.dstEndPoint.nodeId;
        std::cout << "\t(" << (int)connInfo.dstEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.dstEndPoint.pciInfo.function<<")";
        std::cout << "\t" << getBusToPhyId(connInfo.srcEndPoint.nodeId, (int) connInfo.dstEndPoint.pciInfo.bus);
        std::cout << "\t" << connInfo.dstEndPoint.linkIndex;
        std::cout << std::endl;

        it++;
    }
}

void train_intra_connection(lwlink_conn_train_type trainTo, unsigned int connIdx)
{

    int ret_val;

    // first get the required connection from index
    if (connList.size() < connIdx) {
        std::cout <<"Invalid connection index " << std::endl;
        return;
    }
    
    LWLinkConnList::iterator it = connList.begin();
    std::advance(it, connIdx);
    lwlink_connection_info connInfo = *it;

    std::cout << "\tTraining the following connection \n";
    std::cout << " \tnodeId  " << connInfo.srcEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.srcEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
    std::cout << "<======>";
    std::cout << " nodeId = " << connInfo.dstEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.dstEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
    std::cout << std::endl;
    

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
        std::cout << "Train Connection ioctl IOCTL_LWLINK_TRAIN_INTRANODE_CONN failed with:" << trainParam.status << std::endl;
        return;
    }

    std::cout << "\tSrc linkMode " << getMainLinkStateString(trainParam.srcEndState.linkMode);
    std::cout << " Src txSubLinkMode " << getTxSubLinkStateString(trainParam.srcEndState.txSubLinkMode);
    std::cout << " Src rxSubLinkMode " << getRxSubLinkStateString(trainParam.srcEndState.rxSubLinkMode);
    std::cout << std::endl;    
    std::cout << "\tDst linkMode " << getMainLinkStateString(trainParam.dstEndState.linkMode);
    std::cout << " Dst txSubLinkMode " << getTxSubLinkStateString(trainParam.dstEndState.txSubLinkMode);
    std::cout << " Dst rxSubLinkMode " << getRxSubLinkStateString(trainParam.dstEndState.rxSubLinkMode);
    std::cout << std::endl;
}

void train_all_intra_connections(lwlink_conn_train_type trainTo)
{
    LWLinkConnList::iterator it = connList.begin();
    unsigned int connIdx = 0;
    uint64 startTime = lwrrent_timestamp();
    while ( it != connList.end() ) {
        train_intra_connection(trainTo, connIdx);
        connIdx++;
        it++;
    }
    uint64 finishTime = lwrrent_timestamp();
    std::cout << "train_all_connections took " << finishTime - startTime << " milliseconds" << std::endl;
}

void set_mainlink_state(lwlink_link_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint)
{
    int ret_val;
    lwlink_train_internode_conn_link mainLinkParam;
    
    mainLinkParam.trainTo = trainTo;
    mainLinkParam.isMasterEnd = isMasterEnd;
    mainLinkParam.localEndPoint = localEndPoint;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, &mainLinkParam );
    if (mainLinkParam.status != LWL_SUCCESS) {
        std::cout << "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK ioctl failed with:" << mainLinkParam.status << std::endl;
        //return;
    }

    std::cout << "\tlinkMode " << getMainLinkStateString(mainLinkParam.localEndState.linkMode);
    std::cout << " txSubLinkMode " << getTxSubLinkStateString(mainLinkParam.localEndState.txSubLinkMode);
    std::cout << " rxSubLinkMode " << getRxSubLinkStateString(mainLinkParam.localEndState.rxSubLinkMode);
    std::cout << std::endl;
}

void set_sublink_state(lwlink_sublink_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint)
{
    int ret_val;
    lwlink_train_internode_conn_sublink subLinkParam;
    
    subLinkParam.trainTo = trainTo;
    subLinkParam.isMasterEnd = isMasterEnd;
    subLinkParam.localEndPoint = localEndPoint;

    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, &subLinkParam );
    if (subLinkParam.status != LWL_SUCCESS) {
        std::cout << "IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK ioctl failed with:" << subLinkParam.status << std::endl;
        //return;
    }
    
    std::cout << "\tlinkMode " << getMainLinkStateString(subLinkParam.localEndState.linkMode);
    std::cout << " txSubLinkMode " << getTxSubLinkStateString(subLinkParam.localEndState.txSubLinkMode);
    std::cout << " rxSubLinkMode " << getRxSubLinkStateString(subLinkParam.localEndState.rxSubLinkMode);
    std::cout << std::endl;
}

void write_discovery_tokens(DiscoveryTokenList &writeList)
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
            std::cout << "IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS ioctl failed with:" << writeParam.status << std::endl;
            //return;
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
            writeList.push_back(info);
        }
    }
}

void read_discovery_tokens(DiscoveryTokenList &readList)
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
            std::cout << "IOCTL_LWLINK_READ_DISCOVERY_TOKENS ioctl failed with:" << readParam.status << std::endl;
            std::cout << "NodeID = " << myNodeId << " DBDF = " << deviceInfo.devInfo[dev_num].pciInfo.domain << ":" << deviceInfo.devInfo[dev_num].pciInfo.bus << ":" << deviceInfo.devInfo[dev_num].pciInfo.device << ":" << deviceInfo.devInfo[dev_num].pciInfo.function << "\n";
            //return;
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
            readList.push_back(info);
        }
    }
}

void add_internode_connections(lwlink_endpoint &localEndPoint, lwlink_remote_endpoint_info &remoteEndPoint)
{
    int ret_val;
    lwlink_add_internode_conn addParam;
    
    addParam.localEndPoint = localEndPoint;
    addParam.remoteEndPoint = remoteEndPoint;
    
    ret_val = ioctl( DrvHandle, IOCTL_LWLINK_ADD_INTERNODE_CONN, &addParam );
    if (addParam.status != LWL_SUCCESS) {
        std::cout << "IOCTL_LWLINK_ADD_INTERNODE_CONNECTION ioctl failed with:" << addParam.status << std::endl;
    }
}

std::string getMainLinkStateString(LwU32 linkMode)
{
    printf("linkMode = %d\n", linkMode);
    switch (linkMode) {
        case LWLINK_LINKSTATE_OFF:
            return "Init/Off";
        case LWLINK_LINKSTATE_HS:
            return "Active";
        //case lwlink_link_mode_swcfg:
        case LWLINK_LINKSTATE_SAFE:
            return "Swcfg";
        case LWLINK_LINKSTATE_FAULT:
            return "Faulty";
        case LWLINK_LINKSTATE_RECOVERY:
            return "Recovery";
        case LWLINK_LINKSTATE_FAIL:
            return "Fail";
        case LWLINK_LINKSTATE_DETECT:
            return "Detect";
        case LWLINK_LINKSTATE_RESET:
            return "Reset";
        case LWLINK_LINKSTATE_ENABLE_PM:
            return "Enable PM";
        case LWLINK_LINKSTATE_DISABLE_PM:
            return "Disable PM";
        case LWLINK_LINKSTATE_TRAFFIC_SETUP:
            return "Setup Traffic";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string getTxSubLinkStateString(LwU32 txSubLinkMode)
{
    printf("txSubLinkMode = %d\n", txSubLinkMode);
    switch (txSubLinkMode) {
        case LWLINK_SUBLINK_STATE_TX_HS:
            return "High Speed";
        case LWLINK_SUBLINK_STATE_TX_SINGLE_LANE:
            return "Single Lane";
        case LWLINK_SUBLINK_STATE_TX_TRAIN:
            return "Training";
        case LWLINK_SUBLINK_STATE_TX_SAFE:
            return "Safe";
        case LWLINK_SUBLINK_STATE_TX_OFF:
            return "Off";
        case LWLINK_SUBLINK_STATE_TX_COMMON_MODE:
            return "Common Mode Enable";
        case LWLINK_SUBLINK_STATE_TX_COMMON_MODE_DISABLE:
            return "Common Mode Disable";
        case LWLINK_SUBLINK_STATE_TX_DATA_READY:
            return "Data Ready";
        case LWLINK_SUBLINK_STATE_TX_EQ:
            return "Equalization";
        case LWLINK_SUBLINK_STATE_TX_PRBS_EN:
            return "PRBS Generator";
        case LWLINK_SUBLINK_STATE_TX_POST_HS:
            return "Post Active HW Retraining";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

std::string getRxSubLinkStateString(LwU32 rxSubLinkMode)
{
    printf("rxSubLinkMode = %d\n", rxSubLinkMode);
    switch (rxSubLinkMode) {
        case LWLINK_SUBLINK_STATE_RX_HS:
            return "High Speed";
        case LWLINK_SUBLINK_STATE_RX_SINGLE_LANE:
            return "Single Lane";
        case LWLINK_SUBLINK_STATE_RX_TRAIN:
            return "Training";
        case LWLINK_SUBLINK_STATE_RX_SAFE:
            return "Safe";
        case LWLINK_SUBLINK_STATE_RX_OFF:
            return "Off";
        case LWLINK_SUBLINK_STATE_RX_RXCAL:
            return "Calibration";
    }

    // no switch case matched. shouldn't happen
    return "Unknown";
}

