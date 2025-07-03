
#include "master.h"
#include "lwlink_train_cmd_parser.h"
#include "lwlink_lib_ioctl.h"
#include "helper.h"
#include "logging.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>


bool stop;

void run_step(int step)
{
    //PRINT_VERBOSE << "Run step " << step << "\n";
    switch (step) {
        case NT_STEP_EXIT: {
            sendExitMessage();
            stop = true;
            break;
        }
        case NT_ENABLE_COMMON_MODE: {
            doAndSendEnableCommonMode();
            break;
        }
        case NT_CALIBRATE_LINKS: {
            doAndSendCalibrateDevices();
            break;
        }
        case NT_RX_INIT_TERM: {
            doAndSendRxInitTerm();
            break;
        }
        case NT_SET_RX_DETECT: {
            doAndSendSetRxDetect();
            break;
        }
        case NT_GET_RX_DETECT: {
            doAndSendGetRxDetect();
            break;
        }
        case NT_SET_INITPHASE1: {
            doAndSendSetInitphase1Req();
            break;
        }
        case NT_DISABLE_COMMON_MODE: {
            doAndSendDisableCommonMode();
            break;
        }
        case NT_ENABLE_DEVICE_DATA: {
            doAndSendEnableDevicesData();
            break;
        }
        case NT_SET_INITPHASE5: {
            doAndSendSetInitphase5Req();
            break;
        }
        case NT_DO_LINK_INITIALIZATION: {
            doAndSendDoLinkInit();
            break;
        }
        case NT_DO_INITNEGOTIATE: {
            doAndSendDoInitNegotiate();
            break;
        }
        case NT_DO_ALL_THE_INITIALIZATION_IN_SINGLE_STEP: {
            uint64 startTime = lwrrent_timestamp();
            doAndSendSetInitphase1Req();
            doAndSendRxInitTerm();
            doAndSendSetRxDetect();
            doAndSendGetRxDetect();
            doAndSendEnableCommonMode();
            doAndSendCalibrateDevices();
            doAndSendDisableCommonMode();
            doAndSendEnableDevicesData();
            doAndSendSetInitphase5Req();
            doAndSendDoLinkInit();
            doAndSendDoInitNegotiate();
            uint64 finishTime = lwrrent_timestamp();
            PRINT_VERBOSE << "all initialization took " << finishTime - startTime << " milliseconds" << std::endl;
            break;
        }
        case NT_DISCOVER_CONNECTIONS_INTRA_NODE: {
            doAndSendDiscoverIntraConnections();
            getAllIntraConnections();
            printIntraNodeConns();
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_HIGH_SPEED: {
            int connIdx;
            int numNode; 
            std::cout << "Enter the node ";
            std::cin >> numNode;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            uint64 startTime = lwrrent_timestamp();
            doTrainIntraConnection(lwlink_train_conn_swcfg_to_active, numNode, connIdx);
            uint64 finishTime = lwrrent_timestamp();
            std::cout << "Single Tain took " << finishTime - startTime << " milliseconds" << std::endl;
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_OFF: {
            unsigned int connIdx;
            int numNode; 
            std::cout << "Enter the node ";
            std::cin >> numNode;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            doTrainIntraConnection(lwlink_train_conn_to_off, numNode, connIdx);
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_SAFE: {
            unsigned int connIdx;
            int numNode; 
            std::cout << "Enter the node ";
            std::cin >> numNode;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            doTrainIntraConnection(lwlink_train_conn_off_to_swcfg, numNode, connIdx);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_HIGH_SPEED: {
            doAndSendIntraNodeTraining(lwlink_train_conn_swcfg_to_active);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_OFF: {
            doAndSendIntraNodeTraining(lwlink_train_conn_to_off);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_SAFE: {
            doAndSendIntraNodeTraining(lwlink_train_conn_off_to_swcfg);
            break;
        }
        case NT_GET_DEVICE_INFORMATION: {
            doAndSendDeviceInfoReq();
            printDeviceInfo();
            break;
        }
        case NT_DISCOVER_CONNECTIONS_INTER_NODE: {
            discoverInterNodeConnections();
            displayInterNodeConnections();
            break;
        }
        case NT_ADD_INTERNODE_CONNECTIONS_ON_ALL_NODES: {
            addInterNodeConnections();
            break;
        }
        case NT_TRAIN_INTERNODE_CONNECTIONS_TO_HIGH: {
            doInterNodeTraining(lwlink_train_conn_swcfg_to_active);
            break;
        }
        default : {
            PRINT_VERBOSE << "Invalid training step " << step << "\n";
        }
    }
}

void list_steps()
{
    std::cout <<" 0. Exit \n";
    std::cout <<" 1. Get device information \n";
    std::cout <<" 2. Set Initphase1 \n";
    std::cout <<" 3. Rx Init Term \n";
    std::cout <<" 4. Set Rx Detect \n";
    std::cout <<" 5. Get Rx Detect \n";
    std::cout <<" 6. Enable Common Mode \n";
    std::cout <<" 7. Calibrate Links \n";
    std::cout <<" 8. Disable Common Mode \n";
    std::cout <<" 9. Enable Device Data \n";
    std::cout <<" 10. Set Initphase5 \n";
    std::cout <<" 11. Do Link Initialization \n";
    std::cout <<" 12. Do Initnegotiate \n";
    std::cout <<" 13. Do all the Initialization in Single Step \n";
    std::cout <<" 14. Discover Connections - Intra-Node \n";
    std::cout <<" 15. Train an Intra-Node Connection to High Speed \n";
    std::cout <<" 16. Train an Intra-Node Connection to OFF \n";
    std::cout <<" 17. Train an Intra-Node Connection to SAFE \n";
    std::cout <<" 18. Train ALL Intra-Node Connections to High Speed \n";
    std::cout <<" 19. Train ALL Intra-Node Connections to OFF \n";
    std::cout <<" 20. Train ALL Intra-Node Connections to SAFE \n";
    std::cout <<" 21. Discover Internode Connections \n";
    std::cout <<" 22. Add Discovered Internode Connections on all nodes \n";
    std::cout <<" 23. Train all internode connection to HS \n";
}

void show_menu(int nodes)
{
    int option;
    stop = false;
    setNumNodes(nodes);
    while (!stop) {
        list_steps();

        std::cout << "Enter your choice ";
        std::cin >> option; 

        run_step(option);
    }
}

void discover_intra_node_connections_all_steps()
{
    uint64 startTime = lwrrent_timestamp();
    doAndSendSetInitphase1Req();
    doAndSendRxInitTerm();
    doAndSendSetRxDetect();
    doAndSendGetRxDetect();
    doAndSendEnableCommonMode();
    doAndSendCalibrateDevices();
    doAndSendDisableCommonMode();
    doAndSendEnableDevicesData();
    doAndSendSetInitphase5Req();
    doAndSendDoLinkInit();
    doAndSendDoInitNegotiate();
    doAndSendDiscoverIntraConnections();
    getAllIntraConnections();
    printIntraNodeConns();
    uint64 endTime = lwrrent_timestamp();
}
