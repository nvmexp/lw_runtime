
#include "socket_interface.h"
#include "lwlink_lib_ioctl.h"
#include "helper.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>


void run_step(int step)
{
    //std::cout << "Run step " << step << "\n";
    switch (step) {
        case NT_STEP_EXIT: {
            std::cout << " Exiting \n";
            exit(0);
        }
        case NT_ENABLE_COMMON_MODE: {
            enable_devices_common_mode();
            break;
        }
        case NT_CALIBRATE_LINKS: {
            calibrate_devices();
            break;
        }
        case NT_RX_INIT_TERM: {
            rx_init_term();
            break;
        }
        case NT_SET_RX_DETECT: {
            set_rx_detect();
            break;
        }
        case NT_GET_RX_DETECT: {
            get_rx_detect();
            break;
        }
        case NT_SET_INITPHASE1: {
            set_initphase1();
            break;
        }
        case NT_DISABLE_COMMON_MODE: {
            disable_devices_common_mode ();
            break;
        }
        case NT_ENABLE_DEVICE_DATA: {
            enable_devices_data();
            break;
        }
        case NT_SET_INITPHASE5: {
            set_initphase5();
            break;
        }
        case NT_DO_LINK_INITIALIZATION: {
            do_link_init();
            break;
        }
        case NT_DO_INITNEGOTIATE: {
            do_initnegotiate();
            break;
        }
        case NT_DO_ALL_THE_INITIALIZATION_IN_SINGLE_STEP: {
            uint64 startTime = lwrrent_timestamp();
            enable_devices_common_mode();
            calibrate_devices();
            disable_devices_common_mode();
            enable_devices_data();
            set_initphase5();
            do_link_init();
            uint64 finishTime = lwrrent_timestamp();
            //std::cout << "all initialization took " << finishTime - startTime << " milliseconds" << std::endl;
            break;
        }
        case NT_DISCOVER_CONNECTIONS_INTRA_NODE: {
            discover_intra_connections();
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_HIGH_SPEED: {
            int connIdx;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            uint64 startTime = lwrrent_timestamp();
            train_intra_connection(lwlink_train_conn_swcfg_to_active, connIdx);
            uint64 finishTime = lwrrent_timestamp();
            std::cout << "Single Tain took " << finishTime - startTime << " milliseconds" << std::endl;
            
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_OFF: {
            unsigned int connIdx;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            train_intra_connection(lwlink_train_conn_to_off, connIdx);
            break;
        }
        case NT_TRAIN_AN_INTRA_NODE_CONNECTION_TO_SAFE: {
            unsigned int connIdx;
            std::cout << "Enter the connection idx to train ";
            std::cin >> connIdx; 
            train_intra_connection(lwlink_train_conn_off_to_swcfg, connIdx);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_HIGH_SPEED: {
            train_all_intra_connections(lwlink_train_conn_swcfg_to_active);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_OFF: {
            train_all_intra_connections(lwlink_train_conn_to_off);
            break;
        }
        case NT_TRAIN_ALL_INTRA_NODE_CONNECTIONS_TO_SAFE: {
            train_all_intra_connections(lwlink_train_conn_off_to_swcfg);
            break;
        }
        case NT_SHOW_MULTI_NODE_TRAINING_OPTIONS: {
            show_multi_node_training_options();
            break;
        }
        case NT_GET_DEVICE_INFORMATION: {
            get_device_information();
            break;
        }
        default : {
            std::cout << "Invalid training step " << step << "\n";
        }
    }
}

void run_training_steps(ntCmdParser_t *pCmdParser)
{
    if( pCmdParser->mIsServer ) {
        
        discover_intra_node_connections_all_steps();
        run_multi_node_server();

    } else if (pCmdParser->mIsClient) {
        
        discover_intra_node_connections_all_steps();
        run_multi_node_client(pCmdParser->mIpAddress);
    
    } else {
        //Run all the steps specified on the command line
        //discover_connections_all_steps();
        std::string str(pCmdParser->mTrainSteps);
        std::vector<int> vect;

        std::stringstream ss(str);

        unsigned int i;

        while (ss >> i) {
            vect.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }

        for (i=0; i< vect.size(); i++) {
            //std::cout << vect.at(i)<< "\n";
            run_step(vect.at(i));
        }
    }
    return;
}

void list_steps()
{
    std::cout <<" 0. Exit \n";
    std::cout <<" 1. Set Initphase1 \n";
    std::cout <<" 2. Rx Init Term \n";
    std::cout <<" 3. Set Rx Detect \n";
    std::cout <<" 4. Get Rx Detect \n";
    std::cout <<" 5. Enable Common Mode \n";
    std::cout <<" 6. Calibrate Links \n";
    std::cout <<" 7. Disable Common Mode \n";
    std::cout <<" 8. Enable Device Data \n";
    std::cout <<" 9. Set Initphase5 \n";
    std::cout <<" 10. Do Link Initialization \n";
    std::cout <<" 11. Do all the Initialization in Single Step \n";
    std::cout <<" 12. Discover Connections - Intra-Node \n";
    std::cout <<" 13. Train an Intra-Node Connection to High Speed \n";
    std::cout <<" 14. Train an Intra-Node Connection to OFF \n";
    std::cout <<" 15. Train an Intra-Node Connection to SAFE \n";
    std::cout <<" 16. Train ALL Intra-Node Connections to High Speed \n";
    std::cout <<" 17. Train ALL Intra-Node Connections to OFF \n";
    std::cout <<" 18. Train ALL Intra-Node Connections to SAFE \n";
    std::cout <<" 19. Show multi-node training options \n";
    std::cout <<" 20. Get device information \n";
    std::cout <<" 21. Do Initnegotiate \n";

}

void show_train_menu()
{
    int option;

    while (1) {
        list_steps();

        std::cout << "Enter your choice ";
        std::cin >> option; 

        run_step(option);
    }
}

void discover_intra_node_connections_all_steps()
{
    uint64 startTime = lwrrent_timestamp();
    set_initphase1();
    rx_init_term();
    set_rx_detect();
    get_rx_detect();
    enable_devices_common_mode();
    calibrate_devices();
    disable_devices_common_mode();
    enable_devices_data();
    set_initphase5();
    do_link_init();
    do_initnegotiate();
    discover_intra_connections();
}
