#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <syslog.h>
#include <set>
#include <unistd.h>

#include "lwos.h"
#include "lwswitch_audit_node.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_tables.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_paths.h"
#include "lwswitch_audit_dev.h"

/*Compute the number of paths programmed between each pair of GPUs in the system*/
bool naComputePaths(naNodeRequestTables_t &req_tables,  naNodeResponseTables_t &res_tables,
                    int paths_matrix[][MAX_GPU], int num_switches)
{
    int switch_id;
    //far each switch
    for(int dev_id = 0; dev_id < num_switches ; dev_id++)
    {
        switch_id = naGetDevToSwitchID(dev_id);
        //for each access ports
        for(int i = 0; i < NUM_ACCESS_PORTS; i++)
        {
            uint32_t switch_port = getNthAccessPort(switch_id, i);
            int src_gpu_id = getConnectedGPUID(switch_id, switch_port);
            //for each destination dpu
            for (int dest_gpu_id = 0; dest_gpu_id < MAX_GPU; dest_gpu_id++)
            {
                int egress_switch_id;
                int egress_port_id;
                PRINT_VERBOSE("Computing path from GPU=%d to GPU=%d via LWSwitch=%d LWSwitch_port=%d\n",
                                src_gpu_id, dest_gpu_id, switch_id, switch_port);
                //Compute the egress switch/port for a request packet headed to this dest_gpu_id
                if(naComputeReqEgressPort(req_tables, switch_id, switch_port, dest_gpu_id,
                                            egress_switch_id, egress_port_id) == false)
                {
                    paths_matrix[src_gpu_id][dest_gpu_id] == PATH_ERROR;
                    continue;
                }
                if(paths_matrix[src_gpu_id][dest_gpu_id] == PATH_ERROR)
                {
                    continue;
                }
                else if(dest_gpu_id == src_gpu_id)
                {
                    if(egress_port_id != DEST_UNREACHABLE)
                    {
                        paths_matrix[src_gpu_id][dest_gpu_id] = PATH_ERROR;
                        PRINT_ERROR_VERBOSE("Bad Request path: Path to self found\n");
                    }
                    else
                    {
                        PRINT_VERBOSE("\tCorrect Request path: no path to self\n");
                    }
                }
                else if (egress_port_id == DEST_UNREACHABLE)
                {
                    PRINT_VERBOSE("\tNo Request path found!\n");
                }
                else
                {
                    if(naCheckReqEgressPort(dest_gpu_id, egress_switch_id, egress_port_id))
                    {
                        int res_egress_switch_id=-1;
                        int res_egress_port_id=DEST_UNREACHABLE;
                        int src_req_link_id = computeReqLinkID(switch_id, switch_port);
                        //compute the switch/port to which the response reaches back
                        naComputeResEgressPort(res_tables, egress_switch_id, egress_port_id,
                                            src_req_link_id, res_egress_switch_id, res_egress_port_id);
                        //If response reaches back to source port a path is correct
                        if((res_egress_switch_id == switch_id) && (res_egress_port_id == (int)switch_port)) {
                            paths_matrix[src_gpu_id][dest_gpu_id] += 1;
                        } else {
                            paths_matrix[src_gpu_id][dest_gpu_id] = PATH_ERROR;
                            if(res_egress_port_id == DEST_UNREACHABLE)
                            {
                                PRINT_ERROR_VERBOSE("Bad Response path: Path from LWSwitch=%d port_id=%d to dest_gpu_id=%d \n"
                                                "\t\tleads back to incorrect Return LWSwitch=%d Return port ID=unreachable\n",
                                                switch_id, switch_port, dest_gpu_id, res_egress_switch_id);
                            }
                            else
                            {
                                PRINT_ERROR_VERBOSE("Bad Response path: Path from LWSwitch=%d port_id=%d to dest_gpu_id=%d \n"
                                                "\t\tleads back to incorrect Return LWSwitch=%d Return port ID=%d\n",
                                                switch_id, switch_port, dest_gpu_id, res_egress_switch_id, res_egress_port_id);
                            }
                        }
                    }
                    else
                    {
                        paths_matrix[src_gpu_id][dest_gpu_id] = PATH_ERROR;
                        PRINT_ERROR_VERBOSE("Bad Request path: Path from LWSwitch=%d port_id=%d to dest_gpu_id=%d\n"
                                            "\t\tleads to incorrect GPU egress_GPU_id=%d\n",
                                            switch_id, switch_port, dest_gpu_id,
                                            getConnectedGPUID(egress_switch_id, egress_port_id));


                    }
                }
            }
        }
    }
    return true;
}

/*
Print all the src GPU to dst GPU paths computed previously as a matrix.
Each element of the matrix represents the number of paths from src GOU to dest GPU.
Note that values for (A,B) and (B,A) can be different
*/
void naPrintPaths( char *heading, int paths_matrix[][MAX_GPU], bool is_csv)
{
    char *separator;
    std::set<int> gpu_set;
    for(int i = 0; i < MAX_GPU; i++)
        for(int j = 0; j < MAX_GPU; j++)
            if((i != j) && (paths_matrix[i][j] != 0))
            {
                gpu_set.insert(i);
                gpu_set.insert(j);
            }
    //If CVS option is set the separator is a comma and heading is not printed
    if(is_csv)
    {
        separator = ",";
    }
    else
    {
        printf("\n%s\n", heading);
        separator = "";
    }

    printf("GPU");
    for(std::set<int>::iterator it = gpu_set.begin(); it != gpu_set.end(); it++)
    {
        if(is_csv)
            printf("%s%d", separator, *it);
        else
            printf("%s%3d", separator, *it);
    }
    printf("\n");
    for(std::set<int>::iterator it_x = gpu_set.begin(); it_x != gpu_set.end(); it_x++)
    {
        if(is_csv)
            printf("%d", *it_x);
        else
            printf("%3d", *it_x);

        for(std::set<int>::iterator it_y = gpu_set.begin(); it_y != gpu_set.end(); it_y++)
        {
            if((paths_matrix[*it_x][*it_y] == 0) && (*it_x == *it_y))
                if(is_csv)
                    printf("%sX", separator);
                else
                    printf("%s  X", separator);
            else
                if(is_csv)
                    printf("%s%d", separator, paths_matrix[*it_x][*it_y]);
                else
                    printf("%s%3d", separator, paths_matrix[*it_x][*it_y]);
        }
        printf("\n");
    }
}

#ifdef DEBUG
typedef struct debug_rlid {
    int switch_id;
    int switch_port;
    int rlid;
}debugRLID_t;
debugRLID_t dbgRLID = {-1, -1, -1};
//set Requestor link ID
void naSetRLID(int switch_id, int switch_port, int rlid)
{
    dbgRLID.switch_id = switch_id;
    dbgRLID.switch_port = switch_port;
    dbgRLID.rlid = rlid;
}
#endif

/*Verify whether requestor link IDs set in switches is correct
num_switches: Number of switch instances in /dev
Return: Number of ports programmed with wrong requestor link IDs
*/
int naCheckReqLinkIDs(int num_switches, int paths_matrix[][MAX_GPU])
{

    int switch_id;
    int req_link_id;
    int bad_link_ids=0;
    for(int dev_id = 0; dev_id < num_switches ; dev_id++) 
    {

        int fd; 
        if((fd = naOpenSwitchDev(dev_id)) == -1)
        {
            PRINT_ERROR_VERBOSE("Unable to open LWSwitch device %d errno=%d\n", dev_id, errno);
            continue;
        }
        switch_id = naGetDevToSwitchID(dev_id);

        uint64_t port_mask;
        if(naReadPortMask(fd, port_mask) == false)
        {
            PRINT_ERROR_VERBOSE("Unable to read port mask for LWSwitch device %d errno=%d\n", dev_id, errno);
            continue;;
        }

        
        //for all access ports
        for(int i = 0; i < NUM_ACCESS_PORTS; i++)
        {
            uint32_t switch_port = getNthAccessPort(switch_id, i);
            if(!naIsPortEnabled(port_mask, switch_port))
                continue;
            if((req_link_id = naReadReqLinkId(fd, switch_port)) == -1)
            {
                PRINT_ERROR_VERBOSE("Unable to read Requestor Link ID for LWSwitch=%d Port=%d\n", switch_id, switch_port);
                bad_link_ids++;
                continue;
            } 
#ifdef DEBUG
            //over-ride requestor link ID for testing purposes
            if((dbgRLID.switch_id == switch_id) && (dbgRLID.switch_port == (int)switch_port))
                req_link_id = dbgRLID.rlid;
#endif
            if(req_link_id != computeReqLinkID(switch_id, switch_port)) 
            {
                int conn_gpu_id = getConnectedGPUID(switch_id, switch_port);
                for(int gpu_id = 0; gpu_id < MAX_GPU; gpu_id++)
                        if(paths_matrix[conn_gpu_id][gpu_id] != 0)
                            paths_matrix[conn_gpu_id][gpu_id] = PATH_ERROR; 
                
                PRINT_ERROR_VERBOSE("Requestor Link ID for LWSwitch=%d Port=%d Connected GPU=%d set incorectly to %d correct value=%d\n", switch_id, switch_port, conn_gpu_id, req_link_id, computeReqLinkID(switch_id, switch_port));
                bad_link_ids++;
            }
            else
            {
                PRINT_VERBOSE("Requestor Link ID for LWSwitch=%d Port=%d set correctly value=%d\n", switch_id, switch_port, req_link_id);
            } 
        }
        close(fd);
    }
    return bad_link_ids;
}


