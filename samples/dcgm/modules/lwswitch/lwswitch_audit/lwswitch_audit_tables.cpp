#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <syslog.h>
#include <unistd.h>
#include <vector>

#include "lwos.h"
#include "lwswitch_audit_node.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_tables.h"
#include "lwswitch_audit_dev.h"
#include "lwswitch_audit_logging.h"

#define PORT_VALID   0x1
//Get the valid nibble from a 32-bit integer
static int getValidNibble(uint32_t num, int max_nibble, int &valid_n, int base)
{
    int num_valid_nibbles_found = 0;

    for(int i = 0; i < max_nibble; i++) 
    {
        if((num >> (i * 4)) & PORT_VALID) {
            valid_n = i + base;
            num_valid_nibbles_found++;
        }
    }
    return num_valid_nibbles_found;
}
//TODO: In some platforms the packets can be sprayed to multiple destinations.
//get switch egress port ID for which valid request entry has been set
static int getPortFromRequestBitmap(LWSWITCH_INGRESS_REQUEST_ENTRY &entry)
{
    int total_valid_nibbles = 0;
    int valid_n = -1;

    total_valid_nibbles += getValidNibble(entry.vcModeValid7_0, 8, valid_n, 0);
    total_valid_nibbles += getValidNibble(entry.vcModeValid15_8, 8, valid_n, 8);
    total_valid_nibbles += getValidNibble(entry.vcModeValid17_16, 2, valid_n, 16);

    if (total_valid_nibbles != 1) 
    {
        return DEST_ERROR;
    } 
    else
    {
        return valid_n;
    }
}            

//TODO: In some platforms the packets can be sprayed to multiple destinations.
//get switch egress port ID for which valid response entry has been set
static int getPortFromResponseBitmap(LWSWITCH_INGRESS_RESPONSE_ENTRY &entry)
{
    int total_valid_nibbles = 0;
    int valid_n = -1;

    total_valid_nibbles += getValidNibble(entry.vcModeValid7_0, 8, valid_n, 0);
    total_valid_nibbles += getValidNibble(entry.vcModeValid15_8, 8, valid_n, 8);
    total_valid_nibbles += getValidNibble(entry.vcModeValid17_16, 2, valid_n, 16);

    if (total_valid_nibbles != 1) 
    {
        return DEST_ERROR;
    } 
    else
    {
        return valid_n;
    }
}

/*Read one request table
fd: file descriptor to opened switch device
switch_port: switch port 
Return: true=Success false=unable to read request table
*/ 
static bool readRequestTable(int fd, uint32_t switch_port, naPortRequestTable_t &req_table, 
                             uint32_t num_table_entries, int &valid_out_of_range_req_entry)
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switch_port;
    ioctlParams.firstIndex = 0;
    valid_out_of_range_req_entry=0;
    while (ioctlParams.nextIndex < num_table_entries) 
    {
        if(ioctl(fd, IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE, &ioctlParams) == -1) 
        {
            fprintf(stderr, "ioctl failed to read Request Table. errno=%d\n", errno);
            return false;
        } 
        else 
        {
            for (unsigned int n = 0; n < ioctlParams.numEntries; n++)
            {
                uint32_t dest_gpu_id = ioctlParams.entries[n].idx/REQ_ENTRIES_PER_GPU;

                if(ioctlParams.entries[n].idx % REQ_ENTRIES_PER_GPU)
                {
                    //for 2nd, 3rd and 4th entries to GPU
                    if(req_table[dest_gpu_id].egress_port_id == DEST_ERROR)
                    {
                        continue;
                    }
                    else if(ioctlParams.entries[n].entry.entryValid == true) 
                    {
                        if(dest_gpu_id >= MAX_GPU)
                        {
                            valid_out_of_range_req_entry++;
                            PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                                "\tDest GPU %d \tEgress Port %d\n",
                                                dest_gpu_id, getPortFromRequestBitmap(ioctlParams.entries[n].entry));
                        }
                        else if((req_table[dest_gpu_id].egress_port_id == DEST_UNREACHABLE) || 
                                (getPortFromRequestBitmap(ioctlParams.entries[n].entry) != req_table[dest_gpu_id].egress_port_id))
                            req_table[dest_gpu_id].egress_port_id = DEST_ERROR;
                        else
                            req_table[dest_gpu_id].count++;
                    
                    }
                    else if ((dest_gpu_id  < MAX_GPU) && (req_table[dest_gpu_id].egress_port_id != DEST_UNREACHABLE))
                    {
                        req_table[dest_gpu_id].egress_port_id = DEST_ERROR;
                    }
                }
                else
                {
                    //for First entry to GPU
                    if(ioctlParams.entries[n].entry.entryValid == true)
                    {
                        uint32_t egress_port = getPortFromRequestBitmap(ioctlParams.entries[n].entry);
                        if(dest_gpu_id >= MAX_GPU)
                        {
                            valid_out_of_range_req_entry++;
                            PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                                "\tDest GPU %d \tEgress Port %d\n", dest_gpu_id, egress_port);
                            continue;
                        }
                        PRINT_VERBOSE("\tDest GPU %d \tEgress Port %d\n", dest_gpu_id, egress_port);
                        req_table[dest_gpu_id].egress_port_id = egress_port;
                        req_table[dest_gpu_id].count = 1;
                    } 
                    else
                    {
                        PRINT_VERBOSE("\tDest GPU %d \tNO Egress port\n", dest_gpu_id);
                    }
                }
            }

        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }
    //Verify if all entries to the same destination GPU point to the same egress port
    for(uint32_t dest_gpu_id = 0; dest_gpu_id < MAX_GPU; dest_gpu_id++)
    {
        if((req_table[dest_gpu_id].count > 0) && (req_table[dest_gpu_id].count < REQ_ENTRIES_PER_GPU))
        {
            req_table[dest_gpu_id].egress_port_id = DEST_ERROR;
            PRINT_ERROR_VERBOSE("\tNot all entries for addresses belonging to same GPU are valid\n");
        }
    }
    return true;
}

/*Read one response table
fd: file descriptor to opened switch device
switch_port: switch port 
Return: true=Success false=unable to read response table
*/ 
static bool readResponseTable(int fd, uint32_t switch_port, naPortResponseTable_t &res_table, 
                              uint32_t num_table_entries, int &valid_out_of_range_res_entry)
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switch_port;
    ioctlParams.firstIndex = 0;
    valid_out_of_range_res_entry=0;
    while (ioctlParams.nextIndex < num_table_entries) 
    {
        if(ioctl(fd, IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE, &ioctlParams) == -1) 
        {
            fprintf(stderr, "ioctl failed to read Response Table. errno=%d\n", errno);
            close(fd);
            return false;
        } 
        else 
        {
            for (unsigned int n = 0; n < ioctlParams.numEntries; n ++) 
            {
                uint32_t dest_gpu_id = ioctlParams.entries[n].idx / MAX_SWITCH_PER_BASEBOARD;
                uint32_t dest_port_id = ioctlParams.entries[n].idx % MAX_SWITCH_PER_BASEBOARD;
                if(ioctlParams.entries[n].entry.entryValid == true) 
                {
                    uint32_t egress_port = getPortFromResponseBitmap(ioctlParams.entries[n].entry);
                    if(dest_gpu_id >= MAX_GPU)
                    {
                        valid_out_of_range_res_entry++;
                        PRINT_ERROR_VERBOSE("\tOut of range Request entry valid\n"
                                            "\tDest GPU %d \tEgress Port %d\n", dest_gpu_id, egress_port);
                    }
                    else
                    {
                        PRINT_VERBOSE("\tDest GPU %d \tDest_port_id %d \tEgress Port %d\n", dest_gpu_id, dest_port_id, egress_port);
                        res_table[ioctlParams.entries[n].idx] = egress_port;
                    }
                } 
                else
                {
                    PRINT_VERBOSE("\tDest GPU %d \tDest_port_id %d \tNO Egress port\n", dest_gpu_id, dest_port_id);
                }
            }
        }
        ioctlParams.firstIndex = ioctlParams.nextIndex;
    }
    return true;
}

//read all request tables for all ports on all switches on node
bool naReadRequestTables(naNodeRequestTables_t &req_tables, int num_switches, uint32_t num_table_entries) 
{

    int switch_id;
    int valid_out_of_range_req_entry;
    for(int dev_id = 0; dev_id < num_switches ; dev_id++) 
    {

        int fd; 
        PRINT_VERBOSE("\n");
        if((fd = naOpenSwitchDev(dev_id)) == -1)
        {
            fprintf(stderr, "Unable to open LWSwitch device %d errno=%d\n", dev_id, errno);
            return false;
        }
        switch_id = naGetDevToSwitchID(dev_id);
        uint64_t port_mask;
        if(naReadPortMask(fd, port_mask) == false)
        {
            return false;
        }
        
        //for all access ports
        for(int i = 0; i < NUM_ACCESS_PORTS; i++)
        {
            uint32_t switch_port = getNthAccessPort(switch_id, i);
            valid_out_of_range_req_entry=0;
            PRINT_VERBOSE("Reading Request Table for LWSwitch=%d Port=%d(%s)\n", switch_id, switch_port, isTrunkPort(switch_port)?"Trunk port":"Access port");
            if(naIsPortEnabled(port_mask, switch_port) 
                && !readRequestTable(fd, switch_port, req_tables[dev_id][switch_port], num_table_entries, valid_out_of_range_req_entry))
            {
                fprintf(stderr, "Unable to read Request Table for LWSwitch=%d Port=%d\n", switch_id, switch_port);
                close(fd);
                return false;
            }
            if(valid_out_of_range_req_entry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Request entries found for GPU IDS greater than %d\n", 
                        switch_id, switch_port, valid_out_of_range_req_entry, MAX_GPU);
            }
        }

        //for all trunk ports
        for(int i = 0; i < NUM_TRUNK_PORTS; i++)
        {
            uint32_t switch_port;
            switch_port = getNthTrunkPort(switch_id, i);
            valid_out_of_range_req_entry=0;

            PRINT_VERBOSE("Reading Request Table for LWSwitch=%d Port=%d(%s)\n", 
                          switch_id, switch_port, isTrunkPort(switch_port)?"Trunk port":"Access port");
            if(naIsPortEnabled(port_mask, switch_port) 
               && !readRequestTable(fd, switch_port, req_tables[dev_id][switch_port], num_table_entries, valid_out_of_range_req_entry))
            {
                fprintf(stderr, "Unable to read Request Table for LWSwitch=%d Port=%d\n", switch_id, switch_port);
                close(fd);
                return false;
            }
            if(valid_out_of_range_req_entry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Request entries found for GPU IDS greater than %d\n", 
                        switch_id, switch_port, valid_out_of_range_req_entry, MAX_GPU);
            }
        }
        close(fd);
    }
    return true;
}

//read all response tables for all ports on all switches on node
bool naReadResponseTables(naNodeResponseTables_t &res_tables, int num_switches, uint32_t num_table_entries) 
{

    int switch_id;
    int valid_out_of_range_res_entry;
    for(int dev_id = 0; dev_id < num_switches; dev_id++) 
    {
        
        int fd; 
        PRINT_VERBOSE("\n");
        if((fd = naOpenSwitchDev(dev_id)) == -1)
        {
            fprintf(stderr, "Unable to open LWSwitch device %d errno=%d\n", dev_id, errno);
            return false;
        }
        switch_id = naGetDevToSwitchID(dev_id);
        uint64_t port_mask;
        if(naReadPortMask(fd, port_mask) == false)
        {
            return false;
        }

        //for all access ports
        for(int i = 0; i < NUM_ACCESS_PORTS; i++)
        {
            uint32_t switch_port = getNthAccessPort(switch_id, i);
            valid_out_of_range_res_entry = 0;
            PRINT_VERBOSE("Reading Response Table for LWSwitch=%d Port=%d(%s)\n", switch_id, switch_port, isTrunkPort(switch_port)?"Trunk port":"Access port");
            if(naIsPortEnabled(port_mask, switch_port) 
               && !readResponseTable(fd, switch_port, res_tables[dev_id][switch_port], num_table_entries, valid_out_of_range_res_entry))
            {
                fprintf(stderr, "Unable to read Response Table for LWSwitch=%d port=%d\n", switch_id, switch_port);
                close(fd);
                return false;
            }
            if(valid_out_of_range_res_entry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Response entries found for GPU IDS greater than %d\n", 
                        switch_id, switch_port, valid_out_of_range_res_entry, MAX_GPU);
            }

        }

        //for all trunk ports
        for(int i = 0; i < NUM_TRUNK_PORTS; i++)
        {
            uint32_t switch_port;
            switch_port = getNthTrunkPort(switch_id, i);
            valid_out_of_range_res_entry = 0;

            PRINT_VERBOSE("Reading Response Table for LWSwitch=%d Port=%d(%s)\n", switch_id, switch_port, isTrunkPort(switch_port)?"Trunk port":"Access port");
            if(naIsPortEnabled(port_mask, switch_port) 
               && !readResponseTable(fd, switch_port, res_tables[dev_id][switch_port], num_table_entries, valid_out_of_range_res_entry))
            {
                fprintf(stderr, "Unable to read Response Table for LWSwitch=%d port=%d\n", switch_id, switch_port);
                close(fd);
                return false;
            }
            if(valid_out_of_range_res_entry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Response entries found for GPU IDS greater than %d\n", 
                        switch_id, switch_port, valid_out_of_range_res_entry, MAX_GPU);
            }
        }
        close(fd);
    }
    return true;
}

//For request packet compute the final egress switch/port for a starting switch/port/dest_gpu_id
bool
naComputeReqEgressPort(naNodeRequestTables_t &req_tables, int switch_id,
                        uint32_t switch_port, int dest_gpu_id, int &egress_switch_id, int &egress_port_id)
{
    int dev_id = naGetSwitchToDevID(switch_id);
    egress_switch_id = switch_id;
    egress_port_id = req_tables[dev_id][switch_port][dest_gpu_id].egress_port_id;
    if(egress_port_id == DEST_ERROR)
    {
        PRINT_ERROR_VERBOSE("Bad Request path: Inconsistent request table entries to same GPU\n"
                            "\t\tLWSwitch=%d port_id=%d dest_gpu_id=%d", switch_id, switch_port, dest_gpu_id);
    }
    else if(!isTrunkPort(egress_port_id))
    {
        return true;
    }
    else
    {
        uint32_t far_port_id;
        int egress_dev_id;
        egress_switch_id = (switch_id + MAX_SWITCH_PER_BASEBOARD) % MAX_SWITCH;
        far_port_id = getConnectedTrunkPortId(switch_id, egress_port_id);
        egress_dev_id = naGetSwitchToDevID(egress_switch_id);
        if(egress_dev_id == -1)
        {
            PRINT_ERROR_VERBOSE("Bad Request path: Path through trunk port leads to unknown switch\n"
                                "\t\tLWSwitch=%d port_id=%d dest_gpu_id=%d egress LWSwitch=%d\n",
                                    switch_id, switch_port, dest_gpu_id, egress_switch_id);
            egress_port_id = DEST_ERROR;
            return false;
            
        }
        uint32_t tmp_port_id = egress_port_id;
        egress_port_id = req_tables[egress_dev_id][far_port_id][dest_gpu_id].egress_port_id;
        if(egress_port_id == DEST_UNREACHABLE)
        {
            PRINT_ERROR_VERBOSE("Bad Request path: Path from LWSwitch=%d port_id=%d to dest_gpu_id=%d through trunk port=%d\n"
                                 "\t\tleads to egress LWSwitch=%d far_port_id=%d and lead nowhere\n",
                                switch_id, switch_port, dest_gpu_id, tmp_port_id, egress_switch_id, far_port_id);
            egress_port_id = DEST_ERROR;
            return false;
        }
        return true;
    } 
    return true;
}

//For response packet compute the final egress switch/port for a starting switch/port/requestor_link_id
bool
naComputeResEgressPort(naNodeResponseTables_t &res_tables, int switch_id, 
                    uint32_t switch_port, int req_link_id, int &egress_switch_id, int &egress_port_id)
{
    int dest_gpu_id = req_link_id / MAX_SWITCH_PER_BASEBOARD;
    int dest_gpu_port_id = req_link_id % MAX_SWITCH_PER_BASEBOARD;
    egress_switch_id = switch_id;
    int index=((switch_id * NUM_SWITCH_PORTS + switch_port) *  
                    (MAX_GPU + MAX_SWITCH_PER_BASEBOARD)) + req_link_id;

    int dev_id = naGetSwitchToDevID(switch_id);

    egress_port_id = res_tables[dev_id][switch_port][req_link_id];


    if(!isTrunkPort(egress_port_id))
    {
        return true;
    }
    else
    {
        uint32_t far_port_id;
        int egress_dev_id;
        egress_switch_id = (switch_id + MAX_SWITCH_PER_BASEBOARD) % MAX_SWITCH;
        far_port_id = getConnectedTrunkPortId(switch_id, egress_port_id);
        egress_dev_id = naGetSwitchToDevID(egress_switch_id);
        if(egress_dev_id == -1)
        {
            PRINT_ERROR_VERBOSE("Bad Response path: Path through trunk port leads to unknown switch\n"
                                "\t\tLWSwitch=%d port_id=%d req_link_id=%d egress LWSwitch=%d ",
                                    switch_id, switch_port, dest_gpu_id, egress_switch_id);
            egress_port_id = DEST_ERROR;
            return false;
        }
        uint32_t tmp_port_id = egress_port_id;
        egress_port_id = res_tables[egress_dev_id][far_port_id][req_link_id];
        if(egress_port_id == DEST_UNREACHABLE)
        {
            PRINT_ERROR_VERBOSE("Bad Response path: Path from LWSwitch=%d port_id=%d to req_link_id=%d through trunk port=%d\n"
                                 "\t\tleads to egress LWSwitch=%d far_port_id=%d and lead nowhere\n",
                                switch_id, switch_port, dest_gpu_id, tmp_port_id, egress_switch_id, far_port_id);
            egress_port_id = DEST_ERROR;
            return false;
        }
        return true;
    } 
    return true;
}

//check if the final (egress_port_id, egress_switch_ID) is connected to the specified dest_gpu_id
int naCheckReqEgressPort(int dest_gpu_id, int egress_switch_id, int egress_port_id)
{
    if (getConnectedGPUID(egress_switch_id, egress_port_id) == dest_gpu_id) 
    {
        PRINT_VERBOSE("\tCorrect Request path found!\n"); 
        return true;
    } 
    else 
    {
        return false;
    }
}
#ifdef DEBUG
static void setReqEgressPort(LWSWITCH_INGRESS_REQUEST_ENTRY &entry, int port_num)
{
    entry.vcModeValid7_0 = 0;
    entry.vcModeValid15_8 = 0;
    entry.vcModeValid17_16 = 0;
    if(port_num > 0 && port_num < 8) 
    {
         entry.vcModeValid7_0 = 0x1 << (port_num * 4);
    }
    else if(port_num >= 8 && port_num < 16)
    {
         entry.vcModeValid15_8 = 0x1 << ((port_num - 8) * 4);
    }
    else if(port_num == 16 || port_num == 17)
    {
         entry.vcModeValid17_16 = 0x1 << ((port_num - 16)* 4);
    } 
}

static void setResEgressPort(LWSWITCH_INGRESS_RESPONSE_ENTRY &entry, int port_num)
{
    entry.vcModeValid7_0 = 0;
    entry.vcModeValid15_8 = 0;
    entry.vcModeValid17_16 = 0;
    if(port_num > 0 && port_num < 8) 
    {
         entry.vcModeValid7_0 = 0x1 << (port_num * 4);
    }
    else if(port_num >= 8 && port_num < 16)
    {
         entry.vcModeValid15_8 = 0x1 << ((port_num - 8) * 4);
    }
    else if(port_num == 16 || port_num == 17)
    {
         entry.vcModeValid17_16 = 0x1 << ((port_num - 16)* 4);
    } 
}


//set request entry
bool naSetRequestEntry(int switch_id, int switch_port, int dest_gpu_id, int valid, int egress_port)
{
    int fd; 
    if((fd = naOpenSwitchDev(naGetSwitchToDevID(switch_id))) == -1)
    {
        fprintf(stderr, "Unable to open LWSwitch %d errno=%d\n", switch_id, errno);
        return false;
    }
    
    LWSWITCH_SET_INGRESS_REQUEST_TABLE ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switch_port;
    ioctlParams.firstIndex = dest_gpu_id * REQ_ENTRIES_PER_GPU;
    ioctlParams.numEntries = REQ_ENTRIES_PER_GPU;

    for (int i = 0; i < REQ_ENTRIES_PER_GPU; i++) 
    {
        ioctlParams.entries[i].mappedAddress = i;
        if(valid) {
            setReqEgressPort(ioctlParams.entries[i], egress_port);
            ioctlParams.entries[i].entryValid = 1;
            ioctlParams.entries[i].routePolicy = 0;
        }
    }

    if(ioctl(fd, IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE, &ioctlParams) == -1)
    {
        fprintf(stderr, "ioctl failed to set Request Table. errno=%d\n", errno);
        return false;
    }

    close(fd);
    return true;

}

//set response entry
bool naSetResponseEntry(int switch_id, int switch_port, int dest_rlid, int valid, int egress_port)
{
    int fd; 
    if((fd = naOpenSwitchDev(naGetSwitchToDevID(switch_id))) == -1)
    {
        fprintf(stderr, "Unable to open LWSwitch %d errno=%d\n", switch_id, errno);
        return false;
    }

    LWSWITCH_SET_INGRESS_RESPONSE_TABLE ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.portNum = switch_port;
    ioctlParams.firstIndex = dest_rlid;
    ioctlParams.numEntries = 1;

    if(valid) {
        setResEgressPort(ioctlParams.entries[0], egress_port);
        ioctlParams.entries[0].entryValid = 1;
        ioctlParams.entries[0].routePolicy = 0;
    }


    if(ioctl(fd, IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE, &ioctlParams) == -1)
    {
        fprintf(stderr, "ioctl failed to set Request Table. errno=%d\n", errno);
        return false;
    }

    close(fd);
    return true;
}
#endif
