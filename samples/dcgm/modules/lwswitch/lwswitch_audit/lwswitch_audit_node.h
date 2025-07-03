#pragma once
//Abstracts out what is specific to a node type
#include "explorer16common.h"

#define MAX_GPU                 EXPLORER16_NUM_GPU
#define MAX_SWITCH              EXPLORER16_NUM_SWITCH 
#define MAX_SWITCH_PER_BASEBOARD    EXPLORER16_NUM_SWITCH_PER_BASEBOARD
#define REQ_ENTRIES_PER_GPU     EXPLORER16_REQ_ENTRIES_PER_GPU
#define NUM_TRUNK_PORTS         EXPLORER16_NUM_TRUNK_PORTS
#define NUM_ACCESS_PORTS        EXPLORER16_NUM_ACCESS_PORTS
#define NUM_SWITCH_PORTS        EXPLORER16_NUM_SWITCH_PORTS
#define NUM_TABLE_ENTRIES       EXPLORER16_NUM_TABLE_ENTRIES

#define getNthAccessPort        exp16GetNthAccessPort           
#define getNthTrunkPort         exp16GetNthTrunkPort            
#define getConnectedGPUID       exp16GetConnectedGPUID          
#define computeReqLinkID        exp16ComputeReqLinkID           
#define getConnectedTrunkPortId exp16GetConnectedTrunkPortId    
#define switchPhyIDtoSwitchID   exp16SwitchPhyIDtoSwitchID      
#define isTrunkPort             exp16IsTrunkPort                

