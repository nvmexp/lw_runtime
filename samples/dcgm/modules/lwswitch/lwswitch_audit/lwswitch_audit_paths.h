#pragma once

#include "lwswitch_audit_node.h"
#include "lwswitch_audit_tables.h"

#define PATH_ERROR -1

/*Compute the number of paths programmed between each pair of GPUs in the system*/
bool naComputePaths(naNodeRequestTables_t &req_tables,  naNodeResponseTables_t &res_tables,
                    int paths_matrix[][MAX_GPU], int num_switches);

/*
Print all the src GPU to dst GPU paths computed previously as a matrix.
Each element of the matrix represents the number of paths from src GOU to dest GPU.
Note that values for (A,B) and (B,A) can be different
*/
void naPrintPaths(char *heading, int paths_matrix[][MAX_GPU], bool is_csv);

/*Verify whether requestor link IDs set in switches is correct
num_switches: Number of switch instances in /dev
Return: Number of ports programmed with wrong requestor link IDs
*/
int naCheckReqLinkIDs(int num_switches, int paths_matrix[][MAX_GPU]);

//set Requestor link ID
void naSetRLID(int switch_id, int switch_port, int rlid);

