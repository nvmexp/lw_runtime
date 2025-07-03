#pragma once

#include "lwswitch_audit_node.h"

#define PATH_ERROR -1


/*Compute the number of paths programmed between each pair of GPUs in the system*/
bool naComputePaths(naNodeTables_t &reqTables,  naNodeTables_t &resTables,
                    naPathsMatrix_t &pathsMatrix, int numSwitches, node *np);

/*
Print all the src GPU to dst GPU paths computed previously as a matrix.
Each element of the matrix represents the number of paths from src GOU to dest GPU.
Note that values for (A,B) and (B,A) can be different
*/
void naPrintPaths(char *heading, naPathsMatrix_t &pathsMatrix, bool isCsv, node *np, bool printFullMatrix);

#ifdef DEBUG
//set Requestor link ID
void naSetRLID(int switchId, int switchPort, int rlid);
#endif
