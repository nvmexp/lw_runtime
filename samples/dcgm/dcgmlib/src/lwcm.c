#include <stdio.h>
#include "dcgm_client_internal.h"
#include "er.h"

/* Linker stub for dynamic logging. We can remove this if we change er to something else */
int erLogLevelB = ER_ERR; /* Default logging level. Change this or *erLogLevelP to dynamically
                             change the level that things are logged at with ER macros */
int *erLogLevelP = &erLogLevelB;

dcgmReturn_t apiEnter(void)
{
    return DCGM_ST_OK;
}

void apiExit(void)
{
    
}
