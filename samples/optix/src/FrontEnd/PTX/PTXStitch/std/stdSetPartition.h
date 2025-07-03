/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdSetPartition.h
 *
 *  Description              :
 */

#ifndef stdSetPartition_INCLUDED
#define stdSetPartition_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdSet.h"
#include "stdMap.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct {
    stdMap_t      setToPartition;
    stdMap_t      setMap;
    stdSet_t      allPartitions;
    stdMap_t      elementToPartition;
    stdMap_t      setToPartitions;
} *stdPartitionInfo, 
   stdPartitionInfoRec;

/*--------------------------------- Functions --------------------------------*/

stdPartitionInfo STD_CDECL stdPartitionSets( stdSet_t allElements, stdSet_t sets);


#ifdef __cplusplus
}
#endif

#endif
