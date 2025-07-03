/*************************************************************************
 * Copyright (c) 2015-2017, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GROUP_H_
#define NCCL_GROUP_H_

#include "lwcl.h"
#include "core.h"

bool ncclAsyncMode();
ncclResult_t ncclAsyncErrCheck(ncclResult_t ret);

typedef ncclResult_t(*ncclInitFunc_t)(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, int lwdaDev, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);

typedef ncclResult_t(*ncclCollFunc_t)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, lwdaStream_t stream);

ncclResult_t ncclAsyncColl(ncclComm_t comm);
#endif
