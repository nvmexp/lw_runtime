/*************************************************************************
 * Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "core.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelid);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks);

#endif
