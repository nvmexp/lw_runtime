/*************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "lwcl.h"

typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown
} ncclPattern_t;

// Used to pass LWCL call information between functions
struct ncclInfo {
  ncclColl_t coll;
  const char* opName;
  // LWCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  lwdaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  ncclPattern_t pattern;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
};

#endif
