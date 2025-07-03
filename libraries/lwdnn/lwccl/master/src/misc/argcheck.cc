/*************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h"

static ncclResult_t LwdaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  lwdaPointerAttributes attr;
  lwdaError_t err = lwdaPointerGetAttributes(&attr, pointer);
  if (err != lwdaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s is not a valid pointer", opname, ptrname);
    return ncclIlwalidArgument;
  }
#if LWDART_VERSION >= 10000
  if (attr.type == lwdaMemoryTypeDevice && attr.device != comm->lwdaDev) {
#else
  if (attr.memoryType == lwdaMemoryTypeDevice && attr.device != comm->lwdaDev) {
#endif
    WARN("%s : %s allocated on device %d mismatchs with LWCL device %d", opname, ptrname, attr.device, comm->lwdaDev);
    return ncclIlwalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclIlwalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t ArgsCheck(struct ncclInfo* info) {
  NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
  // First, the easy ones
  if (info->root < 0 || info->root >= info->comm->nRanks) {
    WARN("%s : invalid root %d (root should be in the 0..%d range)", info->opName, info->root, info->comm->nRanks);
    return ncclIlwalidArgument;
  }
  if (info->datatype < 0 || info->datatype >= ncclNumTypes) {
    WARN("%s : invalid type %d", info->opName, info->datatype);
    return ncclIlwalidArgument;
  }
  // Type is OK, compute nbytes. Colwert Allgather/Broadcast calls to chars.
  info->nBytes = info->count * ncclTypeSize(info->datatype);
  if (info->coll == ncclCollAllGather || info->coll == ncclCollBroadcast) {
    info->count = info->nBytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclCollAllGather || info->coll == ncclCollReduceScatter) info->nBytes *= info->comm->nRanks; // count is per rank

  if (info->op < 0 || info->op >= ncclNumOps) {
    WARN("%s : invalid reduction operation %d", info->opName, info->op);
    return ncclIlwalidArgument;
  }

  if (info->comm->checkPointers) {
    // Check LWCA device pointers
    if (info->coll != ncclCollBroadcast || info->comm->rank == info->root) {
      NCCLCHECK(LwdaPtrCheck(info->sendbuff, info->comm, "sendbuff", info->opName));
    }
    if (info->coll != ncclCollReduce || info->comm->rank == info->root) {
      NCCLCHECK(LwdaPtrCheck(info->recvbuff, info->comm, "recvbuff", info->opName));
    }
  }
  return ncclSuccess;
}
