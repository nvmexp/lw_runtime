/*************************************************************************
 * Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

#if LWDART_VERSION < 9000
struct lwdaLaunchParams {
  void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  lwdaStream_t stream;
};
#endif

#define MAXCHANNELS 16
#define DEFAULT_BUFFER_SIZE_BYTES (1LL << 22) /* 4MiB */

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define LWDA_IPC_MIN 2097152UL /* 2MiB - not lwrrently used */

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      char pad2[CACHE_LINE_SIZE-sizeof(void*)];
      uint64_t opCount;
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      uint64_t opCount;
      char pad2[CACHE_LINE_SIZE-sizeof(uint64_t)];
      int sizesFifo[NCCL_STEPS];
    };
    char pad4[MEM_ALIGN];
  };
  ncclLLFifoLine llBuff[NCCL_LL_BUFF_LINES];
  char buff[1]; // Actually larger than that
};

struct ncclComm {
  struct ncclChannel channels[MAXCHANNELS];

  struct ncclPeerInfo* peerInfo;

  void* bootstrap;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int lwdaDev; // my lwca device index
  int lwmlDev; // my LWML device number

  enum { GROUP, PARALLEL } launchMode;
  lwdaStream_t userStream;
  bool userStreamSet;
  lwdaEvent_t doneEvent;
  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;

  // Channels for collectives
  int nChannels;
  int nThreads;

  // Low-latency algorithm threshold
  ssize_t llThreshold;
  ssize_t threadThreshold;

  // Tree algorithm threshold
  ssize_t treeThreshold;

  // An internal LWCA stream for LWCL kernel CGMD launches
  int groupLwdaStream;
  lwdaStream_t groupStream;

  // Whether there has been a fatal error in this communicator.
  ncclResult_t fatalError;

  // Error reported by GPU
  volatile ncclDevError_t* fatalDevError;

  // Flag to ask LWCL kernels to abort
  volatile uint32_t *abortFlag;

  // Device side of the communicator
  struct ncclDevComm *devComm;
  // Host copy of the devComm (to free LWCA allocs)
  struct ncclDevComm hostDevComm;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  struct lwdaLaunchParams * intraParams;
  struct lwdaLaunchParams *myParams;
  int* intraLwdaDevs;
  int* intraCGMode; // Whether we can use LWDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  struct ncclColl args;
  void* argsptr;

  // Global proxy thread
  pthread_t proxyThread;
  struct ncclProxyState proxyState;
};

#endif
