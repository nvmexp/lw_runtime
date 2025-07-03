
#ifndef EXPLORER8_COMMON_H
#define EXPLORER8_COMMON_H

  // define enums to take hardware device ID to protobuf device ID
  typedef enum switchID_t {
      U1LWP1 = 2,
      U1LWP2 = 1,
      U1LWP3 = 0
  } switchID;

  typedef enum GPUID_t {
      LWL_GPU1 = 7,
      LWL_GPU2 = 5,
      LWL_GPU3 = 3,
      LWL_GPU4 = 1,
      LWL_GPU5 = 6,
      LWL_GPU6 = 4,
      LWL_GPU7 = 2,
      LWL_GPU8 = 0
  } GPUID;

  typedef struct HWLink_t {
      unsigned int willowIndex;
      unsigned int willowPort;
      unsigned int GPUIndex;
      unsigned int GPUPort;
  } HWLink;

  typedef union vcValid16_t {
      unsigned long long int port0_15;
      struct regs{
          unsigned int port0_7;
          unsigned int port8_15;
      } hwRegs;
  } vcValid;

#endif

