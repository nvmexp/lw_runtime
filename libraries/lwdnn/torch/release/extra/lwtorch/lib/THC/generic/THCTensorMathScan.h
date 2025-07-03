#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathScan.h"
#else

THC_API void THCTensor_(lwmsum)(THCState *state, THCTensor *self, THCTensor *src, long dim);
THC_API void THCTensor_(lwmprod)(THCState *state, THCTensor *self, THCTensor *src, long dim);

#endif
