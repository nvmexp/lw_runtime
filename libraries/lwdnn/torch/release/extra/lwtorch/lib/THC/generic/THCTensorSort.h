#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorSort.h"
#else

/* Performs an in-place sort of (keys, values). Only works for slice sizes
   <= 2048 at the moment (slice size == size of keys/values dim `dim`) */
THC_API void THCTensor_(sortKeyValueInplace)(THCState* state,
                                             THCTensor* keys,
                                             THLwdaLongTensor* values,
                                             int dim, int order);

/* Performs an out-of-place sort of `input`, returning the per-slice indices
   in `indices` and the sorted values in `sorted` */
THC_API void THCTensor_(sort)(THCState* state,
                              THCTensor* sorted,
                              THLwdaLongTensor* indices,
                              THCTensor* input,
                              int dim, int order);

#endif
