#ifndef TH_LWDA_TENSOR_COLW_INC
#define TH_LWDA_TENSOR_COLW_INC

#include "THCTensor.h"

struct THCState;

THC_API void THLwdaTensor_colw2Dmv(struct THCState *state, THLwdaTensor *output,
                                   float beta, THLwdaTensor *input, THLwdaTensor *kernel,
                                   long srow, long scol, const char *type);
THC_API void THLwdaTensor_colw2Dmm(struct THCState *state, THLwdaTensor *output,
                                   float beta, THLwdaTensor *input, THLwdaTensor *kernel,
                                   long srow, long scol, const char *type);

THC_API void THLwdaTensor_colw2DRevger(struct THCState *state, THLwdaTensor *output,
                                       float beta, float alpha, THLwdaTensor *input,
                                       THLwdaTensor *kernel, long srow, long scol);
THC_API void THLwdaTensor_colw2DRevgerm(struct THCState *state, THLwdaTensor *output,
                                        float beta, float alpha, THLwdaTensor *input,
                                        THLwdaTensor *kernel, long srow, long scol);

THC_API void THLwdaTensor_colw2Dmap(struct THCState *state, THLwdaTensor *output,
                                    THLwdaTensor *input, THLwdaTensor *kernel,
                                    long stride_x, long stride_y, THLwdaTensor *table, long fanin);

#endif
