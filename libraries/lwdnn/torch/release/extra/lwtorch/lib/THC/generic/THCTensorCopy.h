#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.h"
#else

THC_API void THCTensor_(copy)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(copyIgnoringOverlaps)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(copyByte)(THCState *state, THCTensor *self, THByteTensor *src);
THC_API void THCTensor_(copyChar)(THCState *state, THCTensor *self, THCharTensor *src);
THC_API void THCTensor_(copyShort)(THCState *state, THCTensor *self, THShortTensor *src);
THC_API void THCTensor_(copyInt)(THCState *state, THCTensor *self, THIntTensor *src);
THC_API void THCTensor_(copyLong)(THCState *state, THCTensor *self, THLongTensor *src);
THC_API void THCTensor_(copyFloat)(THCState *state, THCTensor *self, THFloatTensor *src);
THC_API void THCTensor_(copyDouble)(THCState *state, THCTensor *self, THDoubleTensor *src);
THC_API void THCTensor_(copyHalf)(THCState *state, THCTensor *self, struct THHalfTensor *src);

THC_API void THCTensor_(copyLwdaByte)(THCState *state, THCTensor *dst, struct THLwdaByteTensor *src);
THC_API void THCTensor_(copyLwdaChar)(THCState *state, THCTensor *dst, struct THLwdaCharTensor *src);
THC_API void THCTensor_(copyLwdaShort)(THCState *state, THCTensor *dst, struct THLwdaShortTensor *src);
THC_API void THCTensor_(copyLwdaInt)(THCState *state, THCTensor *dst, struct THLwdaIntTensor *src);
THC_API void THCTensor_(copyLwdaLong)(THCState *state, THCTensor *dst, struct THLwdaLongTensor *src);
THC_API void THCTensor_(copyLwdaFloat)(THCState *state, THCTensor *dst, struct THLwdaTensor *src);
THC_API void THCTensor_(copyLwdaDouble)(THCState *state, THCTensor *dst, struct THLwdaDoubleTensor *src);
#ifdef LWDA_HALF_TENSOR
THC_API void THCTensor_(copyLwdaHalf)(THCState *state, THCTensor *dst, struct THLwdaHalfTensor *src);
#endif

THC_API void TH_CONCAT_2(THByteTensor_copyLwda  , Real)  (THCState *state, THByteTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THCharTensor_copyLwda  , Real)  (THCState *state, THCharTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THShortTensor_copyLwda , Real)  (THCState *state, THShortTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THIntTensor_copyLwda   , Real)  (THCState *state, THIntTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THLongTensor_copyLwda  , Real)  (THCState *state, THLongTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THFloatTensor_copyLwda , Real)  (THCState *state, THFloatTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THDoubleTensor_copyLwda, Real)  (THCState *state, THDoubleTensor *self, THCTensor *src);
THC_API void TH_CONCAT_2(THHalfTensor_copyLwda, Real)    (THCState *state, THHalfTensor *self, THCTensor *src);
THC_API void THCTensor_(copyLwda) (THCState *state, THCTensor *self, THCTensor *src);

THC_API void THTensor_(copyLwda) (THCState *state, THTensor *self, THCTensor *src);
THC_API void THCTensor_(copyCPU) (THCState *state, THCTensor *self, THTensor *src);

THC_API void THCTensor_(copyAsyncCPU)(THCState *state, THCTensor *self, THTensor *src);
THC_API void THTensor_(copyAsyncLwda)(THCState *state, THTensor *self, THCTensor *src);

#endif
