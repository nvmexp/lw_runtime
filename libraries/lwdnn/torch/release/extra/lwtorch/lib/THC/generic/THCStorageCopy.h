#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.h"
#else

/* Support for copy between different Storage types */

THC_API void THCStorage_(rawCopy)(THCState *state, THCStorage *storage, real *src);
THC_API void THCStorage_(copy)(THCState *state, THCStorage *storage, THCStorage *src);
THC_API void THCStorage_(copyByte)(THCState *state, THCStorage *storage, struct THByteStorage *src);
THC_API void THCStorage_(copyChar)(THCState *state, THCStorage *storage, struct THCharStorage *src);
THC_API void THCStorage_(copyShort)(THCState *state, THCStorage *storage, struct THShortStorage *src);
THC_API void THCStorage_(copyInt)(THCState *state, THCStorage *storage, struct THIntStorage *src);
THC_API void THCStorage_(copyLong)(THCState *state, THCStorage *storage, struct THLongStorage *src);
THC_API void THCStorage_(copyFloat)(THCState *state, THCStorage *storage, struct THFloatStorage *src);
THC_API void THCStorage_(copyDouble)(THCState *state, THCStorage *storage, struct THDoubleStorage *src);
THC_API void THCStorage_(copyHalf)(THCState *state, THCStorage *storage, struct THHalfStorage *src);

THC_API void THCStorage_(copyLwdaByte)(THCState *state, THCStorage *storage, struct THLwdaByteStorage *src);
THC_API void THCStorage_(copyLwdaChar)(THCState *state, THCStorage *storage, struct THLwdaCharStorage *src);
THC_API void THCStorage_(copyLwdaShort)(THCState *state, THCStorage *storage, struct THLwdaShortStorage *src);
THC_API void THCStorage_(copyLwdaInt)(THCState *state, THCStorage *storage, struct THLwdaIntStorage *src);
THC_API void THCStorage_(copyLwdaLong)(THCState *state, THCStorage *storage, struct THLwdaLongStorage *src);
THC_API void THCStorage_(copyLwdaFloat)(THCState *state, THCStorage *storage, struct THLwdaStorage *src);
THC_API void THCStorage_(copyLwdaDouble)(THCState *state, THCStorage *storage, struct THLwdaDoubleStorage *src);
#ifdef LWDA_HALF_TENSOR
THC_API void THCStorage_(copyLwdaHalf)(THCState *state, THCStorage *storage, struct THLwdaHalfStorage *src);
#endif

THC_API void TH_CONCAT_2(THByteStorage_copyLwda  , Real)(THCState *state, THByteStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THCharStorage_copyLwda  , Real)(THCState *state, THCharStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THShortStorage_copyLwda , Real)(THCState *state, THShortStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THIntStorage_copyLwda   , Real)(THCState *state, THIntStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THLongStorage_copyLwda  , Real)(THCState *state, THLongStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THFloatStorage_copyLwda , Real)(THCState *state, THFloatStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THDoubleStorage_copyLwda, Real)(THCState *state, THDoubleStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THHalfStorage_copyLwda, Real)(THCState *state, THHalfStorage *self, struct THCStorage *src);

THC_API void THStorage_(copyLwda)(THCState *state, THStorage *self, THCStorage *src);
THC_API void THCStorage_(copyLwda)(THCState *state, THCStorage *self, THCStorage *src);
THC_API void THCStorage_(copyCPU)(THCState *state, THCStorage *self, THStorage *src);

#endif
