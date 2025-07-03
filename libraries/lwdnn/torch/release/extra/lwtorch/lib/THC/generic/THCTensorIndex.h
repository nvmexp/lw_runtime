#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorIndex.h"
#else

THC_API void THCTensor_(indexCopy)(THCState *state, THCTensor *res_, int dim, THLwdaLongTensor *indices, THCTensor *src);
THC_API void THCTensor_(indexAdd)(THCState *state, THCTensor *res_, int dim, THLwdaLongTensor *indices, THCTensor *src);
THC_API void THCTensor_(indexFill)(THCState *state, THCTensor *tensor, int dim, THLwdaLongTensor *index, real val);
THC_API void THCTensor_(indexSelect)(THCState *state, THCTensor *tensor, THCTensor *src, int dim, THLwdaLongTensor *index);

THC_API void THCTensor_(indexCopy_long)(THCState *state, THCTensor *res_, int dim, THLongTensor *indices, THCTensor *src);
THC_API void THCTensor_(indexAdd_long)(THCState *state, THCTensor *res_, int dim, THLongTensor *indices, THCTensor *src);
THC_API void THCTensor_(indexFill_long)(THCState *state, THCTensor *tensor, int dim, THLongTensor *index, real val);
THC_API void THCTensor_(indexSelect_long)(THCState *state, THCTensor *tensor, THCTensor *src, int dim, THLongTensor *index);

THC_API void THCTensor_(callwlateAdvancedIndexingOffsets)(THCState *state, THLwdaLongTensor *output, THCTensor *indexed, ptrdiff_t baseOffset, THLwdaLongTensor **indexers);

#endif
