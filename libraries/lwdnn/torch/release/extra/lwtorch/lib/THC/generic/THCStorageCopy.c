#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.c"
#else

void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THLwdaCheck(lwdaMemcpy(self->data, src->data, self->size * sizeof(real), lwdaMemcpyHostToDevice));
}

#define TH_LWDA_STORAGE_IMPLEMENT_COPY(TYPEC)                          \
void THCStorage_(copy##TYPEC)(THCState *state, THCStorage *self, struct TH##TYPEC##Storage *src)  \
{                                                                      \
  THCTensor* selfTensor =                                              \
      THCTensor_(newWithStorage1d)(state, self, 0, self->size, 1);     \
  struct TH##TYPEC##Tensor* srcTensor =                                \
      TH##TYPEC##Tensor_newWithStorage1d(src, 0, src->size, 1);        \
  THCTensor_(copy##TYPEC)(state, selfTensor, srcTensor);               \
  TH##TYPEC##Tensor_free(srcTensor);                                   \
  THCTensor_(free)(state, selfTensor);                                 \
}
TH_LWDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Char)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Short)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Int)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Long)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Float)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Half)
TH_LWDA_STORAGE_IMPLEMENT_COPY(Double)

void THStorage_(copyLwda)(THCState *state, THStorage *self, struct THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THLwdaCheck(lwdaMemcpy(self->data, src->data, self->size * sizeof(real), lwdaMemcpyDeviceToHost));
}

#define TH_LWDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                             \
void TH_CONCAT_4(TH,TYPEC,Storage_copyLwda,Real)(THCState *state, TH##TYPEC##Storage *self, struct THCStorage *src) \
{                                                                           \
  TH##TYPEC##Tensor* selfTensor =                                           \
      TH##TYPEC##Tensor_newWithStorage1d(self, 0, self->size, 1);           \
  struct THCTensor* srcTensor =                                             \
      THCTensor_(newWithStorage1d)(state, src, 0, src->size, 1);            \
  TH_CONCAT_4(TH,TYPEC,Tensor_copyLwda,Real)(state, selfTensor, srcTensor); \
  THCTensor_(free)(state, srcTensor);                                       \
  TH##TYPEC##Tensor_free(selfTensor);                                   \
}
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Float)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Half)
TH_LWDA_STORAGE_IMPLEMENT_COPYTO(Double)

#undef TH_LWDA_STORAGE_IMPLEMENT_COPY
#undef TH_LWDA_STORAGE_IMPLEMENT_COPYTO

#endif
