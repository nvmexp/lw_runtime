#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.c"
#else

/* specific methods */

void THCTensor_(copyCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");

  {
    THCTensor *selfc = THCTensor_(newContiguous)(state, self);
    src = THTensor_(newContiguous)(src);

    THLwdaCheck(lwdaMemcpy(THCTensor_(data)(state,selfc),
                           THTensor_(data)(src),
                           THTensor_(nElement)(src) * sizeof(real),
                           lwdaMemcpyHostToDevice));

    THTensor_(free)(src);
    THCTensor_(freeCopyTo)(state, selfc, self);
  }
}

#define IMPLEMENT_TH_LWDA_TENSOR_COPY(TYPEC)                            \
void THCTensor_(copy##TYPEC)(THCState *state, THCTensor *self, struct TH##TYPEC##Tensor *src)                \
{                                                                       \
  THArgCheck(THCTensor_(nElement)(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
  if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {               \
    THCTensor_(copyCPU)(state, self, (THTensor*) src);  /* cast just removes warnings */                     \
  } else {                                                              \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THTensor *srcf = THTensor_(newWithSize)(size, NULL);                \
                                                                        \
    THTensor_(copy##TYPEC)(srcf, src);                                  \
    THCTensor_(copyCPU)(state, self, srcf);                             \
                                                                        \
    THLongStorage_free(size);                                           \
    THTensor_(free)(srcf);                                              \
  }                                                                     \
}

IMPLEMENT_TH_LWDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Char)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Short)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Int)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Long)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Float)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Double)
IMPLEMENT_TH_LWDA_TENSOR_COPY(Half)

/* copyLwda */

void THTensor_(copyLwda)(THCState *state, THTensor *self, struct THCTensor *src)
{
  THArgCheck(THTensor_(nElement)(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");

  {
    THTensor *selfc = THTensor_(newContiguous)(self);
    src = THCTensor_(newContiguous)(state, src);

    THLwdaCheck(lwdaMemcpy(THTensor_(data)(selfc),
                           THCTensor_(data)(state, src),
                           THCTensor_(nElement)(state, src) * sizeof(real),
                           lwdaMemcpyDeviceToHost));

    THCTensor_(free)(state, src);
    THTensor_(freeCopyTo)(selfc, self);
  }
}

#define IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(TYPEC)                           \
  void TH_CONCAT_4(TH,TYPEC,Tensor_copyLwda,Real)(THCState *state, TH##TYPEC##Tensor *self, struct THCTensor *src) \
  {                                                                       \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");       \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {   \
      THTensor_(copyLwda)(state, (THTensor*) self, src);  /* cast just removes compiler warning */                   \
    } else {                                                              \
      THLongStorage *size = THCTensor_(newSizeOf)(state, src);            \
      THTensor *srcf = THTensor_(newWithSize)(size, NULL);                \
                                                                          \
      THTensor_(copyLwda)(state, srcf, src);                              \
      TH_CONCAT_4(TH,TYPEC,Tensor_copy,Real)(self, srcf);                 \
                                                                          \
      THLongStorage_free(size);                                           \
      THTensor_(free)(srcf);                                              \
    }                                                                     \
  }

IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Float)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Double)
IMPLEMENT_TH_LWDA_TENSOR_COPY_TO(Half)

void THCTensor_(copyLwda)(THCState *state, THCTensor *self, THCTensor *src)
{
  THCTensor_(copy)(state, self, src);
}

void THCTensor_(copyAsyncCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");
  THArgCheck(THCTensor_(isContiguous)(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THTensor_(isContiguous)(src), 3, "Source tensor must be contiguous");

  if (THCTensor_(nElement)(state, self) == 0) return;

  // Perform the copy wrt the current stream on the LwdaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, self);
  int lwrrentDevice;
  THLwdaCheck(lwdaGetDevice(&lwrrentDevice));

  if (lwrrentDevice != tensorDevice) {
    THLwdaCheck(lwdaSetDevice(tensorDevice));
  }

  THCStream *stream  = THCState_getStream(state);
  THLwdaCheck(lwdaMemcpyAsync(THCTensor_(data)(state, self),
                              THTensor_(data)(src),
                              THTensor_(nElement)(src) * sizeof(real),
                              lwdaMemcpyHostToDevice,
                              stream->stream));

  THLwdaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (lwrrentDevice != tensorDevice) {
    THLwdaCheck(lwdaSetDevice(lwrrentDevice));
  }
}

void THTensor_(copyAsyncLwda)(THCState *state, THTensor *self, struct THCTensor *src)
{
  THArgCheck(THTensor_(nElement)(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");
  THArgCheck(THTensor_(isContiguous)(self), 2, "Target tensor must be contiguous");
  THArgCheck(THCTensor_(isContiguous)(state, src), 3, "Source tensor must be contiguous");

  if (THTensor_(nElement)(self) == 0) return;

  // Perform the copy wrt the current stream on the LwdaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, src);
  int lwrrentDevice;
  THLwdaCheck(lwdaGetDevice(&lwrrentDevice));

  if (lwrrentDevice != tensorDevice) {
    THLwdaCheck(lwdaSetDevice(tensorDevice));
  }

  THCStream *stream = THCState_getStream(state);
  THLwdaCheck(lwdaMemcpyAsync(THTensor_(data)(self),
                              THCTensor_(data)(state, src),
                              THCTensor_(nElement)(state, src) * sizeof(real),
                              lwdaMemcpyDeviceToHost,
                              stream->stream));

  THLwdaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (lwrrentDevice != tensorDevice) {
    THLwdaCheck(lwdaSetDevice(lwrrentDevice));
  }
}

#undef IMPLEMENT_TH_LWDA_TENSOR_COPY
#undef IMPLEMENT_TH_LWDA_TENSOR_COPY_TO

#endif
