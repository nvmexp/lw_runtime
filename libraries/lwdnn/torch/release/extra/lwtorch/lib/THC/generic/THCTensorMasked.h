#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMasked.h"
#else

THC_API void THCTensor_(maskedFill)(THCState *state,
                                    THCTensor *tensor,
                                    THLwdaByteTensor *mask,
                                    real value);

// FIXME: remove now that we have THLwdaByteTensor?
THC_API void THCTensor_(maskedFillByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        real value);

THC_API void THCTensor_(maskedCopy)(THCState *state,
                                    THCTensor *tensor,
                                    THLwdaByteTensor *mask,
                                    THCTensor *src);

// FIXME: remove now that we have THLwdaByteTensor?
THC_API void THCTensor_(maskedCopyByte)(THCState *state,
                                        THCTensor *tensor,
                                        THByteTensor *mask,
                                        THCTensor *src);

THC_API void THCTensor_(maskedSelect)(THCState *state,
                                      THCTensor *tensor,
                                      THCTensor *src,
                                      THLwdaByteTensor *mask);

// FIXME: remove now that we have THLwdaByteTensor?
THC_API void THCTensor_(maskedSelectByte)(THCState *state,
                                          THCTensor *tensor,
                                          THCTensor *src,
                                          THByteTensor *mask);

#endif
