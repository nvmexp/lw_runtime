#include "THCStream.h"

#include <mutex>
#include <lwda_runtime_api.h>
#include "THAtomic.h"

#define MAX_DEVICES 256
static THCStream default_streams[MAX_DEVICES];

static void initialize_default_streams()
{
  for (int i = 0; i < MAX_DEVICES; i++) {
    default_streams[i].device = i;
  }
}

THCStream* THCStream_new(int flags)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THLwdaCheck(lwdaGetDevice(&self->device));
  THLwdaCheck(lwdaStreamCreateWithFlags(&self->stream, flags));
  return self;
}

THC_API THCStream* THCStream_defaultStream(int device)
{
  // default streams aren't refcounted
  THAssert(device >= 0 && device < MAX_DEVICES);
  std::once_flag once;
  std::call_once(once, &initialize_default_streams);
  return &default_streams[device];
}

THCStream* THCStream_newWithPriority(int flags, int priority)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THLwdaCheck(lwdaGetDevice(&self->device));
  THLwdaCheck(lwdaStreamCreateWithPriority(&self->stream, flags, priority));
  return self;
}

void THCStream_free(THCStream* self)
{
  if (!self || !self->stream) {
    return;
  }
  if (THAtomicDecrementRef(&self->refcount)) {
    THLwdaCheckWarn(lwdaStreamDestroy(self->stream));
    free(self);
  }
}

void THCStream_retain(THCStream* self)
{
  if (self->stream) {
    THAtomicIncrementRef(&self->refcount);
  }
}
