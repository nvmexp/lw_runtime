#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CStorage.c"
#else

#include "THHalf.h"

/* everything is as the generic Storage.c, except few things (see below) */

// FixMe: Requires an unsafe colwersion in that we colwert from lwtorch's 'half'
// to torch's THHalf.  These types are required to be defined in the same way
// (is there some way to enforce this?)
#ifdef THC_REAL_IS_HALF
#define THFILE_REAL_CAST(x) (THHalf *)x
#else
#define THFILE_REAL_CAST(x) x
#endif

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    TH_CONCAT_3(THFile_read,Real,Raw)(file, THFILE_REAL_CAST(fdata), size);               \
    THLwdaCheck(lwdaMemcpy(data, fdata, size * sizeof(real), lwdaMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    THLwdaCheck(lwdaMemcpy(fdata, data, size * sizeof(real), lwdaMemcpyDeviceToHost)); \
    TH_CONCAT_3(THFile_write,Real,Raw)(file, THFILE_REAL_CAST(fdata), size);              \
    THFree(fdata);                                                      \
  }

#define TH_GENERIC_FILE "generic/Storage.c"
#include "generic/Storage.c"

#undef TH_GENERIC_FILE
#undef THFILE_REAL_CAST
#undef THFile_readRealRaw
#undef THFile_writeRealRaw

/* now we overwrite some methods specific to LwdaStorage */

static int lwtorch_Storage_(copy)(lua_State *L)
{
  THCState *state = lwtorch_getstate(L);
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.LwdaByteStorage")) )
    THCStorage_(copyLwdaByte)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaCharStorage")) )
    THCStorage_(copyLwdaChar)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaShortStorage")) )
    THCStorage_(copyLwdaShort)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaIntStorage")) )
    THCStorage_(copyLwdaInt)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaLongStorage")) )
    THCStorage_(copyLwdaLong)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaStorage")) )
    THCStorage_(copyLwdaFloat)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaDoubleStorage")) )
    THCStorage_(copyLwdaDouble)(state, storage, src);
#ifdef LWDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.LwdaHalfStorage")) )
    THCStorage_(copyLwdaHalf)(state, storage, src);
#endif

  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THCStorage_(copyByte)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THCStorage_(copyChar)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THCStorage_(copyShort)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THCStorage_(copyInt)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THCStorage_(copyLong)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THCStorage_(copyFloat)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THCStorage_(copyDouble)(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfStorage")) )
    THCStorage_(copyHalf)(state, storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

static int lwtorch_Storage_(getDevice)(lua_State *L) {
  THCStorage *storage = luaT_checkudata(L, 1, torch_Storage);
  lua_pushinteger(L, THCStorage_(getDevice)(lwtorch_getstate(L), storage) + 1);
  return 1;
}

void lwtorch_Storage_(init)(lua_State* L)
{
  /* the standard stuff */
  torch_Storage_(init)(L);

  // Register this even though it is generated elsewhere.
  lwtorch_StorageCopy_(init)(L);

  luaT_pushmetatable(L, torch_Storage);
  lua_pushcfunction(L, lwtorch_Storage_(copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);

  luaT_pushmetatable(L, torch_Storage);
  lua_pushcfunction(L, lwtorch_Storage_(getDevice));
  lua_setfield(L, -2, "getDevice");
  lua_pop(L, 1);
}

#endif
