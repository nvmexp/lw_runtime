#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CTensorCopy.c"
#else

static int TH_CONCAT_3(lwtorch_,Real,Tensor_copy)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  void *src;
  if( (src = luaT_toudata(L, 2, TH_CONCAT_STRING_3(torch.,Real,Tensor)) ))
    THTensor_(copy)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THTensor_(copyByte)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THTensor_(copyChar)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THTensor_(copyShort)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THTensor_(copyInt)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THTensor_(copyLong)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THTensor_(copyFloat)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THTensor_(copyDouble)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfTensor")) )
    THTensor_(copyHalf)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaByteTensor")) )
    THTensor_(copyLwdaByte)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaCharTensor")) )
    THTensor_(copyLwdaChar)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaShortTensor")) )
    THTensor_(copyLwdaShort)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaIntTensor")) )
    THTensor_(copyLwdaInt)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaLongTensor")) )
    THTensor_(copyLwdaLong)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaTensor")) )
    THTensor_(copyLwdaFloat)(lwtorch_getstate(L), tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LwdaDoubleTensor")) )
    THTensor_(copyLwdaDouble)(lwtorch_getstate(L), tensor, src);
#ifdef LWDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.LwdaHalfTensor")) )
    THTensor_(copyLwdaHalf)(lwtorch_getstate(L), tensor, src);
#endif
  else
    luaL_typerror(L, 2, "torch.*Tensor");

  lua_settop(L, 1);
  return 1;
}

void lwtorch_TensorCopy_(init)(lua_State* L)
{
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  lua_pushcfunction(L, TH_CONCAT_3(lwtorch_,Real,Tensor_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
}

#endif
