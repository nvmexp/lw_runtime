#include "utils.h"
THCState* getLwtorchState(lua_State* L)
{
  lua_getglobal(L, "lwtorch");
  lua_getfield(L, -1, "getState");
  lua_call(L, 0, 1);
  THCState *state = (THCState*) lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}
