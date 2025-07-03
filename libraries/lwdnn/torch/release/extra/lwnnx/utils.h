#ifndef LWNN_UTILS_H
#define LWNN_UTILS_H
#include <lua.h>
#include "THCGeneral.h"
THCState* getLwtorchState(lua_State* L);
#endif
