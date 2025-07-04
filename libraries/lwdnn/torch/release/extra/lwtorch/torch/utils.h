#ifndef LWTORCH_UTILS_INC
#define LWTORCH_UTILS_INC

#include "luaT.h"
#include "TH.h"

#ifdef __cplusplus
# define TORCH_EXTERNC extern "C"
#else
# define TORCH_EXTERNC extern
#endif

#ifdef __GNUC__
# define TORCH_UNUSED __attribute__((unused))
#else
# define TORCH_UNUSED
#endif

#ifdef _WIN32
# ifdef lwtorch_EXPORTS
#  define TORCH_API TORCH_EXTERNC __declspec(dllexport)
# else
#  define TORCH_API TORCH_EXTERNC __declspec(dllimport)
# endif
#else
# define TORCH_API TORCH_EXTERNC
#endif

#ifndef HAS_LUAL_SETFUNCS
/*
** Adapted from Lua 5.2.0
*/
TORCH_UNUSED static void luaL_setfuncs (lua_State *L, const luaL_Reg *l, int nup) {
  luaL_checkstack(L, nup+1, "too many upvalues");
  for (; l->name != NULL; l++) {  /* fill the table with given functions */
    int i;
    lua_pushstring(L, l->name);
    for (i = 0; i < nup; i++)  /* copy upvalues to the top */
      lua_pushvalue(L, -(nup+1));
    lua_pushcclosure(L, l->func, nup);  /* closure with those upvalues */
    lua_settable(L, -(nup + 3));
  }
  lua_pop(L, nup);  /* remove upvalues */
}
#endif

#if LUA_VERSION_NUM >= 503
/* one can simply enable LUA_COMPAT_5_2 to be backward compatible.
However, this does not work when we are trying to use system-installed lua,
hence these redefines
*/
#define luaL_optlong(L,n,d)     ((long)luaL_optinteger(L, (n), (d)))
#define luaL_optint(L,n,d)  ((int)luaL_optinteger(L, (n), (d)))
#define luaL_checklong(L,n)     ((long)luaL_checkinteger(L, (n)))
#define luaL_checkint(L,n)      ((int)luaL_checkinteger(L, (n)))
#endif

TORCH_API THLongStorage* lwtorch_checklongargs(lua_State *L, int index);
TORCH_API int lwtorch_islongargs(lua_State *L, int index);

struct THCState;
TORCH_API struct THCState* lwtorch_getstate(lua_State* L);

#endif
