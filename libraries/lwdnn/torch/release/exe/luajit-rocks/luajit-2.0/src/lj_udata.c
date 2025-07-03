/*
** Userdata handling.
** Copyright (C) 2005-2017 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_udata_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_udata.h"

GLwdata *lj_udata_new(lua_State *L, MSize sz, GCtab *elw)
{
  GLwdata *ud = lj_mem_newt(L, sizeof(GLwdata) + sz, GLwdata);
  global_State *g = G(L);
  newwhite(g, ud);  /* Not finalized. */
  ud->gct = ~LJ_TUDATA;
  ud->udtype = UDTYPE_USERDATA;
  ud->len = sz;
  /* NOBARRIER: The GLwdata is new (marked white). */
  setgcrefnull(ud->metatable);
  setgcref(ud->elw, obj2gco(elw));
  /* Chain to userdata list (after main thread). */
  setgcrefr(ud->nextgc, mainthread(g)->nextgc);
  setgcref(mainthread(g)->nextgc, obj2gco(ud));
  return ud;
}

void LJ_FASTCALL lj_udata_free(global_State *g, GLwdata *ud)
{
  lj_mem_free(g, ud, sizeudata(ud));
}

