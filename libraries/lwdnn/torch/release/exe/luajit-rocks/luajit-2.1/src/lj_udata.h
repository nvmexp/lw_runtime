/*
** Userdata handling.
** Copyright (C) 2005-2017 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_UDATA_H
#define _LJ_UDATA_H

#include "lj_obj.h"

LJ_FUNC GLwdata *lj_udata_new(lua_State *L, MSize sz, GCtab *elw);
LJ_FUNC void LJ_FASTCALL lj_udata_free(global_State *g, GLwdata *ud);

#endif
