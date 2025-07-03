/*
 * Shmulik Regev <shmulbox@gmail.com>
 */
/* -*- c-basic-offset: 2 -*- */
/* -*- c-default-style: "k&r" -*- */

#include <ctype.h>
#include <string.h>

#include "lmdb.h"

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#if LUA_VERSION_NUM<=501
typedef luaL_reg lua_reg_t;
# define lua_type_error luaL_typerror
void lua_set_funcs(lua_State *L, const char *libname,const lua_reg_t *l) {
  lua_setglobal(L,libname);
  luaL_register(L,libname,l);
}

#else
typedef luaL_Reg lua_reg_t;
void lua_set_funcs(lua_State *L, const char *libname,const lua_reg_t *l) {
  luaL_setfuncs(L,l,0);
}
int lua_type_error(lua_State *L,int narg,const char *tname) {
  const char *msg = lua_pushfstring(L, "%s expected, got %s",
                                    tname, luaL_typename(L, narg));
  return luaL_argerror(L, narg, msg);
}
#endif

#ifndef luaL_reg
#define luaL_reg luaL_Reg
#endif

#include "lpack.c"

#define LIGHTNING "lightningmdb"
#define ELW "lightningmdb_elw"
#define TXN "lightningmdb_txn"
#define CURSOR "lightningmdb_lwrsor"

#define setfield_enum(x) lua_pushnumber(L,x); lua_setfield(L,-2,#x)

#define DEFINE_register_methods(x,X)                                  \
  void x##_register(lua_State* L) {                                   \
    luaL_newmetatable(L, X);                                          \
    lua_pushstring(L, "__index");                                     \
    lua_pushvalue(L, -2);        /* push metatable */                 \
    lua_rawset(L, -3);           /* metatable.__index = metatable */  \
    lua_set_funcs(L,X,x##_methods);                                   \
  }

#define set_meta_and_return(x,X)                            \
  *(MDB_##x **)(lua_newuserdata(L,sizeof(MDB_##x *))) = x;  \
  luaL_getmetatable(L,X);                                   \
  lua_setmetatable(L,-2);                                   \
  return 1;


static int clean_metatable(lua_State* L) {
  lua_pushnil(L);
  lua_setmetatable(L,-2);
  return 0;
}


static int str_error_and_out(lua_State* L,const char* err) {
  lua_pushnil(L);
  if ( err ) {
    lua_pushstring(L,err);
  } else {
    lua_pushstring(L,"Unknown error");
  }
  return 2;
}

static int error_and_out(lua_State* L,int err) {
  str_error_and_out(L,mdb_strerror(err));
  lua_pushinteger(L,err);
  return 3;
}

static int success_or_err(lua_State* L,int err) {
  if ( err ) {
    return error_and_out(L,err);
  }
  lua_settop(L,1);
  return 1;
}

static int unimplemented(lua_State* L) {
  return str_error_and_out(L,"method unimplemented");
}

#define DEFINE_check(x,NAME)                            \
  static MDB_##x* check_##x(lua_State *L, int index) {  \
    MDB_##x* y;                                         \
    luaL_checktype(L, index, LUA_TUSERDATA);            \
    y = *(MDB_##x**)luaL_checkudata(L, index, NAME);    \
    if (y == NULL) lua_type_error(L,index,NAME);        \
    return y;                                           \
  }

DEFINE_check(elw,ELW)
DEFINE_check(txn,TXN)
DEFINE_check(cursor,CURSOR)

static int stat_to_table(lua_State *L,MDB_stat *stat) {
  lua_newtable(L);
  lua_pushnumber(L,stat->ms_psize);
  lua_setfield(L,-2,"ms_psize");
  lua_pushnumber(L,stat->ms_depth);
  lua_setfield(L,-2,"ms_depth");
  lua_pushnumber(L,stat->ms_branch_pages);
  lua_setfield(L,-2,"ms_branch_pages");
  lua_pushnumber(L,stat->ms_leaf_pages);
  lua_setfield(L,-2,"ms_leaf_pages");
  lua_pushnumber(L,stat->ms_overflow_pages);
  lua_setfield(L,-2,"ms_overflow_pages");
  lua_pushnumber(L,stat->ms_entries);
  lua_setfield(L,-2,"ms_entries");
  return 1;
}


static MDB_val* pop_val(lua_State* L,int index,MDB_val* val) {
  if ( lua_isnil(L,index) ) {
    return NULL;
  }
  val->mv_data = (void*)luaL_checklstring(L,index,&val->mv_size);
  return val;
}

/* elw */
static int elw_open(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  const char* path = luaL_checkstring(L,2);
  mode_t flags = (mode_t)luaL_checkinteger(L,3);
  int mode = luaL_checkinteger(L,4);
  int err;

  if ( !path ) {
    return str_error_and_out(L,"path required");
  }
  err = mdb_elw_open(elw,path,flags,mode);
  return success_or_err(L,err);
}

static int elw_copy(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  const char* path = luaL_checkstring(L,2);
  int err;

  if ( !path ) {
    return str_error_and_out(L,"path required");
  }
  err = mdb_elw_copy(elw,path);
  return success_or_err(L,err);
}

static int elw_stat(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  MDB_stat stat;
  mdb_elw_stat(elw,&stat);
  return stat_to_table(L,&stat);
}

static int elw_info(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  MDB_elwinfo info;
  mdb_elw_info(elw,&info);
  lua_newtable(L);

  lua_pushnumber(L,info.me_mapsize);
  lua_setfield(L,-2,"me_mapsize");
  lua_pushnumber(L,info.me_last_pgno);
  lua_setfield(L,-2,"ms_last_pgno");
  lua_pushnumber(L,info.me_last_txnid);
  lua_setfield(L,-2,"me_last_txnid");
  lua_pushnumber(L,info.me_maxreaders);
  lua_setfield(L,-2,"me_maxreaders");
  lua_pushnumber(L,info.me_numreaders);
  lua_setfield(L,-2,"me_numreaders");
  return 1;

}

static int elw_sync(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  int force = luaL_checkinteger(L,2);
  int err = mdb_elw_sync(elw,force);
  return success_or_err(L,err);
}

static int elw_close(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  mdb_elw_close(elw);
  clean_metatable(L);
  return 0;
}

static int elw_set_flags(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  unsigned int flags = luaL_checkinteger(L,2);
  int onoff = luaL_checkinteger(L,3);
  int err = mdb_elw_set_flags(elw,flags,onoff);
  return success_or_err(L,err);
}

static int elw_get_flags(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  unsigned int flags = 0;
  int err = mdb_elw_get_flags(elw,&flags);
  if ( err ) {
    return error_and_out(L,err);
  }
  lua_pushinteger(L,flags);
  return 1;
}

static int elw_get_path(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  const char* path;
  int err = mdb_elw_get_path(elw,&path);
  if ( err ) {
    return error_and_out(L,err);
  }
  lua_pushstring(L,path);
  return 1;
}

static int elw_set_mapsize(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  size_t size = luaL_checkinteger(L,2);
  int err = mdb_elw_set_mapsize(elw,size);
  return success_or_err(L,err);
}

static int elw_set_maxreaders(lua_State *L) {
  /* TODO */
  return unimplemented(L);
}

static int elw_get_maxreaders(lua_State *L) {
  /* TODO */
  return unimplemented(L);
}

static int elw_set_maxdbs(lua_State *L) {
  MDB_elw* elw = check_elw(L,1);
  int num = luaL_checkinteger(L,2);
  int err = mdb_elw_set_maxdbs(elw,num);
  return success_or_err(L,err);
}

static int elw_txn_begin(lua_State* L) {
  MDB_elw* elw = check_elw(L,1);
  MDB_txn* parent = lua_isnil(L,2) ? NULL : check_txn(L,2);
  unsigned int flags = luaL_checkinteger(L,3);
  int err;
  MDB_txn* txn;
  if ( !elw ) {
    return str_error_and_out(L,"bad params");
  }

  err = mdb_txn_begin(elw,parent,flags,&txn);
  if ( err ) {
    return error_and_out(L,err);
  }
  set_meta_and_return(txn,TXN);
}

static int elw_dbi_close(lua_State* L) {
  MDB_elw* elw = check_elw(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);

  mdb_dbi_close(elw,dbi);
  return 0;
}


static const lua_reg_t elw_methods[] = {
  {"__gc",elw_close},
  {"open",elw_open},
  {"copy",elw_copy},
  {"stat",elw_stat},
  {"info",elw_info},
  {"sync",elw_sync},
  {"close",elw_close},
  {"set_flags",elw_set_flags},
  {"get_flags",elw_get_flags},
  {"get_path",elw_get_path},
  {"set_mapsize",elw_set_mapsize},
  {"set_maxreaders",elw_set_maxreaders},
  {"get_maxreaders",elw_get_maxreaders},
  {"set_maxdbs",elw_set_maxdbs},
  {"txn_begin",elw_txn_begin},
  {"dbi_close",elw_dbi_close},
  {0,0}
};


DEFINE_register_methods(elw,ELW)

/* cursor */
static int lwrsor_close(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  mdb_lwrsor_close(cursor);
  clean_metatable(L);
  return 0;
}

static int lwrsor_txn(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  MDB_txn* txn = mdb_lwrsor_txn(cursor);
  (void)txn;
  /* TODO - we'll need a mapping from MDB_txn* to the lua userdata */
  return unimplemented(L);
}

static int lwrsor_dbi(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  MDB_dbi dbi = mdb_lwrsor_dbi(cursor);
  lua_pushinteger(L,dbi);
  return 1;
}

static int lwrsor_get(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  MDB_val k,v;
  MDB_lwrsor_op op = luaL_checkinteger(L,3);
  int err;
  pop_val(L,2,&k);
  err = mdb_lwrsor_get(cursor,&k,&v,op);
  switch (err) {
  case MDB_NOTFOUND:
    lua_pushnil(L);
    return 1;
  case 0:
    lua_pushlstring(L,k.mv_data,k.mv_size);
    lua_pushlstring(L,v.mv_data,v.mv_size);
    return 2;
  }
  return error_and_out(L,err);
}

static int lwrsor_get_key(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  MDB_val k;
  MDB_lwrsor_op op = luaL_checkinteger(L,3);
  int err;
  pop_val(L,2,&k);
  err = mdb_lwrsor_get(cursor,&k,NULL,op);
  switch (err) {
  case MDB_NOTFOUND:
    lua_pushnil(L);
    return 1;
  case 0:
    lua_pushlstring(L,k.mv_data,k.mv_size);
    return 1;
  }
  return error_and_out(L,err);
}

static int lwrsor_put(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  MDB_val k,v;
  unsigned int flags = luaL_checkinteger(L,4);
  int err;
  pop_val(L,2,&k);
  pop_val(L,3,&v);
  err = mdb_lwrsor_put(cursor,&k,&v,flags);
  return success_or_err(L,err);
}

static int lwrsor_del(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  unsigned int flags = luaL_checkinteger(L,2);
  int err = mdb_lwrsor_del(cursor,flags);
  return success_or_err(L,err);
}

static int lwrsor_count(lua_State *L) {
  MDB_lwrsor* cursor = check_lwrsor(L,1);
  size_t count = 0;
  int err = mdb_lwrsor_count(cursor,&count);
  if ( err ) {
    return error_and_out(L,err);
  }
  lua_pushinteger(L,count);
  return 1;
}

static const lua_reg_t lwrsor_methods[] = {
  {"__gc",lwrsor_close},
  {"close",lwrsor_close},
  {"txn",lwrsor_txn},
  {"dbi",lwrsor_dbi},
  {"get",lwrsor_get},
  {"get_key",lwrsor_get_key},
  {"put",lwrsor_put},
  {"del",lwrsor_del},
  {"count",lwrsor_count},

  {0,0}
};

DEFINE_register_methods(cursor,CURSOR)

/* txn */

static int txn_commit(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  int err = mdb_txn_commit(txn);

  if ( err ) {
    return error_and_out(L,err);
  }

  clean_metatable(L);
  lua_pushboolean(L,1);
  return 1;
}

static int txn_abort(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  mdb_txn_abort(txn);
  clean_metatable(L);
  return 0;
}

static int txn_reset(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  mdb_txn_reset(txn);
  return 0;
}

static int txn_renew(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  mdb_txn_renew(txn);
  return 0;
}

static int txn_dbi_open(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  const char* name = lua_isnil(L,2) ? NULL : luaL_checkstring(L,2);
  unsigned int flags = luaL_checkinteger(L,3);
  MDB_dbi dbi;
  int err = mdb_dbi_open(txn,name,flags,&dbi);
  if ( err ) {
    return error_and_out(L,err);
  }

  lua_pushinteger(L,dbi);
  return 1;
}

static int txn_stat(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_stat stat;
  mdb_stat(txn,dbi,&stat);
  return stat_to_table(L,&stat);
}

static int txn_dbi_drop(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  int del = luaL_checkinteger(L,3);
  int err = mdb_drop(txn,dbi,del);
  return success_or_err(L,err);
}

static int txn_get(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_val k,v;
  int err;

  err = mdb_get(txn,dbi,pop_val(L,3,&k),&v);
  switch (err) {
  case MDB_NOTFOUND:
    lua_pushnil(L);
    return 1;
  case 0:
    lua_pushlstring(L,v.mv_data,v.mv_size);
    return 1;
  }
  return error_and_out(L,err);
}

static int txn_put(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_val k,v;
  unsigned int flags = luaL_checkinteger(L,5);
  int err;

  err = mdb_put(txn,dbi,pop_val(L,3,&k),pop_val(L,4,&v),flags);
  return success_or_err(L,err);
}

static int txn_del(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_val k,v;
  int err;
  pop_val(L,3,&k);
  err = mdb_del(txn,dbi,&k,pop_val(L,4,&v));
  return success_or_err(L,err);
}

static int txn_cmp_helper(lua_State* L,int mode) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_val a,b;
  pop_val(L,3,&a);
  pop_val(L,4,&b);

  if ( mode==MDB_DUPSORT )
    return mdb_dcmp(txn,dbi,&a,&b);
  return mdb_cmp(txn,dbi,&a,&b);
}

static int txn_cmp(lua_State* L) {
  return txn_cmp_helper(L,0);
}

static int txn_dcmp(lua_State* L) {
  return txn_cmp_helper(L,MDB_DUPSORT);
}

static int txn_id(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  lua_pushinteger(L,mdb_txn_id(txn));
  return 1;
}

static int txn_lwrsor_open(lua_State* L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_dbi dbi = luaL_checkinteger(L,2);
  MDB_lwrsor* cursor;
  int err = mdb_lwrsor_open(txn,dbi,&cursor);
  if ( err ) {
    return error_and_out(L,err);
  }

  set_meta_and_return(cursor,CURSOR);
}

static int txn_lwrsor_renew(lua_State *L) {
  MDB_txn* txn = check_txn(L,1);
  MDB_lwrsor* cursor = check_lwrsor(L,2);
  int err = mdb_lwrsor_renew(txn,cursor);
  return success_or_err(L,err);
}

static const lua_reg_t txn_methods[] = {
  {"__gc",clean_metatable},
  {"id",txn_id},
  {"commit",txn_commit},
  {"abort",txn_abort},
  {"reset",txn_reset},
  {"renew",txn_renew},
  {"dbi_open",txn_dbi_open},
  {"stat",txn_stat},
  {"dbi_drop",txn_dbi_drop},
  {"get",txn_get},
  {"put",txn_put},
  {"del",txn_del},
  {"cmp",txn_cmp},
  {"dcmp",txn_dcmp},
  {"lwrsor_open",txn_lwrsor_open},
  {"lwrsor_renew",txn_lwrsor_renew},
  {0,0}
};

DEFINE_register_methods(txn,TXN)

/* globals */
static int lmdb_version(lua_State *L) {
  const char* ver = mdb_version(NULL,NULL,NULL);
  lua_pushstring(L,ver);
  return 1;
}

static int lmdb_strerror(lua_State *L) {
  int err = luaL_checkinteger(L,1);
  lua_pushstring(L,mdb_strerror(err));
  return 1;
}

static int lmdb_elw_create(lua_State *L) {
  MDB_elw* elw = NULL;
  int err = mdb_elw_create(&elw);
  if ( err ) {
    lua_pushnil(L);
    lua_pushstring(L,mdb_strerror(err));
    return 2;
  }

  set_meta_and_return(elw,ELW);
}

static const lua_reg_t globals[] = {
  {"version",lmdb_version},
  {"strerror",lmdb_strerror},
  {"elw_create",lmdb_elw_create},
  {NULL,  NULL}
};



int luaopen_lightningmdb(lua_State *L) {
  luaopen_pack(L);

  luaL_newmetatable(L,LIGHTNING);
  lua_set_funcs(L,LIGHTNING,globals);
  setfield_enum(MDB_NOOVERWRITE);
  setfield_enum(MDB_NODUPDATA);
  setfield_enum(MDB_LWRRENT);
  setfield_enum(MDB_RESERVE);
  setfield_enum(MDB_APPEND);
  setfield_enum(MDB_APPENDDUP);
  setfield_enum(MDB_MULTIPLE);

  setfield_enum(MDB_SUCCESS);
  setfield_enum(MDB_KEYEXIST);
  setfield_enum(MDB_NOTFOUND);
  setfield_enum(MDB_PAGE_NOTFOUND);
  setfield_enum(MDB_CORRUPTED);
  setfield_enum(MDB_PANIC);
  setfield_enum(MDB_VERSION_MISMATCH);
  setfield_enum(MDB_ILWALID);
  setfield_enum(MDB_MAP_FULL);
  setfield_enum(MDB_DBS_FULL);
  setfield_enum(MDB_READERS_FULL);
  setfield_enum(MDB_TLS_FULL);
  setfield_enum(MDB_TXN_FULL);
  setfield_enum(MDB_LWRSOR_FULL);
  setfield_enum(MDB_PAGE_FULL);
  setfield_enum(MDB_MAP_RESIZED);
  setfield_enum(MDB_INCOMPATIBLE);

  setfield_enum(MDB_REVERSEKEY);
  setfield_enum(MDB_DUPSORT);
  setfield_enum(MDB_INTEGERKEY);
  setfield_enum(MDB_DUPFIXED);
  setfield_enum(MDB_INTEGERDUP);
  setfield_enum(MDB_REVERSEDUP);
  setfield_enum(MDB_CREATE);

  setfield_enum(MDB_FIXEDMAP);
  setfield_enum(MDB_NOSUBDIR);
  setfield_enum(MDB_RDONLY);
  setfield_enum(MDB_WRITEMAP);
  setfield_enum(MDB_NOMETASYNC);
  setfield_enum(MDB_NOSYNC);
  setfield_enum(MDB_MAPASYNC);
  setfield_enum(MDB_NOTLS);
  setfield_enum(MDB_NOLOCK);
  setfield_enum(MDB_NORDAHEAD);
  setfield_enum(MDB_NOMEMINIT);

  setfield_enum(MDB_FIRST);
  setfield_enum(MDB_FIRST_DUP);
  setfield_enum(MDB_GET_BOTH);
  setfield_enum(MDB_GET_BOTH_RANGE);
  setfield_enum(MDB_GET_LWRRENT);
  setfield_enum(MDB_GET_MULTIPLE);
  setfield_enum(MDB_LAST);
  setfield_enum(MDB_LAST_DUP);
  setfield_enum(MDB_NEXT);
  setfield_enum(MDB_NEXT_DUP);
  setfield_enum(MDB_NEXT_MULTIPLE);
  setfield_enum(MDB_NEXT_NODUP);
  setfield_enum(MDB_PREV);
  setfield_enum(MDB_PREV_DUP);
  setfield_enum(MDB_PREV_NODUP);
  setfield_enum(MDB_SET);
  setfield_enum(MDB_SET_KEY);
  setfield_enum(MDB_SET_RANGE );

  elw_register(L);
  txn_register(L);
  lwrsor_register(L);
  luaL_getmetatable(L,LIGHTNING);
  return 1;
}
