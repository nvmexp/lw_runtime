/*
** LuaJIT frontend. Runs commands, scripts, read-eval-print (REPL) etc.
** Copyright (C) 2005-2017 Mike Pall. See Copyright Notice in luajit.h
**
** Major portions taken verbatim or adapted from the Lua interpreter.
** Copyright (C) 1994-2008 Lua.org, PUC-Rio. See Copyright Notice in lua.h
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define luajit_c

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luajit.h"

#include "lj_arch.h"

#if LJ_TARGET_POSIX
#include <unistd.h>
#define lua_stdin_is_tty()	isatty(0)
#elif LJ_TARGET_WINDOWS
#include <io.h>
#ifdef __BORLANDC__
#define lua_stdin_is_tty()	isatty(_fileno(stdin))
#else
#define lua_stdin_is_tty()	_isatty(_fileno(stdin))
#endif
#else
#define lua_stdin_is_tty()	1
#endif

#if !LJ_TARGET_CONSOLE
#include <signal.h>
#endif

static lua_State *globalL = NULL;
static const char *progname = LUA_PROGNAME;

/* ------------------------------------------------------------------------ */

#ifdef LUA_USE_READLINE

#include <readline/readline.h>
#include <readline/history.h>

#define lua_readline(L,b,p) ((void)L, ((b)=readline(p)) != NULL)
#define lua_saveline(L,idx) \
  if (lua_strlen(L,idx) > 0)  /* non-empty line? */ \
    add_history(lua_tostring(L, idx));  /* add it to history */
#define lua_freeline(L,b) ((void)L, free(b))

/*
** Lua 5.1.4 advanced readline support for the GNU readline and history
** libraries or compatible replacements.
**
** Author: Mike Pall.
** Maintainer: Sean Bolton (sean at smbolton dot com).
**
** Copyright (C) 2004-2006, 2011 Mike Pall. Same license as Lua. See lua.h.
**
** Advanced features:
** - Completion of keywords and global variable names.
** - Relwrsive and metatable-aware completion of variable names.
** - Context sensitive delimiter completion.
** - Save/restore of the history to/from a file (LUA_HISTORY elw variable).
** - Setting a limit for the size of the history (LUA_HISTSIZE elw variable).
** - Setting the app name to allow for $if lua ... $endif in ~/.inputrc.
**
** Start lua and try these (replace ~ with the TAB key):
**
** ~~
** fu~foo() ret~fa~end<CR>
** io~~~s~~~o~~~w~"foo\n")<CR>
**
** The ~~ are just for demonstration purposes (io~s~o~w~ suffices, of course).
**
** If you are used to zsh/tcsh-style completion support, try adding
** 'TAB: menu-complete' and 'C-d: possible-completions' to your ~/.inputrc.
**
** The patch has been successfully tested with:
**
** GNU    readline 2.2.1  (1998-07-17)
** GNU    readline 4.0    (1999-02-18) [harmless compiler warning]
** GNU    readline 4.3    (2002-07-16)
** GNU    readline 5.0    (2004-07-27)
** GNU    readline 5.1    (2005-12-07)
** GNU    readline 5.2    (2006-10-11)
** GNU    readline 6.0    (2009-02-20)
** GNU    readline 6.2    (2011-02-13)
** MacOSX libedit  2.11   (2008-07-12)
** NETBSD libedit  2.6.5  (2002-03-25)
** NETBSD libedit  2.6.9  (2004-05-01)
**
** Change Log:
** 2004-2006  Mike Pall   - original patch
** 2009/08/24 Sean Bolton - updated for GNU readline version 6
** 2011/12/14 Sean Bolton - fixed segfault when using Mac OS X libedit 2.11
*/

#include <ctype.h>

static char *lua_rl_hist;
static int lua_rl_histsize;

static lua_State *lua_rl_L;  /* User data is not passed to rl callbacks. */

/* Reserved keywords. */
static const char *const lua_rl_keywords[] = {
  "and", "break", "do", "else", "elseif", "end", "false",
  "for", "function", "if", "in", "local", "nil", "not", "or",
  "repeat", "return", "then", "true", "until", "while", NULL
};

static int valididentifier(const char *s)
{
  if (!(isalpha(*s) || *s == '_')) return 0;
  for (s++; *s; s++) if (!(isalpha(*s) || isdigit(*s) || *s == '_')) return 0;
  return 1;
}

/* Dynamically resizable match list. */
typedef struct {
  char **list;
  size_t idx, allocated, matchlen;
} dmlist;

/* Add prefix + string + suffix to list and compute common prefix. */
static int lua_rl_dmadd(dmlist *ml, const char *p, size_t pn, const char *s,
			int suf)
{
  char *t = NULL;

  if (ml->idx+1 >= ml->allocated &&
      !(ml->list = realloc(ml->list, sizeof(char *)*(ml->allocated += 32))))
    return -1;

  if (s) {
    size_t n = strlen(s);
    if (!(t = (char *)malloc(sizeof(char)*(pn+n+(suf?2:1))))) return 1;
    memcpy(t, p, pn);
    memcpy(t+pn, s, n);
    n += pn;
    t[n] = suf;
    if (suf) t[++n] = '\0';

    if (ml->idx == 0) {
      ml->matchlen = n;
    } else {
      size_t i;
      for (i = 0; i < ml->matchlen && i < n && ml->list[1][i] == t[i]; i++) ;
      ml->matchlen = i;  /* Set matchlen to common prefix. */
    }
  }

  ml->list[++ml->idx] = t;
  return 0;
}

/* Get __index field of metatable of object on top of stack. */
static int lua_rl_getmetaindex(lua_State *L)
{
  if (!lua_getmetatable(L, -1)) { lua_pop(L, 1); return 0; }

  /* prefer __metatable if it exists */
  lua_pushstring(L, "__metatable");
  lua_rawget(L, -2);
  if(lua_istable(L, -1))
  {
    lua_remove(L, -2);
    return 1;
  }
  else
    lua_pop(L, 1);

  lua_pushstring(L, "__index");
  lua_rawget(L, -2);
  lua_replace(L, -2);
  if (lua_isnil(L, -1) || lua_rawequal(L, -1, -2)) { lua_pop(L, 2); return 0; }
  lua_replace(L, -2);
  return 1;
}  /* 1: obj -- val, 0: obj -- */

/* Get field from object on top of stack. Avoid calling metamethods. */
static int lua_rl_getfield(lua_State *L, const char *s, size_t n)
{
  int i = 20;  /* Avoid infinite metatable loops. */
  do {
    if (lua_istable(L, -1)) {
      lua_pushlstring(L, s, n);
      lua_rawget(L, -2);
      if (!lua_isnil(L, -1)) { lua_replace(L, -2); return 1; }
      lua_pop(L, 1);
    }
  } while (--i > 0 && lua_rl_getmetaindex(L));
  lua_pop(L, 1);
  return 0;
}  /* 1: obj -- val, 0: obj -- */

/* Completion callback. */
static char **lua_rl_complete(const char *text, int start, int end)
{
  lua_State *L = lua_rl_L;
  dmlist ml;
  const char *s;
  size_t i, n, dot, loop;
  int savetop;

  if (!(text[0] == '\0' || isalpha(text[0]) || text[0] == '_')) return NULL;

  ml.list = NULL;
  ml.idx = ml.allocated = ml.matchlen = 0;

  savetop = lua_gettop(L);
  lua_pushvalue(L, LUA_GLOBALSINDEX);
  for (n = (size_t)(end-start), i = dot = 0; i < n; i++)
    if (text[i] == '.' || text[i] == ':') {
      if (!lua_rl_getfield(L, text+dot, i-dot))
	goto error;  /* Invalid prefix. */
      dot = i+1;  /* Points to first char after dot/colon. */
    }

  /* Add all matches against keywords if there is no dot/colon. */
  if (dot == 0)
    for (i = 0; (s = lua_rl_keywords[i]) != NULL; i++)
      if (!strncmp(s, text, n) && lua_rl_dmadd(&ml, NULL, 0, s, ' '))
	goto error;

  /* Add all valid matches from all tables/metatables. */
  loop = 0;  /* Avoid infinite metatable loops. */
  do {
    if (lua_istable(L, -1) &&
	(loop == 0 || !lua_rawequal(L, -1, LUA_GLOBALSINDEX)))
      for (lua_pushnil(L); lua_next(L, -2); lua_pop(L, 1))
	if (lua_type(L, -2) == LUA_TSTRING) {
	  s = lua_tostring(L, -2);
	  /* Only match names starting with '_' if explicitly requested. */
	  if (!strncmp(s, text+dot, n-dot) && valididentifier(s) &&
	      (*s != '_' || text[dot] == '_')) {
	    int suf = ' ';  /* Default suffix is a space. */
	    switch (lua_type(L, -1)) {
	    case LUA_TTABLE:	suf = '.'; break;  /* No way to guess ':'. */
	    case LUA_TFUNCTION:	suf = '('; break;
	    case LUA_TUSERDATA:
	      if (lua_getmetatable(L, -1)) { lua_pop(L, 1); suf = ':'; }
	      break;
	    }
	    if (lua_rl_dmadd(&ml, text, dot, s, suf)) goto error;
	  }
	}
  } while (++loop < 20 && lua_rl_getmetaindex(L));

  if (ml.idx == 0) {
error:
    lua_settop(L, savetop);
    return NULL;
  } else {
    /* list[0] holds the common prefix of all matches (may be ""). */
    /* If there is only one match, list[0] and list[1] will be the same. */
    if (!(ml.list[0] = (char *)malloc(sizeof(char)*(ml.matchlen+1))))
      goto error;
    memcpy(ml.list[0], ml.list[1], ml.matchlen);
    ml.list[0][ml.matchlen] = '\0';
    /* Add the NULL list terminator. */
    if (lua_rl_dmadd(&ml, NULL, 0, NULL, 0)) goto error;
  }

  lua_settop(L, savetop);
#if RL_READLINE_VERSION >= 0x0600
  rl_completion_suppress_append = 1;
#endif
  return ml.list;
}

/* Initialize readline library. */
static void lua_rl_init(lua_State *L)
{
  char *s;

  lua_rl_L = L;

  /* This allows for $if lua ... $endif in ~/.inputrc. */
  rl_readline_name = "lua";
  /* Break words at every non-identifier character except '.' and ':'. */
  rl_completer_word_break_characters =
    "\t\r\n !\"#$%&'()*+,-/;<=>?@[\\]^`{|}~";
  rl_completer_quote_characters = "\"'";
#if RL_READLINE_VERSION < 0x0600
  rl_completion_append_character = '\0';
#endif
  rl_attempted_completion_function = lua_rl_complete;
  rl_initialize();

  /* Start using history, optionally set history size and load history file. */
  using_history();
  if ((s = getelw("LUA_HISTSIZE")) &&
      (lua_rl_histsize = atoi(s))) stifle_history(lua_rl_histsize);
  if ((lua_rl_hist = getelw("LUA_HISTORY"))) read_history(lua_rl_hist);
}

/* Finalize readline library. */
static void lua_rl_exit(lua_State *L)
{
  /* Optionally save history file. */
  if (lua_rl_hist) write_history(lua_rl_hist);
}
#else
#define lua_readline(L,b,p) \
  ((void)L, fputs(p, stdout), fflush(stdout),  /* show prompt */ \
   fgets(b, LUA_MAXINPUT, stdin) != NULL)  /* get line */
#define lua_saveline(L,idx) { (void)L; (void)idx; }
#define lua_freeline(L,b) { (void)L; (void)b; }
#define lua_rl_init(L)		((void)L)
#define lua_rl_exit(L)		((void)L)
#endif

/* ------------------------------------------------------------------------ */

#if !LJ_TARGET_CONSOLE
static void lstop(lua_State *L, lua_Debug *ar)
{
  (void)ar;  /* unused arg. */
  lua_sethook(L, NULL, 0, 0);
  /* Avoid luaL_error -- a C hook doesn't add an extra frame. */
  luaL_where(L, 0);
  lua_pushfstring(L, "%sinterrupted!", lua_tostring(L, -1));
  lua_error(L);
}

static void laction(int i)
{
  signal(i, SIG_DFL); /* if another SIGINT happens before lstop,
			 terminate process (default action) */
  lua_sethook(globalL, lstop, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1);
}
#endif

static void print_usage(void)
{
  fputs("usage: ", stderr);
  fputs(progname, stderr);
  fputs(" [options]... [script [args]...].\n"
  "Available options are:\n"
  "  -e chunk  Execute string " LUA_QL("chunk") ".\n"
  "  -l name   Require library " LUA_QL("name") ".\n"
  "  -b ...    Save or list bytecode.\n"
  "  -j cmd    Perform LuaJIT control command.\n"
  "  -O[opt]   Control LuaJIT optimizations.\n"
  "  -i        Enter interactive mode after exelwting " LUA_QL("script") ".\n"
  "  -v        Show version information.\n"
  "  -E        Ignore environment variables.\n"
  "  --        Stop handling options.\n"
  "  -         Execute stdin and stop handling options.\n", stderr);
  fflush(stderr);
}

static void l_message(const char *pname, const char *msg)
{
  if (pname) { fputs(pname, stderr); fputc(':', stderr); fputc(' ', stderr); }
  fputs(msg, stderr); fputc('\n', stderr);
  fflush(stderr);
}

static int report(lua_State *L, int status)
{
  if (status && !lua_isnil(L, -1)) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error object is not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  return status;
}

static int traceback(lua_State *L)
{
  if (!lua_isstring(L, 1)) { /* Non-string error object? Try metamethod. */
    if (lua_isnoneornil(L, 1) ||
	!luaL_callmeta(L, 1, "__tostring") ||
	!lua_isstring(L, -1))
      return 1;  /* Return non-string error object. */
    lua_remove(L, 1);  /* Replace object by result of __tostring metamethod. */
  }
  luaL_traceback(L, L, lua_tostring(L, 1), 1);
  return 1;
}

static int docall(lua_State *L, int narg, int clear)
{
  int status;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushcfunction(L, traceback);  /* push traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
#if !LJ_TARGET_CONSOLE
  signal(SIGINT, laction);
#endif
  status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);
#if !LJ_TARGET_CONSOLE
  signal(SIGINT, SIG_DFL);
#endif
  lua_remove(L, base);  /* remove traceback function */
  /* force a complete garbage collection in case of errors */
  if (status != LUA_OK) lua_gc(L, LUA_GCCOLLECT, 0);
  return status;
}

static void print_version(void)
{
  fputs(LUAJIT_VERSION " -- " LUAJIT_COPYRIGHT ". " LUAJIT_URL "\n", stdout);
  fputs("\n", stdout);
  fputs(" _____              _     \n", stdout);
  fputs("|_   _|            | |    \n", stdout);
  fputs("  | | ___  _ __ ___| |__  \n", stdout);
  fputs("  | |/ _ \\| '__/ __| '_ \\ \n", stdout);
  fputs("  | | (_) | | | (__| | | |\n", stdout);
  fputs("  \\_/\\___/|_|  \\___|_| |_|\n", stdout);
  fputs("\n", stdout);
}

static void print_jit_status(lua_State *L)
{
  int n;
  const char *s;
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, -1, "jit");  /* Get jit.* module table. */
  lua_remove(L, -2);
  lua_getfield(L, -1, "status");
  lua_remove(L, -2);
  n = lua_gettop(L);
  lua_call(L, 0, LUA_MULTRET);
  fputs(lua_toboolean(L, n) ? "JIT: ON" : "JIT: OFF", stdout);
  for (n++; (s = lua_tostring(L, n)); n++) {
    putc(' ', stdout);
    fputs(s, stdout);
  }
  putc('\n', stdout);
}

static void createargtable(lua_State *L, char **argv, int argc, int argf)
{
  int i;
  lua_createtable(L, argc - argf, argf);
  for (i = 0; i < argc; i++) {
    lua_pushstring(L, argv[i]);
    lua_rawseti(L, -2, i - argf);
  }
  lua_setglobal(L, "arg");
}

static int dofile(lua_State *L, const char *name)
{
  int status = luaL_loadfile(L, name) || docall(L, 0, 1);
  return report(L, status);
}

static int dostring(lua_State *L, const char *s, const char *name)
{
  int status = luaL_loadbuffer(L, s, strlen(s), name) || docall(L, 0, 1);
  return report(L, status);
}

static int dolibrary(lua_State *L, const char *name)
{
  lua_getglobal(L, "require");
  lua_pushstring(L, name);
  return report(L, docall(L, 1, 1));
}

static const char* get_prompt(lua_State *L, int firstline)
{
  const char *p;
  lua_getfield(L, LUA_GLOBALSINDEX, firstline ? "_PROMPT" : "_PROMPT2");
  p = lua_tostring(L, -1);
  if (p == NULL) p = firstline ? LUA_PROMPT : LUA_PROMPT2;
  lua_pop(L, 1);  /* remove global */
  return p;
}

static int incomplete(lua_State *L, int status)
{
  if (status == LUA_ERRSYNTAX) {
    size_t lmsg;
    const char *msg = lua_tolstring(L, -1, &lmsg);
    const char *tp = msg + lmsg - (sizeof(LUA_QL("<eof>")) - 1);
    if (strstr(msg, LUA_QL("<eof>")) == tp) {
      lua_pop(L, 1);
      return 1;
    }
  }
  return 0;  /* else... */
}

static int pushline(lua_State *L, int firstline)
{
  char buf[LUA_MAXINPUT];
  char *b = buf;
  const char *prmt = get_prompt(L, firstline);
  if (lua_readline(L, b, prmt)) {
    size_t len = strlen(b);
    if (len > 0 && b[len-1] == '\n')
      b[len-1] = '\0';
    if (firstline && b[0] == '=')
      lua_pushfstring(L, "return %s", b+1);
    else
      lua_pushstring(L, b);
    lua_freeline(L, b);
    return 1;
  }
  return 0;
}

static int loadline(lua_State *L)
{
  int status;
  lua_settop(L, 0);
  if (!pushline(L, 1))
    return -1;  /* no input */
  for (;;) {  /* repeat until gets a complete line */
    status = luaL_loadbuffer(L, lua_tostring(L, 1), lua_strlen(L, 1), "=stdin");
    if (!incomplete(L, status)) break;  /* cannot try to add lines? */
    if (!pushline(L, 0))  /* no more input? */
      return -1;
    lua_pushliteral(L, "\n");  /* add a new line... */
    lua_insert(L, -2);  /* ...between the two lines */
    lua_concat(L, 3);  /* join them */
  }
  lua_saveline(L, 1);
  lua_remove(L, 1);  /* remove line */
  return status;
}

static void dotty(lua_State *L)
{
  int status;
  const char *oldprogname = progname;
  progname = NULL;
  lua_rl_init(L);
  while ((status = loadline(L)) != -1) {
    if (status == LUA_OK) status = docall(L, 0, 0);
    report(L, status);
    if (status == LUA_OK && lua_gettop(L) > 0) {  /* any result to print? */
      lua_getglobal(L, "print");
      lua_insert(L, 1);
      if (lua_pcall(L, lua_gettop(L)-1, 0, 0) != 0)
	l_message(progname,
	  lua_pushfstring(L, "error calling " LUA_QL("print") " (%s)",
			      lua_tostring(L, -1)));
    }
  }
  lua_settop(L, 0);  /* clear stack */
  fputs("\n", stdout);
  fflush(stdout);
  lua_rl_exit(L);
  progname = oldprogname;
}

static int handle_script(lua_State *L, char **argx)
{
  int status;
  const char *fname = argx[0];
  if (strcmp(fname, "-") == 0 && strcmp(argx[-1], "--") != 0)
    fname = NULL;  /* stdin */
  status = luaL_loadfile(L, fname);
  if (status == LUA_OK) {
    /* Fetch args from arg table. LUA_INIT or -e might have changed them. */
    int narg = 0;
    lua_getglobal(L, "arg");
    if (lua_istable(L, -1)) {
      do {
	narg++;
	lua_rawgeti(L, -narg, narg);
      } while (!lua_isnil(L, -1));
      lua_pop(L, 1);
      lua_remove(L, -narg);
      narg--;
    } else {
      lua_pop(L, 1);
    }
    status = docall(L, narg, 0);
  }
  return report(L, status);
}

/* Load add-on module. */
static int loadjitmodule(lua_State *L)
{
  lua_getglobal(L, "require");
  lua_pushliteral(L, "jit.");
  lua_pushvalue(L, -3);
  lua_concat(L, 2);
  if (lua_pcall(L, 1, 1, 0)) {
    const char *msg = lua_tostring(L, -1);
    if (msg && !strncmp(msg, "module ", 7))
      goto nomodule;
    return report(L, 1);
  }
  lua_getfield(L, -1, "start");
  if (lua_isnil(L, -1)) {
  nomodule:
    l_message(progname,
	      "unknown luaJIT command or jit.* modules not installed");
    return 1;
  }
  lua_remove(L, -2);  /* Drop module table. */
  return 0;
}

/* Run command with options. */
static int runcmdopt(lua_State *L, const char *opt)
{
  int narg = 0;
  if (opt && *opt) {
    for (;;) {  /* Split arguments. */
      const char *p = strchr(opt, ',');
      narg++;
      if (!p) break;
      if (p == opt)
	lua_pushnil(L);
      else
	lua_pushlstring(L, opt, (size_t)(p - opt));
      opt = p + 1;
    }
    if (*opt)
      lua_pushstring(L, opt);
    else
      lua_pushnil(L);
  }
  return report(L, lua_pcall(L, narg, 0, 0));
}

/* JIT engine control command: try jit library first or load add-on module. */
static int dojitcmd(lua_State *L, const char *cmd)
{
  const char *opt = strchr(cmd, '=');
  lua_pushlstring(L, cmd, opt ? (size_t)(opt - cmd) : strlen(cmd));
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, -1, "jit");  /* Get jit.* module table. */
  lua_remove(L, -2);
  lua_pushvalue(L, -2);
  lua_gettable(L, -2);  /* Lookup library function. */
  if (!lua_isfunction(L, -1)) {
    lua_pop(L, 2);  /* Drop non-function and jit.* table, keep module name. */
    if (loadjitmodule(L))
      return 1;
  } else {
    lua_remove(L, -2);  /* Drop jit.* table. */
  }
  lua_remove(L, -2);  /* Drop module name. */
  return runcmdopt(L, opt ? opt+1 : opt);
}

/* Optimization flags. */
static int dojitopt(lua_State *L, const char *opt)
{
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, -1, "jit.opt");  /* Get jit.opt.* module table. */
  lua_remove(L, -2);
  lua_getfield(L, -1, "start");
  lua_remove(L, -2);
  return runcmdopt(L, opt);
}

/* Save or list bytecode. */
static int dobytecode(lua_State *L, char **argv)
{
  int narg = 0;
  lua_pushliteral(L, "bcsave");
  if (loadjitmodule(L))
    return 1;
  if (argv[0][2]) {
    narg++;
    argv[0][1] = '-';
    lua_pushstring(L, argv[0]+1);
  }
  for (argv++; *argv != NULL; narg++, argv++)
    lua_pushstring(L, *argv);
  report(L, lua_pcall(L, narg, 0, 0));
  return -1;
}

/* check that argument has no extra characters at the end */
#define notail(x)	{if ((x)[2] != '\0') return -1;}

#define FLAGS_INTERACTIVE	1
#define FLAGS_VERSION		2
#define FLAGS_EXEC		4
#define FLAGS_OPTION		8
#define FLAGS_NOELW		16

static int collectargs(char **argv, int *flags)
{
  int i;
  for (i = 1; argv[i] != NULL; i++) {
    if (argv[i][0] != '-')  /* Not an option? */
      return i;
    switch (argv[i][1]) {  /* Check option. */
    case '-':
      notail(argv[i]);
      return i+1;
    case '\0':
      return i;
    case 'i':
      notail(argv[i]);
      *flags |= FLAGS_INTERACTIVE;
      /* fallthrough */
    case 'v':
      notail(argv[i]);
      *flags |= FLAGS_VERSION;
      break;
    case 'e':
      *flags |= FLAGS_EXEC;
    case 'j':  /* LuaJIT extension */
    case 'l':
      *flags |= FLAGS_OPTION;
      if (argv[i][2] == '\0') {
	i++;
	if (argv[i] == NULL) return -1;
      }
      break;
    case 'O': break;  /* LuaJIT extension */
    case 'b':  /* LuaJIT extension */
      if (*flags) return -1;
      *flags |= FLAGS_EXEC;
      return i+1;
    case 'E':
      *flags |= FLAGS_NOELW;
      break;
    default: return -1;  /* invalid option */
    }
  }
  return i;
}

static int runargs(lua_State *L, char **argv, int argn)
{
  int i;
  for (i = 1; i < argn; i++) {
    if (argv[i] == NULL) continue;
    lua_assert(argv[i][0] == '-');
    switch (argv[i][1]) {
    case 'e': {
      const char *chunk = argv[i] + 2;
      if (*chunk == '\0') chunk = argv[++i];
      lua_assert(chunk != NULL);
      if (dostring(L, chunk, "=(command line)") != 0)
	return 1;
      break;
      }
    case 'l': {
      const char *filename = argv[i] + 2;
      if (*filename == '\0') filename = argv[++i];
      lua_assert(filename != NULL);
      if (dolibrary(L, filename))
	return 1;
      break;
      }
    case 'j': {  /* LuaJIT extension. */
      const char *cmd = argv[i] + 2;
      if (*cmd == '\0') cmd = argv[++i];
      lua_assert(cmd != NULL);
      if (dojitcmd(L, cmd))
	return 1;
      break;
      }
    case 'O':  /* LuaJIT extension. */
      if (dojitopt(L, argv[i] + 2))
	return 1;
      break;
    case 'b':  /* LuaJIT extension. */
      return dobytecode(L, argv+i);
    default: break;
    }
  }
  return LUA_OK;
}

static int handle_luainit(lua_State *L)
{
#if LJ_TARGET_CONSOLE
  const char *init = NULL;
#else
  const char *init = getelw(LUA_INIT);
#endif
  if (init == NULL)
    return LUA_OK;
  else if (init[0] == '@')
    return dofile(L, init+1);
  else
    return dostring(L, init, "=" LUA_INIT);
}

static struct Smain {
  char **argv;
  int argc;
  int status;
} smain;

static int pmain(lua_State *L)
{
  struct Smain *s = &smain;
  char **argv = s->argv;
  int argn;
  int flags = 0;
  globalL = L;
  if (argv[0] && argv[0][0]) progname = argv[0];

  LUAJIT_VERSION_SYM();  /* Linker-enforced version check. */

  argn = collectargs(argv, &flags);
  if (argn < 0) {  /* Invalid args? */
    print_usage();
    s->status = 1;
    return 0;
  }

  if ((flags & FLAGS_NOELW)) {
    lua_pushboolean(L, 1);
    lua_setfield(L, LUA_REGISTRYINDEX, "LUA_NOELW");
  }

  /* Stop collector during library initialization. */
  lua_gc(L, LUA_GCSTOP, 0);
  luaL_openlibs(L);
  lua_gc(L, LUA_GCRESTART, -1);

  createargtable(L, argv, s->argc, argn);

  if (!(flags & FLAGS_NOELW)) {
    s->status = handle_luainit(L);
    if (s->status != LUA_OK) return 0;
  }

  if ((flags & FLAGS_VERSION)) print_version();

  s->status = runargs(L, argv, argn);
  if (s->status != LUA_OK) return 0;

  if (s->argc > argn) {
    s->status = handle_script(L, argv + argn);
    if (s->status != LUA_OK) return 0;
  }

  if ((flags & FLAGS_INTERACTIVE)) {
    print_jit_status(L);
    dotty(L);
  } else if (s->argc == argn && !(flags & (FLAGS_EXEC|FLAGS_VERSION))) {
    if (lua_stdin_is_tty()) {
      print_version();
      print_jit_status(L);
      dotty(L);
    } else {
      dofile(L, NULL);  /* Exelwtes stdin as a file. */
    }
  }
  return 0;
}

int main(int argc, char **argv)
{
  int status;
  lua_State *L = lua_open();
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  smain.argc = argc;
  smain.argv = argv;
  status = lua_cpcall(L, pmain, NULL);
  report(L, status);
  lua_close(L);
  return (status || smain.status > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}

