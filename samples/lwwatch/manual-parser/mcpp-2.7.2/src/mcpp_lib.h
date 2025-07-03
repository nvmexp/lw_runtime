/* mcpp_lib.h: declarations of libmcpp exported (visible) functions */
#ifndef _MCPP_LIB_H
#define _MCPP_LIB_H

#include <stdio.h>

#ifndef _MCPP_OUT_H
#include    "mcpp_out.h"            /* declaration of OUTDEST   */
#endif

#ifndef _MCPP_CALLBACKS_H
#include    "mcpp_callbacks.h"      /* declaration of CALLBACKS */
#endif

#if _WIN32 || _WIN64 || __CYGWIN__ || __CYGWIN64__ || __MINGW32__   \
            || __MINGW64__
#if     DLL_EXPORT || (__CYGWIN__ && PIC)
#define DLL_DECL    __declspec( dllexport)
#elif   DLL_IMPORT
#define DLL_DECL    __declspec( dllimport)
#else
#define DLL_DECL
#endif
#else
#define DLL_DECL
#endif

extern DLL_DECL int     mcpp_lib_main( int argc, char ** argv);
extern DLL_DECL void    mcpp_reset_def_out_func( void);
extern DLL_DECL void    mcpp_set_out_func(
                    int (* func_fputc)  ( int c, OUTDEST od),
                    int (* func_fputs)  ( const char * s, OUTDEST od),
                    int (* func_fprintf)( OUTDEST od, const char * format, ...)
                    );
extern DLL_DECL void    mcpp_use_out_mem_buffers( int tf);
extern DLL_DECL char *  mcpp_get_out_mem_buffer( OUTDEST od);
extern DLL_DECL void    mcpp_set_in_mem_buffer( int tf, const char *, size_t);
extern DLL_DECL void    mcpp_set_callbacks( int tf, CALLBACKS *);
extern DLL_DECL void    mcpp_set_persistence( int tf);
extern DLL_DECL const char *mcpp_get_last_comment( void);
#endif  /* _MCPP_LIB_H  */
