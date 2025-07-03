/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#ifndef _TRACE_H_
#define _TRACE_H_

//#define ENABLE_TRACE

#ifdef __cplusplus
extern "C" {
#endif

#if defined ENABLE_TRACE && defined ANDROID
void trace_init();
void trace_begin(const char *name);
void trace_end();
#else
#define trace_init()
#define trace_begin(X) X
#define trace_end()
#endif
    
#ifdef __cplusplus
}
#endif

#endif
