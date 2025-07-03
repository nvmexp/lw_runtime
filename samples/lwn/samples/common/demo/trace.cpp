/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/


#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>

#include <trace.h>

#if defined ENABLE_TRACE && defined ANDROID

extern "C"
{

#define ATRACE_MESSAGE_LEN 256
int     atrace_marker_fd = -1;
int active = 0;

void trace_init()
{
  atrace_marker_fd = open("/sys/kernel/debug/tracing/trace_marker", O_WRONLY);
  if (atrace_marker_fd == -1)   { /* do error handling */ }
}

void trace_begin(const char *name)
{
    if (active)
        trace_end();

    char buf[ATRACE_MESSAGE_LEN];
    int len = snprintf(buf, ATRACE_MESSAGE_LEN, "B|%d|%s", getpid(), name);
    write(atrace_marker_fd, buf, len);
    active = 1;
}

void trace_end()
{
    if (!active)
        return;

    active = 0;
    char c = 'E';
    write(atrace_marker_fd, &c, 1);
}

}

#endif
