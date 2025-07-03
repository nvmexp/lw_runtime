//*****************************************************
//
// lwwatch linux gdb module Extension
// simLinux.h
// Linux Sim gdb OS dependent routines...
//*****************************************************

#ifndef _SIM_LINUX_H_
#define _SIM_LINUX_H_

#ifndef _OS_H_
#error simLinux.h should not be #included directly, #include os.h instead
#endif

//#include <linux/types.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdlib.h>

#ifndef __cdecl
#define __cdecl
#endif

#if !defined(BOOL)
#define BOOL LwS32
#define TRUE 1
#define FALSE 0
#endif

void   initLwWatch(void);
void   exitLwWatch(void);

LW_STATUS   readPhysicalMem(LwU64 address, void *buf, LwU64 size, LwU64 *sizer);
LW_STATUS   writePhysicalMem(LwU64 address, void *buf, LwU64 size, LwU64 *sizew);
LW_STATUS   readVirtMem(LwU64 address, void *buf, LwU64 size, LwU64 *sizer);
LW_STATUS   writeVirtMem(LwU64 address, void *buf, LwU64 size, LwU64 *sizew);
LwU64 virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64 physToVirt(LwU64 physAddr, LwU32 flags);

int    lw_dprintf(const char *format, ...);
void   lw_hsort(void *, int, int, int (*cmp)(const void *, const void *));

void   lw_memcpy(void *dest, void *src, unsigned int bytes);
void   lw_memset(void *dest, int c, unsigned int bytes);
int    lw_memcmp(const void *buf1, const void *buf2, unsigned int bytes);
int    lw_sprintf(char *str, const char *format, ...);
int    lw_sscanf(const char *str, const char *format, ...);

#define dprintf lw_dprintf

#define qsort(b,n,w,c) lw_hsort(b,n,w,c)

#ifndef min
#define min(a,b)  ((a) < (b) ? (a) : (b))
#endif

LwU32 GetExpression(const char *args);
BOOL GetExpressionEx(const char *args, LwU64 *val, char **endp);

#ifdef XP_PC
int strcasecmp(const char *s0, const char *s1);
int strncasecmp(const char *s0, const char *s1, int n);

void *realloc(void *current, size_t size);
#endif //XP_PC

#endif /* _SIM_LINUX_H_ */
