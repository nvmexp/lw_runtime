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

#ifndef LONG
typedef long LONG;
#endif

#ifndef LONG64
typedef  long long LONG64;
#endif

#ifndef ULONG64
typedef  unsigned long long ULONG64;
#endif

#ifndef PULONG
#define PULONG ULONG *
#endif

#ifndef PVOID
#define PVOID void *
#endif

#if !defined(BOOL)
#define BOOL S032
#define TRUE 1
#define FALSE 0
#endif

VOID   initLwWatch(VOID);
VOID   exitLwWatch(VOID);

VOID   WriteIoSpace(U016 offset, ULONG val, ULONG *size);
VOID   ReadIoSpace(U016 offset, ULONG *val, ULONG *size);
U032   readPhysicalMem(ULONG64 address, PVOID buf, ULONG size, U032 *sizer);
U032   writePhysicalMem(ULONG64 address, PVOID buf, ULONG size, U032 *sizew);
U032   readVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG sizer);
U032   writeVirtMem(ULONG64 address, PVOID buf, ULONG size, PULONG sizew);
LwU64 virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64 physToVirt(LwU64 physAddr, LwU32 flags);

int    lw_dprintf(const char *format, ...);
char*  lw_getelw(const char *str);
void   lw_hsort(void *, int, int, int (*cmp)(const void *, const void *));

void*  lw_malloc(unsigned int bytes);
void   lw_free(void *addr);

int    lw_strcmp(char *dest, const char *source);
char*  lw_strcpy(char *dest, const char *source);
char*  lw_strcat(char *dest, const char *source);
int    lw_strtoul(const char *nptr, char **endptr, int base);
int    lw_strlen(const char *string);
void   lw_memcpy(void *dest, void *src, unsigned int bytes);
void   lw_memset(void *dest, int c, unsigned int bytes);
int    lw_memcmp(const void *buf1, const void *buf2, unsigned int bytes);
int    lw_sprintf(char *str, const char *format, ...);
int    lw_sscanf(const char *str, const char *format, ...);

#define malloc  lw_malloc
#define free    lw_free

#define dprintf lw_dprintf
#define getelw  lw_getelw

#define qsort(b,n,w,c) lw_hsort(b,n,w,c)

#ifndef min
#define min(a,b)  ((a) < (b) ? (a) : (b))
#endif

U032 GetExpression(const char *args);
BOOL GetExpressionEx(const char *args, ULONG64 *val, char **endp);

#ifdef XP_PC
int strcasecmp(const char *s0, const char *s1);
int strncasecmp(const char *s0, const char *s1, int n);

void *realloc(void *current, size_t size);
#endif //XP_PC

#endif /* _SIM_LINUX_H_ */
