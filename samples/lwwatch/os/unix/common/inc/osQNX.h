//*****************************************************
//
// QNX OS dependent routines...
//*****************************************************

#ifndef _OS_QNX_H_
#define _OS_QNX_H_

#ifndef _OS_H_
#error osQNX.h should not be #included directly, #include os.h instead
#endif

#ifndef __cdecl
#define __cdecl
#endif

#if !defined(BOOL)
#define BOOL LwS32
#define TRUE 1
#define FALSE 0
#endif

typedef struct
{
    LwU32 physaddr;
    LwU32 size;
    void *map;
} lwapt_t;

typedef struct
{ 
    LwU32 bus;
    LwU32 slot;
    LwU32 devid;
    lwapt_t regs;
    lwapt_t fb;
    lwapt_t inst;
} lwdev_t;

extern lwdev_t *pdev;

void   initLwWatch(void);

LW_STATUS   readPhysicalMem(LwU64 address, void * buf, LwU64 size, LwU64 *sizer);
LW_STATUS   writePhysicalMem(LwU64 address, void * buf, LwU64 size, LwU64 *sizew);
LW_STATUS   readVirtMem(LwU64 address, void * buf, LwU64 size, LwU64 *sizer);
LW_STATUS   writeVirtMem(LwU64 address, void * buf, LwU64 size, LwU64 *sizew);
LwU64  virtToPhys(LwU64 virtAddr, LwU32 pid);
LwU64  physToVirt(LwU64 physAddr, LwU32 flags);

int    lw_dprintf(const char *format, ...);
void   lw_hsort(void *, int, int, int (*cmp)(const void *, const void *));

#ifndef min
#define min(a,b)  ((a) < (b) ? (a) : (b))
#endif

LwU32 GetExpression(const char *args);
BOOL GetExpressionEx(const char *args, LwU64 *val, char **endp);

#endif /* _OS_QNX_H_ */
