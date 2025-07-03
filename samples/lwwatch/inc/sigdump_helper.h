#ifndef __SIGDUMP_HELPER_H_INCLUDED__
#define __SIGDUMP_HELPER_H_INCLUDED__

#if defined(SIGDUMP_ENABLE)

#define POLL_TIMEOUT_COUNT           20

#define REG_WRITE_CHECK_FAILED      -1
#define REG_WRITE_SKIPPED            0
#define REG_WRITE_SUCCESSFUL         1

#define REG_READ_FAILED             -1
#define REG_READ_SKIPPED             0
#define REG_READ_SUCCESSFUL          1

#ifdef __cplusplus
extern "C"
{
#endif

LwBool checkRegWrite (pmcsSigdump *dump, LwU32 i);
int optimizedRegWriteWrapper (pmcsSigdump *dump, LwU32 i, LwBool optimization, LwBool checkWrites );
void clearRegWriteCache (void);

#ifdef __cplusplus
}
#endif

#endif // defined(SIGDUMP_ENABLE)

#endif // __SIGDUMP_HELPER_H_INCLUDED__
