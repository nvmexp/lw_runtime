#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_fields.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dlfcn.h>

static void *lib_handle = NULL;


static const char wrongLibraryMessage[] = 
    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "WARNING:\n"
    "\n"
    "Wrong libdcgm.so installed\n"
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

static void dcgmCheckLinkedLibrary(void) {
#if defined(LW_LINUX)
    FILE *f;
    char command_line[512];
    char buff[256];
    int ret;

    /**
     * Check if the commands (lsof, tr and cut) exists on the underlying platform
     */
    sprintf(command_line, "command -v lsof >/dev/null 2>&1 && command -v tr >/dev/null 2>&1" 
            " && command -v cut >/dev/null 2>&1  && echo 'true' || echo 'false'");
    if (!(f = popen(command_line, "r"))) {
        return;
    }

    while (fgets(buff, sizeof (buff), f) != NULL) {
        char *p;
        if ((p = strstr(buff, "false")) != NULL) {
            return;
        }
    }
    
    pclose(f);

    pid_t pid = getpid();
    sprintf(command_line, "lsof -p %d | tr -s ' ' | cut -d ' ' -f 9 ", pid);
    if (!(f = popen(command_line, "r"))) {
        return;
    }
    
    while(fgets(buff, sizeof(buff), f) != NULL) {
        char *p;
        if ((p = strstr(buff, "libdcgm")) != NULL) {
            printf("Linked to libdcgm library at wrong path : %s\n", buff);
            break;
        }
    }
    
    pclose(f);
#endif
}

#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define PROBE(x) x, 1

#define CHECK(...) CHECK_N(__VA_ARGS__, 0)
#define CHECK_N(x, n, ...) n

#define IIF_NOT(c) PRIMITIVE_CAT(IIF_NOT_, c)
#define IIF_NOT_0(t, ...) t
#define IIF_NOT_1(t, ...) __VA_ARGS__

#define API_SKIPPED_PROBE(newName) API_SKIPPED_PROBE_PROXY(SKIP_##newName)
#define API_SKIPPED_PROBE_PROXY(...) API_SKIPPED_PROBE_PRIMITIVE(__VA_ARGS__)
#define API_SKIPPED_PROBE_PRIMITIVE(x) API_SKIPPED_PROBE_COMBINE_ x
#define API_SKIPPED_PROBE_COMBINE_(...) PROBE(~)

#define IS_API_SKIPPED(newName) CHECK(API_SKIPPED_PROBE(newName))

#define DCGM_DYNAMIC_WRAP_PROXY(newName, libFunctionName, argtypes, ...)                                               \
    dcgmReturn_t newName argtypes                                                                                      \
    {                                                                                                                  \
        dcgmReturn_t(*fn) argtypes;                                                                                    \
        if (lib_handle)                                                                                                \
        {                                                                                                              \
            fn = dlsym(lib_handle, __FUNCTION__);                                                                      \
            return (*fn)(__VA_ARGS__);                                                                                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            printf("%s", wrongLibraryMessage);                                                                         \
            return DCGM_ST_UNINITIALIZED;                                                                              \
        }                                                                                                              \
    }

#define DCGM_DYNAMIC_WRAP(newName, libFunctionName, argtypes, ...)                                                     \
    IIF_NOT(IS_API_SKIPPED(newName))                                                                                   \
    (DCGM_DYNAMIC_WRAP_PROXY(newName, libFunctionName, argtypes, ##__VA_ARGS__), /*nothing*/                           \
    )

#define DCGM_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)                                              \
    DCGM_DYNAMIC_WRAP(dcgmFuncname, dcgmFuncname, argtypes, ##__VA_ARGS__)

// You may '#define SKIP_symbolname ()' to make DCGM_ENTRY_POINT for the 'symbolname' a nop
#define SKIP_dcgmInit ()
#define SKIP_dcgmStartEmbedded ()
#define SKIP_dcgmStopEmbedded ()
#define SKIP_dcgmShutdown ()

//#define DCGM_INT_ENTRY_POINT(dcgmFuncname, tsapiFuncname, argtypes, fmt, ...)
#include "entry_point.h"
//#undef DCGM_INT_ENTRY_POINT
#undef DCGM_ENTRY_POINT

dcgmReturn_t dcgmInit(void)
{
    dcgmReturn_t (*fn)(void);
    lib_handle = dlopen("libdcgm.so.1", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmInit");
        return (*fn)();
    } else 
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmOperationMode_t, dcgmHandle_t *);
    lib_handle = dlopen("libdcgm.so.1", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStartEmbedded");
        return (*fn)(opMode, pDcgmHandle);
    } else
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    dcgmReturn_t (*fn)(dcgmHandle_t);
    lib_handle = dlopen("libdcgm.so.1", RTLD_LAZY);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmStopEmbedded");
        return (*fn)(pDcgmHandle);
    } else
    {
        printf("%s", wrongLibraryMessage);
        dcgmCheckLinkedLibrary();
        return DCGM_ST_UNINITIALIZED;
    }
}

dcgmReturn_t dcgmShutdown(void)
{
    dcgmReturn_t dcgmResult;
    dcgmReturn_t (*fn)(void);
    if (lib_handle)
    {
        fn = dlsym(lib_handle, "dcgmShutdown");
        dcgmResult = (*fn)();
    } else 
    {
        printf("%s", wrongLibraryMessage);
        return DCGM_ST_UNINITIALIZED;
    }
    if (lib_handle) dlclose(lib_handle);
    return dcgmResult;
}

#if 0  
// left here in case we ever want to include internal apis... must also include dcgm_*interal.h above
DCGM_DYNAMIC_WRAP(dcgmInternalGetExportTable, dcgmInternalGetExportTable,
        (const void **ppExportTable, const dcgmUuid_t *pExportTableId),
        ppExportTable, pExportTableId)
#endif

const char* dcgmErrorString(dcgmReturn_t result)
{
    return wrongLibraryMessage;
}
