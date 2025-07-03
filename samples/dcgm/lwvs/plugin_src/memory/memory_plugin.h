#ifndef MEMORY_H
#define MEMORY_H 

#include "Plugin.h"
#include "PluginStrings.h"
#include "Memory_wrapper.h"
#include "TestParameters.h"
#include "LwvsDeviceList.h"
#include "DcgmRecorder.h"
#include <lwca.h>

/*****************************************************************************/
/* String constants */

/* Public parameters - we expect users to change these */

/* Public sub-test parameters. These apply to some sub tests and not others */

/* Private parameters - we can have users change these but we don't need to
 * document them until users will change them
 */

/*****************************************************************************/
/* Sub tests */

/*****************************************************************************/
/* Struct to hold global information for this test */
typedef struct mem_globals_t
{
    TestParameters *testParameters; /* Parameters passed in from the framework */

    Memory *memory;   /* Plugin handle for setting status */

    int lwmlInitialized; /* Has lwmlInit been called yet? */
    int lwdaInitialized; /* Has lwInit been called yet? */
    unsigned int lwmlGpuIndex; /* LWML gpu index for the GPU */
    lwmlDevice_t lwmlDevice; /* LWML device handle for the GPU */
    LWdevice lwDevice; /* Lwca device handle to dispatch work to */

    LWcontext lwCtx; /* Lwca context to dispatch work to */
    int lwCtxCreated; /* Does lwCtx need to be freed? */

    LwvsDevice *lwvsDevice; /* LWVS device object for controlling/querying this device */

    DcgmRecorder *m_dcgmRecorder;

} mem_globals_t, *mem_globals_p;

/*****************************************************************************/
int main_entry(unsigned int gpu, Memory *memObj, TestParameters *testParameters);

/*****************************************************************************/

#endif //MEMORY_H
