#ifndef BUSGRINDMAIN_H
#define BUSGRINDMAIN_H

#include "Plugin.h"
#include <PluginDevice.h>
#include "BusGrind.h"
#include "TestParameters.h"
#include "lwda_runtime.h"
#include "PluginStrings.h"
#include "LwvsDeviceList.h"
#include "DcgmRecorder.h"

#define BG_MAX_GPUS     16

/*****************************************************************************/

/* Struct to hold global information for this test */
class BusGrindGlobals
{
public:
    TestParameters *testParameters; /* Parameters passed in from the framework */

    /* Cached parameters */
    bool test_pinned;
    bool test_unpinned;
    bool test_p2p_on;
    bool test_p2p_off;

    BusGrind *busGrind;   /* Plugin handle for setting status */

    int lwmlInitialized;   /* Has lwmlInit been called successfully yet? */

    std::vector<PluginDevice *> gpu; /* Per-gpu information */
    DcgmRecorder *m_dcgmRecorder;

    bool m_dcgmCommErrorOclwrred;
    bool m_printedConlwrrentGpuErrorMessage;

    // None of the initializations matter because the entire class is zero'd out in BusGrindMain.cpp
    BusGrindGlobals() : testParameters(0), test_pinned(false), test_unpinned(false), test_p2p_on(false),
                        test_p2p_off(false), busGrind(0), lwmlInitialized(0), gpu(), m_dcgmRecorder(0), 
                        m_dcgmCommErrorOclwrred(false), m_printedConlwrrentGpuErrorMessage(false)
    {
    }

};

/*****************************************************************************/
int main_entry(const std::vector<unsigned int> &gpuList, BusGrind *busGrind,
               TestParameters *testParameters);

/*****************************************************************************/

#endif //BUSGRINDMAIN_H
