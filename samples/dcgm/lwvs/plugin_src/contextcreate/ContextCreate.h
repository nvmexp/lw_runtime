#ifndef CONTEXTCREATE_H
#define CONTEXTCREATE_H

#include "lwca.h"
#include "Plugin.h"
#include "PluginDevice.h"
#include "lwml.h"
#include "DcgmRecorder.h"
#include "timelib.h"
#include "DcgmHandle.h"
#include "dcgm_structs.h"

#define CTX_CREATED 0x0
#define CTX_SKIP    0x1
#define CTX_FAIL    0x2

class ContextCreateDevice : public PluginDevice
{
public:
    LWdevice      lwDevice;
    LWcontext     lwContext;

    ContextCreateDevice(unsigned int ndi, Plugin *p, DcgmHandle &handle) : PluginDevice(ndi, p)
    {
        char buf[256] = {0};
        const char *errorString;
        dcgmDeviceAttributes_t attr;
        memset(&attr, 0, sizeof(attr));
        attr.version = dcgmDeviceAttributes_version1;

        dcgmReturn_t ret = dcgmGetDeviceAttributes(handle.GetHandle(), ndi, &attr);

        if (ret != DCGM_ST_OK)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmGetDeviceAttributes");
            snprintf(buf, sizeof(buf), "for GPU %u", this->lwmlDeviceIndex);
            d.AddDetail(buf);
            throw d;
        }

        LWresult lwSt = lwInit(0);
        
        if (lwSt)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwInit");
            lwGetErrorString(lwSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d)", errorString, static_cast<int>(lwSt));
                d.AddDetail(buf);
            }
            throw d;
        }
        
        lwSt = lwDeviceGetByPCIBusId(&lwDevice, attr.identifiers.pciBusId);
        if (lwSt)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwDeviceGetByPCIBusId");
            lwGetErrorString(lwSt, &errorString);
            if (errorString == NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u",
                         errorString, static_cast<int>(lwSt), this->lwmlDeviceIndex);
                d.AddDetail(buf);
            }
            throw d;
        }
    }

};

class ContextCreate 
{
public:
    ContextCreate(TestParameters *testParameters, Plugin *plugin);
    ~ContextCreate();

    /*************************************************************************/
    /*
     * Initialize devices and resources for this plugin.
     */
    std::string Init(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object, including memory and 
     * file handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Run ContextCreate tests
     *
     * Returns 0 on success
     *        <0 on failure or early exit
     */
    int Run(const std::vector<unsigned int> &gpuList);

    /*************************************************************************/
    /*
     * Attempt to create a context for each GPU in the list
     *
     * @return:  0 on success
     *           1 for skipping
     *          -1 for failure to create
     */
    int CanCreateContext();

private:

    /*************************************************************************/
    /*
     * GpusAreNonExclusive()
     *
     * Returns: true if the compute mode allows us to run this test
     */
    bool GpusAreNonExclusive();

    /*************************************************************************/
    Plugin         *m_plugin;              /* Which plugin we're a part of. This is a paramter to the instance */
    TestParameters *m_testParameters;      /* The test parameters for this run of LWVS */
    std::vector<ContextCreateDevice *> m_device; /* Per-device data */
    DcgmRecorder   *m_dcgmRecorder;
    DcgmHandle      m_dcgmHandle;
    DcgmGroup       m_dcgmGroup;
};

#endif
