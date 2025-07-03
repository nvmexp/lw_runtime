
#ifndef __PLUGINDEVICE_H__
#define __PLUGINDEVICE_H__

#include "lwca.h"
#include "lwda_runtime_api.h"
#include "LwvsDeviceList.h"
#include "DcgmError.h"

class PluginDevice
{
public:
    int             lwdaDeviceIdx;
    lwmlDevice_t    lwmlDevice;
    unsigned int    dcgmDeviceIndex;
    unsigned int    lwmlDeviceIndex;
    lwdaDeviceProp  lwdaDevProp;
    LwvsDevice     *lwvsDevice;
	std::string     warning;

    PluginDevice(unsigned int ndi, Plugin *p) : lwdaDeviceIdx(0), dcgmDeviceIndex(ndi), lwmlDeviceIndex(ndi),
                                                lwvsDevice(0), warning()
    {
		int st;
		lwmlReturn_t lwmlSt;
		char buf[256] = {0};
		lwmlPciInfo_t pciInfo;
		lwdaError_t lwSt;
		lwvsReturn_t lwvsReturn;

		memset(&this->lwdaDevProp, 0, sizeof(this->lwdaDevProp));

		this->lwvsDevice = new LwvsDevice(p);
		st = this->lwvsDevice->Init(this->lwmlDeviceIndex);
		if (st)
		{
			snprintf(buf, sizeof(buf), "Couldn't initialize LwvsDevice for GPU %u",
					 this->lwmlDeviceIndex);
			throw(std::runtime_error(buf));
		}

		this->lwmlDevice = this->lwvsDevice->GetLwmlDevice();

		/* Resolve lwca device index from lwml device index */
		lwmlSt = lwmlDeviceGetPciInfo(this->lwmlDevice, &pciInfo);
		if (lwmlSt != LWML_SUCCESS)
		{
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetPciInfo", lwmlErrorString(lwmlSt));
			warning = d.GetMessage();
			throw(d);
		}

		lwSt = lwdaDeviceGetByPCIBusId(&this->lwdaDeviceIdx, pciInfo.busId);
		if (lwSt != lwdaSuccess)
		{
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWDA_API, d, "lwdaDeviceGetByPCIBudId");
			snprintf(buf, sizeof(buf)-1, "'%s' for GPU %u, bus ID = %s",
					 lwdaGetErrorString(lwSt), this->lwmlDeviceIndex, pciInfo.busId);
            d.AddDetail(buf);
			warning = d.GetMessage();
            throw(d);
		}

#if 0 /* Don't lock clocks. Otherwise, we couldn't call this for non-root and would have different behavior
		 between root and nonroot */
		/* Turn off auto boost if it is present on the card and enabled */
		st = this->lwvsDevice->DisableAutoBoostedClocks();
		if(st)
			return -1;

		/* Try to maximize application clocks */
		unsigned int maxMemoryClock = (unsigned int)bgGlobals->testParameters->GetDouble(BG_STR_MAX_MEMORY_CLOCK);
		unsigned int maxGraphicsClock = (unsigned int)bgGlobals->testParameters->GetDouble(BG_STR_MAX_GRAPHICS_CLOCK);

		st = this->lwvsDevice->SetMaxApplicationClocks(maxMemoryClock, maxGraphicsClock);
		if(st)
			return -1;
#endif
	}

	~PluginDevice()
	{
		if (this->lwvsDevice)
		{
            this->lwvsDevice->RestoreState();
			delete lwvsDevice;
			this->lwvsDevice = 0;
		}
	}

};

#endif // __PLUGINDEVICE_H__
