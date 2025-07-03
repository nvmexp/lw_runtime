#ifndef LWVSDEVICELIST_H
#define LWVSDEVICELIST_H

#include <vector>
#include "Plugin.h"
#include "lwml.h"

/*****************************************************************************/
#define LWVS_DL_AFFINITY_SIZE 16

/* Struct for storing the device's state */
typedef struct lwvs_device_state_t
{
    int populated; /* Has this struct already been populated? Prevents overwriting
                      1=Yes. 0=No */

    int appClocksSupported; /* 1 yes. 0 no. Following fields are only valid for == 1 */
    unsigned int appClockGraphics; /* LWML_CLOCK_GRAPHICS */
    unsigned int appClockMemory;   /* LWML_CLOCK_MEM */

    int computeModeSupported; /* 1 yes. 0 no. Following fields are only valid for == 1 */
    lwmlComputeMode_t computeMode;

    int autoBoostSupported; /* 1 yes. 0 no. are autoboost features supported? */
    lwmlEnableState_t autoBoostEnabled;
    lwmlEnableState_t autoBoostDefaultEnabled;

    int cpuAffinitySupported; /* 1 yes. 0 no. Are cpu affinity functions supported? */
    int cpuAffinityWasSet;    /* 1 yes 0 no. Have we changed cpu affinity? */
    unsigned long idealCpuAffinity[LWVS_DL_AFFINITY_SIZE];

} lwvs_device_state_t, *lwvs_device_state_p;

/*****************************************************************************/
/* Class to represent a single LWVS GPU */
class LwvsDevice
{
public:

    LwvsDevice(Plugin *plugin);
    ~LwvsDevice();

    /*************************************************************************/
    /* Initialize this object from its LWML index, saving the current
     * state in the process
     *
     * Returns 0 on success.
     *         Nonzero on failure
     **/
    int Init(int lwmlDeviceIndex);

    /*************************************************************************/
    /*
     * Back up the state of this device to the passed in struct for later
     * restoration or comparison to an already-saved struct
     *
     * Returns 0 on success.
     *        !0 on error
     *
     */
    int SaveState(lwvs_device_state_t *savedState);


    /*************************************************************************/
    /*
     * Get the LWML device handle associated with this device
     *
     */
    lwmlDevice_t GetLwmlDevice();

    /*************************************************************************/
    /*
     * Get the LWML device index associated with this device
     *
     */
    unsigned int GetLwmlIndex();

    /*************************************************************************/
    /*
     * Try to disable auto boosted clocks for this device
     *
     * Returns 0 on success.
     *        !0 on error
     *
     */
    int DisableAutoBoostedClocks();

    /*************************************************************************/
    /*
     * Try to set maximum application clocks for this device
     *
     * maxMemClockMhz: Maximum memory clock frequence in MHZ that we will not 
     *                 set memory clocks higher than. 0 = no limit
     * maxGraphicsClockMhz: Maximum graphics clock frequency in MHZ that we will
     *                      not set graphics clocks higher than. 0 = no limit
     *
     * Returns 0 on success.
     *        !0 on error
     */
    int SetMaxApplicationClocks(unsigned int maxMemClockMhz, unsigned int maxGraphicsClockMhz);

    /*************************************************************************/
    int SetCpuAffinity();
    /*
     * Try to set the CPU affinity of this device
     *
     * Returns 0 on success.
     *        !0 on error
     */

    /*************************************************************************/
    /*
     * Restore any state that has changed since the object was instantiated
     *
     * Returns: 0 if there was no state to restore
     *         <0 on error
     *         >0 if state was restored
     *
     */
    int RestoreState();

    /*************************************************************************/
    /*
     * Reports on whether this LWVS device is lwrrently considered idle
     *
     * idleTemp: Temperature that is considered idle. Ex: 50.0 = 50 celcius
     * idlePowerWatts: Power draw that is considered idle. Ex: 100.0 = 100 watts
     *
     * Returns: 1 if device is lwrrently considered idle
     *          0 if device is NOT lwrrently considered idle
     *         <0 on error
     *
     */
    int IsIdle(double idleTemp, double idlePowerWatts);

    /*************************************************************************/
    /*
     * Wrapper functions for logging to our associated plugin or the log file
     * if we don't have an associated plugin
     *
     * failPlugin: If this is 1, we will either set the plugin object to be failed
     *             if the plugin object is non-null or we will throw an exception
     *             to be caught at a higher level. 0=just log it
     */

    void RecordWarning(const DcgmError &d, int failPlugin);
    void RecordInfo(char *logText);

    /*************************************************************************/
    /*
     * Query this GPU to see if it lwrrently has a GOM mode of LWML_GOM_LOW_DP
     * If this GPU does not support reading GOM mode, it will be considered to not
     * be in GOM mode LWML_GOM_LOW_DP.
     *
     * Returns 0 if GPU does NOT have GOM of LWML_GOM_LOW_DP
     *         1 if GPU has GOM mode of LWML_GOM_LOW_DP
     */
    int HasGomLowDp(void);

    /*************************************************************************/

private:
    unsigned int m_lwmlIndex; /* LWML index of the GPU */
    //int m_lwdaIndex;          /* Lwca index of the GPU */

    Plugin *m_plugin;         /* Plugin object to log to */
    //struct lwdaDeviceProp m_lwdaDeviceProp;
    lwmlDevice_t m_lwmlDevice;/* Lwml device handle */

    lwvs_device_state_t m_savedState; /* Saved state of the GPU when we first
                                         called Init() */
};

/*****************************************************************************/
/* Class to represent all of the devices our tests will run on */
class LwvsDeviceList
{
public:
    /*************************************************************************/
    /* Constructor.
     *
     * plugin: Plugin object to log to. NULL=not inside plugin
     *
     */
    LwvsDeviceList(Plugin *plugin);

    /*************************************************************************/
    /* Destructor */
    ~LwvsDeviceList();

    /*************************************************************************/
    /*
     * Initialize the object from a list of LWML indices
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int Init(std::vector<unsigned int>lwmlIndexes);

    /*************************************************************************/
    /*
     * Wrapper functions for logging to our associated plugin or the log file
     * if we don't have an associated plugin
     *
     * failPlugin: If this is 1, we will either set the plugin object to be failed
     *             if the plugin object is non-null or we will throw an exception
     *             to be caught at a higher level. 0=just log it
     */

    void RecordWarning(const DcgmError &d, int failPlugin);
    void RecordInfo(char *logText);

    /*************************************************************************/
    /*
     * Restore any device state that has changed since the object was instantiated
     *
     * failOnRestore: Should a fatal error be set in the plugin/thrown if any state
     *                was left changed? 1=yes. 0=no
     *
     * Returns: 0 if there was no state to restore
     *         <0 on error
     *         >0 if state was restored
     *
     */
    int RestoreState(int failOnRestore);

    /*************************************************************************/
    /*
     * Wait for all GPUs managed by this object to return to idle
     *
     * timeout: Timeout in seconds to wait before returning regardless of idle
     *          state. <0.0 = use default timeout (lwrrently 60 seconds)
     * idleTemp: Idle temperature (degrees celcius) to be below to be considered idle.
     *           <0.0 = pick a reasonable default (lwrrently 65.0)
     * idlePowerWatts: Idle power (watts) to be below to be considered idle.
     *           <0.0 = pick a reasonable default (lwrrently 100.0)
     *
     * Returns: 0 if reached idle state
     *          1 if timed out before reaching idle state
     *         <0 on other error
     *
     */
    int WaitForIdle(double timeout, double idleTemp, double idlePowerWatts);

    /*************************************************************************/
    /*
     * Query all GPUs to see if any GPUs lwrrently have a GOM mode of LWML_GOM_LOW_DP
     * If a GPU does not support reading GOM mode, it will be considered to not
     * be in GOM mode LWML_GOM_LOW_DP.
     *
     * Returns 0 if none of the GPUs have LWML_GOM_LOW_DP
     *         1 if any of the GPUs have LWML_GOM_LOW_DP
     */
    int DoAnyGpusHaveGomLowDp(void);

    /*************************************************************************/

private:
    Plugin *m_plugin; /* Plugin object to log to. Can be null */
    std::vector<LwvsDevice *>m_devices;
};

/*****************************************************************************/

#endif //LWVSDEVICELIST_H
