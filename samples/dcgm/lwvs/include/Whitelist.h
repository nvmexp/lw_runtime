#ifndef _LWVS_LWVS_Whitelist_H
#define _LWVS_LWVS_Whitelist_H

#include <set>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include "TestParameters.h"
#include "common.h"
#include "Gpu.h"

class Whitelist
{
/***************************PUBLIC***********************************/
public:
    Whitelist();
    ~Whitelist();

    //getters
    bool isWhitelisted(std::string deviceId);
    void getDefaultsByDeviceId(const std::string &testName, const std::string &deviceId, TestParameters * tp);
    
    /****************************************************************/
    /*
     * Adjust the whitelist values once GPUs have been read from DCGM.
     * This must be done separate from the constructor because the
     * whitelist must be bootstrapped in order to generate the list of
     * supported GPUs
     */
    void postProcessWhitelist(std::vector<Gpu *> & gpus);

/***************************PRIVATE**********************************/
private:
    void FillMap();

    /****************************************************************/
    /*
     * Updates global (test-agnostic) configuration (e.g. throttle mask) for the given device if it is whitelisted.
     */
    void UpdateGlobalsForDeviceId(const std::string &deviceId);

    /* Per-hardware whitelist parameters database keyed by deviceId and
     * then plugin name */
    std::map<std::string, std::map<std::string, TestParameters *> >m_featureDb;

    /* Set of hardware deviceIds which require global configuration changes. */
    std::set<std::string> m_globalChanges;
};


#endif // _LWVS_LWVS_Whitelist_H
