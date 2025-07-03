#ifndef _LWVS_LWVS_Software_H_
#define _LWVS_LWVS_Software_H_

#include <string>
#include <vector>
#include <iostream>
#include "Plugin.h"
#include "Gpu.h"
#include "TestParameters.h"
#include "lwml.h"
#include "PluginStrings.h"

class Software : public Plugin
{
public:
    Software();
    ~Software() { delete tp; }

    void Go(TestParameters *testParameters, unsigned int)
    {
        throw std::runtime_error("Not implemented in this test.");
    }
    void Go(TestParameters *testParameters)
    {
        throw std::runtime_error("Not implemented in this test.");        
    }
    void Go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList);
    void setArgs(std::string args) { myArgs = args; }
    

private:
	enum libraryCheck_t 
    {
        CHECK_LWML,             // LWML library
        CHECK_LWDA,             // LWCA library (installed as part of driver)
        CHECK_LWDATK,           // LWCA toolkit libraries (blas, fft, etc.)
    };

    //variables
    std::string myArgs;
    TestParameters * tp;

    //methods
    bool checkPermissions();
	bool checkLibraries(libraryCheck_t libs);
	bool checkBlacklist();
    bool findLib(std::string, std::string &error);
    int  checkDriverPathBlacklist(std::string, const std::string*);
    int retrieveDeviceCount(unsigned int *count);
    int checkPersistenceMode(const std::vector<unsigned int> &gpuList);
    int checkForGraphicsProcesses(const std::vector<unsigned int> &gpuList);
    int checkForBadElwVaribles();
    int checkPageRetirement(const std::vector<unsigned int> &gpuList);
    int checkInforom(const std::vector<unsigned int> &gpuList);

    void *LoadLwmlLib();
    int LoadAndCallLwmlInit(void *lib_handle);
    int LoadAndCallLwmlShutdown(void *lib_handle);
};



#endif // _LWVS_LWVS_Software_H_
