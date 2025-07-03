#ifndef _LWVS_LWVS_JsonOutput_H
#define _LWVS_LWVS_JsonOutput_H

#include <iostream>
#include <vector>
#include <sstream>
#include "common.h"
#include "json/json.h"
#include "Output.h"
#include "LwvsJsonStrings.h"

class JsonOutput : public Output
{
/***************************PUBLIC***********************************/
public:
    JsonOutput(const std::vector<unsigned int> &gpuIndices) : jv(), testIndex(0), headerIndex(0), gpuId(-1),
                                                              globalInfoCount(0), gpuList(), m_gpuIndices(gpuIndices),
                                                              softwareTest(false)
    {
        char buf[50];
        for (size_t i = 0; i < gpuIndices.size(); i++)
        {
            if (i != 0)
            {
                snprintf(buf, sizeof(buf), ",%u", gpuIndices[i]);
            }
            else
            {
                snprintf(buf, sizeof(buf), "%u", gpuIndices[i]);
            }

            gpuList += buf;
        }
    }
    ~JsonOutput() {}


    void header(const std::string &headerSting);
    void Result(const lwvsPluginGpuResults_t &results, const std::vector<std::string> &warningVector,
                const std::vector<DcgmError> &errorVector, const lwvsPluginGpuErrors_t &errorsPerGpu,
                const std::vector<std::string> &infoVector, const lwvsPluginGpuMessages_t &infoPerGpu);
    void prep(const std::string &testString);
    void updatePluginProgress(unsigned int progress, bool clear);
	void print();
    void addInfoStatement(const std::string &info);
    void AddTrainingResult(const std::string &trainingOut);

/***************************PRIVATE**********************************/
private:
	Json::Value  jv;
	unsigned int testIndex;
	unsigned int headerIndex;
    int          gpuId;
    int          globalInfoCount;
    std::string  gpuList;
    std::vector<unsigned int> m_gpuIndices;
    bool         softwareTest;
};

#endif // _LWVS_LWVS_JsonOutput_H
