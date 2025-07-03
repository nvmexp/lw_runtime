/*
 * Diag.h
 *
 *  Created on: Oct 13, 2015
 *      Author: chris
 */

#ifndef DIAG_H_
#define DIAG_H_

#include "Command.h"
#include "CommandOutputController.h"
#include "yaml-cpp/yaml.h"
#include "json/json.h"

#define STOP_DIAG_ELW_VARIABLE_NAME "DCGMI_STOP_DIAG_HOSTNAME"

class Diag {
public:
    Diag();
    virtual ~Diag();
    dcgmReturn_t RunStartDiag(dcgmHandle_t mLwcmHandle);
    dcgmReturn_t RulwiewDiag();

    void setDcgmRunDiag(dcgmRunDiag_t *drd);
    void setJsonOutput(bool jsonOutput);
    std::string Sanitize(const std::string &toOutput);
    void DisplayVerboseInfo(CommandOutputController &cmdView, const std::string &name, const std::string &info);

    // Made public for unit tests
    std::string HelperGetPluginName(unsigned int index);
    void HelperJsonBuildOutput(Json::Value &output, dcgmDiagResponse_t diagResult,
            const std::vector<unsigned int> &gpuIndices);
    bool HelperJsonAddResult(dcgmDiagResponsePerGpu_v2 &gpuResult, Json::Value &testEntry, unsigned int gpuIndex,
                             unsigned int testIndex, size_t i);
    void HelperJsonAddBasicTests(Json::Value &output, int &headerIndex, dcgmDiagResponse_t &diagResult);
    dcgmReturn_t GetFailureResult(dcgmDiagResponse_t &diagResult);
    void PopulateGpuList(const dcgmDiagResponse_t &diagResult, std::vector<unsigned int> &gpuVec);
    void InitializeDiagResponse(dcgmDiagResponse_t &diagResult);

private:
    void HelperDisplayDeployment(dcgmDiagResponse_t &diagResult);
    void HelperDisplayHardware(dcgmDiagResponsePerGpu_v2 *diagResults,
                               const std::vector<unsigned int> &gpuIndices);
    void HelperDisplayIntegration(dcgmDiagResponsePerGpu_v2 *diagResults,
                                  const std::vector<unsigned int> &gpuIndices);
    void HelperDisplayPerformance(dcgmDiagResponsePerGpu_v2 *diagResults,
                                  const std::vector<unsigned int> &gpuIndices);
    void HelperDisplayDeploymentResult(CommandOutputController &cmdView, const std::string &nameTag,
                                       dcgmDiagTestResult_v2 result);
    std::string HelperDisplayDiagResult(dcgmDiagResult_t val);

    void HelperDisplayGpuResults(std::string dataName, unsigned int testIndex,
                                 dcgmDiagResponsePerGpu_v2 *diagResults,
                                 const std::vector<unsigned int> &gpuIndices);

    dcgmReturn_t HelperDisplayAsJson(dcgmDiagResponse_t &diagResult, const std::vector<unsigned int> &gpuIndices);
    void HelperJsonAddPlugin(Json::Value &category, int &pluginCount, Json::Value &testEntry);
    void HelperJsonAddCategory(Json::Value &output, int &categoryIndex, Json::Value &category, int categoryCount);
    void HelperDisplayDetails(bool forceWarnings, const std::vector<unsigned int> &gpuIndices,
                              unsigned int testIndex, CommandOutputController &cmdView,
                              dcgmDiagResponsePerGpu_v2 *diagResults);
    void HelperDisplayTrainingOutput(dcgmDiagResponse_t &diagResult);

    /*
     * Displays a complete failure message for the diag accounting for JSON or normal output
     *
     */
    void HelperDisplayFailureMessage(const std::string &errMsg, dcgmReturn_t ret);

    bool isWhitespace(char c);

    dcgmRunDiag_t mDrd;
    bool          mJsonOutput;
};

/**
 * Start Diagnostics Ilwoker
 */
class StartDiag : public Command
{
public:
    StartDiag(const std::string &hostname, const std::string &parms, const std::string &configPath, bool jsonOutput,
              dcgmRunDiag_t &drd, const std::string &pathToDcgmExelwtable);
    virtual ~StartDiag();

    int Execute();

private:
    Diag mDiagObj;

    bool validGpuListFormat(const std::string &gpuList);

    /*
     * Makes the embedded host engine listen on a port so that DCGM Diag can talk to it successfully
     */
    dcgmReturn_t StartListenerServer();
};

/**
 * Abort Diagnostics Ilwoker
 */
class AbortDiag : public Command
{
public:
    AbortDiag(std::string hostname);
    virtual ~AbortDiag();

    int Execute();
};

#endif /* DIAG_H_ */
