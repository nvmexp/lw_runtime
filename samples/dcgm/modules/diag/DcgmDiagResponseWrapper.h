#ifndef DCGM_DIAG_RESPONSE_WRAPPER_H
#define DCGM_DIAG_RESPONSE_WRAPPER_H

#include <string>

#include "dcgm_structs.h"
#include "json/json.h"

extern const std::string blacklistName;
extern const std::string lwmlLibName;
extern const std::string lwdaMainLibName;
extern const std::string lwdaTkLibName;
extern const std::string permissionsName;
extern const std::string persistenceName;
extern const std::string elwName;
extern const std::string pageRetirementName;
extern const std::string graphicsName;
extern const std::string inforomName;

extern const std::string swTestNames[];

/*****************************************************************************/
/*
 * Class for handling the different versions of the diag response
 */
class DcgmDiagResponseWrapper
{
public:
    /*****************************************************************************/
    DcgmDiagResponseWrapper();

    /*****************************************************************************/
    void InitializeResponseStruct(unsigned int numGpus);

    /*****************************************************************************/
    void SetPerGpuResponseState(unsigned int testIndex, dcgmDiagResult_t result, unsigned int gpuIndex,
                                unsigned int rc = 0);

    /*****************************************************************************/
    void AddPerGpuMessage(unsigned int testIndex, const std::string &msg, unsigned int gpuIndex, bool warning);

    /*****************************************************************************/
    void SetGpuIndex(unsigned int gpuIndex);

    /*****************************************************************************/
    void RecordSystemError(const std::string &errorStr);
    
    /*****************************************************************************/
    void SetGpuCount(unsigned int gpuCount);

    /*****************************************************************************/
    unsigned int GetBasicTestResultIndex(const std::string &testname);

    /*****************************************************************************/
    dcgmReturn_t SetVersion3(dcgmDiagResponse_v3 *response);

    /*****************************************************************************/
    dcgmReturn_t SetVersion4(dcgmDiagResponse_v4 *response);
    
    /*****************************************************************************/
    dcgmReturn_t SetVersion5(dcgmDiagResponse_v5 *response);

    /*****************************************************************************/
    dcgmReturn_t RecordTrainingMessage(const std::string &trainingMsg);
               
    /*****************************************************************************/
    dcgmReturn_t AddErrorDetail(unsigned int gpuIndex, unsigned int testIndex, const std::string &testname,
                                dcgmDiagErrorDetail_t &ed, dcgmDiagResult_t result);
    
    /*****************************************************************************/
    bool IsValidGpuIndex(unsigned int gpuIndex);
private:
    union 
    {
        dcgmDiagResponse_v3 *v3ptr;          // A pointer to the version3 struct
        dcgmDiagResponse_v4 *v4ptr;          // A pointer to the version4 struct
        dcgmDiagResponse_v5 *v5ptr;          // A pointer to the version5 struct
    } m_response;
    unsigned int         m_version;          // records the version of our dcgmDiagResponse_t
    
    /*****************************************************************************/
    bool StateIsValid() const;
};

#endif

