#ifndef _LWVS_LWVS_Output_H
#define _LWVS_LWVS_Output_H

#include <iostream>
#include <vector>
#include <sstream>
#include "common.h"

class Output
{
/***************************PUBLIC***********************************/
public:
    Output() : m_out (std::cout.rdbuf()), m_err (std::cerr.rdbuf()), globalInfo()
    {
        if (lwvsCommon.quietMode)
        {
            m_out.rdbuf (m_nullSS.rdbuf());
            m_err.rdbuf (m_nullSS.rdbuf());
        }
        m_oldCout = std::cout.rdbuf();
        m_oldCerr = std::cerr.rdbuf();
    }

    virtual ~Output() {}
    
    enum fillType_enum
    {
        LWVS_FILL_PREFACE,
        LWVS_FILL_DELIMITER1,
        LWVS_FILL_DELIMITER2,
        LWVS_FILL_DOT,
    };

    virtual void header(const std::string &headerSting);
    virtual void Result(const lwvsPluginGpuResults_t &results, const std::vector<std::string> &warningVector,
                        const std::vector<DcgmError> &errorVector, const lwvsPluginGpuErrors_t &errorsPerGpu,
                        const std::vector<std::string> &infoVector, const lwvsPluginGpuMessages_t &infoPerGpu);
    virtual void prep(const std::string &testString);
    virtual void updatePluginProgress(unsigned int progress, bool clear);
    virtual void print();
    virtual void addInfoStatement(const std::string &info);
    virtual void AddTrainingResult(const std::string &trainingOut);

/***************************PRIVATE**********************************/
private:
	std::streambuf* m_oldCout;
	std::streambuf* m_oldCerr;

    std::stringstream m_nullSS;

    //methods
    std::string fill(fillType_enum type);

protected:
    std::ostream m_out;
    std::ostream m_err;
    
    std::string resultEnumToString(lwvsPluginResult_t result);
    std::vector<std::string> globalInfo;

    std::string RemoveNewlines(const std::string &str);
};

#endif // _LWVS_LWVS_Output_H
