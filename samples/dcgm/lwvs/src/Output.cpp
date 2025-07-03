#include "Output.h"
#include "LwidiaValidationSuite.h"
#include <sstream>
#include <stdexcept>
#include "common.h"

#define MAX_LINE_LENGTH 50

/*****************************************************************************/
void Output::header(const std::string &headerString)
{
    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (!lwvsCommon.parse && !lwvsCommon.training)
        m_out << "\t" << headerString << std::endl;
}

/*****************************************************************************/
std::string Output::fill(fillType_enum type)
{
    std::string ret;
    if (lwvsCommon.parse)
    {
        switch (type)
        {
            case LWVS_FILL_PREFACE:
            case LWVS_FILL_DOT:
            case LWVS_FILL_DELIMITER2:
            default:
                ret = "";
                break;
            case LWVS_FILL_DELIMITER1:
                ret = ":";
                break;
        }
    }
    else
    {
        switch (type)
        {
            case LWVS_FILL_PREFACE:
                ret = "\t\t";
                break;
            case LWVS_FILL_DOT:
                ret = ".";
                break;
            case LWVS_FILL_DELIMITER1:
            case LWVS_FILL_DELIMITER2:
            default:
                ret = " ";
                break;
        }
    }
    return ret;
}

/*****************************************************************************/
void Output::prep(const std::string &testString)
{
    if (lwvsCommon.training)
        return; // Training mode doesn't want this output

    std::stringstream ss;

    ss << fill(LWVS_FILL_PREFACE) << testString << fill(LWVS_FILL_DELIMITER1);
    for (unsigned int i = 0; i < (unsigned int)(MAX_LINE_LENGTH - testString.length()); i++)
    {
        ss << fill(LWVS_FILL_DOT);
    }
    m_out << ss.str() << fill(LWVS_FILL_DELIMITER2);
    m_out.flush();
}

/*****************************************************************************/
void Output::Result(const lwvsPluginGpuResults_t &results, const std::vector<std::string> &warningVector,
                    const std::vector<DcgmError> &errorVector, const lwvsPluginGpuErrors_t &errorsPerGpu,
                    const std::vector<std::string> &infoVector, const lwvsPluginGpuMessages_t &infoPerGpu)
{
    if (lwvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    std::string resultString;
    std::stringstream ss;
    lwvsPluginResult_t result = Plugin::GetOverallResult(results);
    resultString = resultEnumToString(result);

    ss << resultString << std::endl;
    m_out << ss.str();
    if (resultString != "PASS")
    {
        for (std::vector<std::string>::const_iterator it = warningVector.begin(); it != warningVector.end(); ++it)
        {
            m_out << "\t\t   " << "*** " << (*it) << std::endl;
        }
        for (lwvsPluginGpuErrors_t::const_iterator gpuIt = errorsPerGpu.begin();
            gpuIt != errorsPerGpu.end(); ++gpuIt)
        {
            for (std::vector<DcgmError>::const_iterator it = gpuIt->second.begin(); it != gpuIt->second.end(); ++it)
            {
                m_out << "\t\t   " << "*** (GPU " << gpuIt->first << ") " << it->GetMessage() << std::endl;
            }
        }
    }
    else if (lwvsCommon.verbose)
    {
        for (std::vector<std::string>::const_iterator it = infoVector.begin(); it != infoVector.end(); ++it)
        {
            m_out << "\t\t   info: " << (*it) << std::endl;
        }
        for (lwvsPluginGpuMessages_t::const_iterator gpuIt = infoPerGpu.begin(); gpuIt != infoPerGpu.end(); ++gpuIt)
        {
            for (std::vector<std::string>::const_iterator it = gpuIt->second.begin(); it != gpuIt->second.end(); ++it)
            {
                m_out << "\t\t   info (GPU " << gpuIt->first << "): " << (*it) << std::endl;
            }
        }
    }
}

/*****************************************************************************/
std::string Output::resultEnumToString(lwvsPluginResult_t resultEnum)
{
    std::string result;
    switch(resultEnum)
    {
        case LWVS_RESULT_PASS:
            result = "PASS";
            break;
        case LWVS_RESULT_WARN:
            result = "WARN";
            break;
        case LWVS_RESULT_SKIP:
            result = "SKIP";
            break;
        case LWVS_RESULT_FAIL:
        default:
            result = "FAIL";
            break;
    }
    return result;
}

void Output::updatePluginProgress(unsigned int progress, bool clear)
{
    static bool display = false;
    static unsigned int previousStrLength = 0;

    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (!lwvsCommon.parse && !lwvsCommon.training)
    {

        std::stringstream ss;
        ss << progress;
        if (display || clear)
        {
            for (unsigned int j = 0; j < previousStrLength; j++)
                m_out << "\b";
        }

        if (!clear)
        {
            m_out << ss.str() << "%";
            m_out.flush();

            // set up info for next progress call.
            previousStrLength = ss.str().length() + 1;
            display = true;
        }
        else // reset
        {
            previousStrLength = 0;
            display = false;
        }
    }

    return;
}

void Output::print()
{
    // This output is only for regular stdout, not training or 'parse' mode (which is deprecated)
    if (lwvsCommon.parse || lwvsCommon.training)
        return;

    if (globalInfo.size() > 0)
    {
        m_out << std::endl << std::endl;
    }

    for (size_t i = 0; i < globalInfo.size(); i++)
    {
        if (globalInfo[i].find("***") == std::string::npos)
        {
            m_out << " *** ";
        }
        m_out << globalInfo[i] << std::endl;
    }

    if (lwvsCommon.fromDcgm == false)
    {
        m_out << DEPRECATION_WARNING << std::endl;
    }
}

std::string Output::RemoveNewlines(const std::string &str)
{
    std::string altered(str);
    size_t pos;
    // Remove newlines
    while ((pos = altered.find('\n')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    while ((pos = altered.find('\r')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    while ((pos = altered.find('\f')) != std::string::npos)
    {
        altered[pos] = ' ';
    }
    return altered;
}

void Output::addInfoStatement(const std::string &info)
{
    this->globalInfo.push_back(RemoveNewlines(info));
}

void Output::AddTrainingResult(const std::string &trainingOut)
{
    m_out << trainingOut;
}
