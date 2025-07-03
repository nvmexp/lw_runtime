
#include "JsonOutput.h"
#include "LwidiaValidationSuite.h"
#include "common.h"

/* This class fills in a json object in the format:
 * {
 *   "DCGM GPU Diagnostic" : {
 *     "test_categories" : [
 *       {
 *         "category" : "<header>",    # One of Deployment|Hardware|Integration|Performance|Custom
 *         "tests" : [
 *           {
 *             "name" : <name>,
 *             "results" : [              # There is one results entry per GPU for all tests except Software/Deployment. Software test has one results entry which represents all GPUs.
 *               {
 *                 "gpu_ids" : <gpu_ids>, # GPU ID (as string) (name is "gpu_ids" for backwards compatibility). For deployment test, this is a CSV string of GPU ids
 *                 "status : "<status>",  # One of PASS|FAIL|WARN|SKIPPED
 *                 "warnings" : [         # Optional, depends on test output and result
 *                   "<warning_text>", ...
 *                 ],
 *                 "info" : [             # Optional, depends on test output and result
 *                    "<info_text>", ...
 *                 ]
 *               }, ...
 *             ]
 *           }, ...
 *         ]
 *       }, ...  
 *     ],
 *     "version" : "<version_str>" # 1.7
 *   }
 * }  
 */

// Forward declarations
void AddStringVectorToJson(Json::Value &value, const std::vector<std::string> &strings, const char* prefix="");

void JsonOutput::header(const std::string &headerString)
{
    if (lwvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

	if (testIndex != 0)
	{
		headerIndex++;
		testIndex = 0;
        gpuId = -1;
	} 
	else
	{
    	jv[LWVS_VERSION_STR] = LWVS_VERSION;
	}
	jv[LWVS_HEADERS][headerIndex][LWVS_HEADER] = headerString;
}

bool isSoftwareTest(const std::string &testName)
{
    if (testName == "Blacklist" || 
        testName == "LWML Library" ||
        testName == "LWCA Main Library" ||
        testName == "LWCA Toolkit Libraries" ||
        testName == "Permissions and OS-related Blocks" ||
        testName == "Persistence Mode" ||
        testName == "Elwironmental Variables" ||
        testName == "Page Retirement" ||
        testName == "Graphics Processes" ||
        testName == "Inforom")
    {
        return true;
    }
    return false;
}

void JsonOutput::prep(const std::string &testString)
{
    if (lwvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    softwareTest = isSoftwareTest(testString);
    std::size_t pos = testString.find(" GPU");
    if (pos == std::string::npos)
    {
        if (this->gpuId != -1)
        {
            testIndex++; // Move past the individually reported tests
            this->gpuId = -1;
        }

        jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_TEST_NAME] = testString;
    }
    else
    {
        int nextGpuId;

        // testString is in the format "Test Name GPU<index>", so we need to make the gpu results a member of this 
        // test.
        std::string testName(testString.substr(0, pos)); // Capture just the name
        nextGpuId = strtol(testString.substr(pos + 4).c_str(), NULL, 10); // Capture just the <index>

        if (nextGpuId <= this->gpuId)
            testIndex++;
        this->gpuId = nextGpuId;

        jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_TEST_NAME] = testName;
    }
}

void AddStringVectorToJson(Json::Value &errorArray, const std::vector<std::string> &strings,
                           const char* prefix)
{
    for (size_t i = 0; i < strings.size(); i++)
    {
        errorArray.append(prefix + strings[i]);
    }
}

void AddErrorVectorToJson(Json::Value &errorArray, const std::vector<DcgmError> &errors, const char *prefix="")
{
    for (size_t i = 0; i < errors.size(); i++)
    {
        Json::Value entry;
        entry[LWVS_WARNING] = prefix + errors[i].GetMessage();
        entry[LWVS_ERROR_ID] = errors[i].GetCode();
    
        errorArray.append(entry);
    }
}

void JsonOutput::Result(const lwvsPluginGpuResults_t &results, const std::vector<std::string> &warningVector,
                        const std::vector<DcgmError> &errorVector, const lwvsPluginGpuErrors_t &errorsPerGpu,
                        const std::vector<std::string> &infoVector, const lwvsPluginGpuMessages_t &infoPerGpu)
{
    if (lwvsCommon.training)
    {
        return; // Training mode doesn't want this output
    }

    char        buf[26];
    std::string resultStr;

    // Empty results mean the test was not run / skipped
    if (results.empty())
    {
        resultStr = resultEnumToString(LWVS_RESULT_SKIP);
        if (this->gpuId == -1)
        {
            for (size_t i = 0; i < m_gpuIndices.size(); i++)
            {
                Json::Value resultField;
                resultField[LWVS_STATUS] = resultStr;
                snprintf(buf, sizeof(buf), "%d", m_gpuIndices[i]);
                resultField[LWVS_GPU_IDS] = buf;
                jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_RESULTS][m_gpuIndices[i]] = resultField;
            }
            testIndex++;
        }
        else
        {
            // The test reports individually for each gpu
            Json::Value resultField;
            resultField[LWVS_STATUS] = resultStr;
            snprintf(buf, sizeof(buf), "%d", this->gpuId);
            resultField[LWVS_GPU_IDS] = buf;
            jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_RESULTS][this->gpuId] = resultField;
        }
        return;
    }

    // Software tests are independent of GPUs and have the same results for all GPUs
    if (softwareTest)
    {
        Json::Value resultField;
        resultStr = resultEnumToString(results.begin()->second);
        
        resultField[LWVS_STATUS] = resultStr;
        resultField[LWVS_GPU_IDS] = gpuList;

        if (!errorVector.empty())
        {
            AddErrorVectorToJson(resultField[LWVS_WARNINGS], errorVector);
        }

        if (!warningVector.empty())
        {
            AddStringVectorToJson(resultField[LWVS_WARNINGS], warningVector);
        }

        if (!infoVector.empty())
        {
            Json::Value info;
            AddStringVectorToJson(info, infoVector);
            resultField[LWVS_INFO] = info;
        }

        jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_RESULTS][0] = resultField;
        testIndex++;
        return;
    }

    // Iterate over results for all GPUs
    for (lwvsPluginGpuResults_t::const_iterator gpuResultsIt = results.begin(); 
        gpuResultsIt != results.end(); ++gpuResultsIt)
    {
        Json::Value     resultField;
        unsigned int    gpuId = gpuResultsIt->first;

        resultStr = resultEnumToString(gpuResultsIt->second);
        resultField[LWVS_STATUS] = resultStr;

        // GPU %u: Prefix for general warnings/info messages
        snprintf(buf, sizeof(buf), "GPU %u: ", gpuId);
        
        // Report warnings
        if (!errorsPerGpu.find(gpuId)->second.empty() || !errorVector.empty())
        {
            Json::Value warnings;
            // Report gpu specific warnings
            AddErrorVectorToJson(warnings, errorsPerGpu.find(gpuId)->second);
            // Treat general warnings as GPU specific warnings
            AddErrorVectorToJson(warnings, errorVector, buf);
            resultField[LWVS_WARNINGS] = warnings;
        }
        
        // Report info
        if (!infoPerGpu.find(gpuId)->second.empty() || !infoVector.empty())
        {
            Json::Value info;
            // Report gpu specific info
            AddStringVectorToJson(info, infoPerGpu.find(gpuId)->second);
            // Treat general info messages as GPU specific messages
            AddStringVectorToJson(info, infoVector, buf);
            resultField[LWVS_INFO] = info;
        }

        /* The results map (if non-empty) is guaranteed to have the appropriate GPU id even for tests that 
         * report on a per GPU basis.
         */
        snprintf(buf, sizeof(buf), "%d", gpuId);
        resultField[LWVS_GPU_IDS] = buf;
        jv[LWVS_HEADERS][headerIndex][LWVS_TESTS][testIndex][LWVS_RESULTS][gpuId] = resultField;
    }

    // If this is not a test that reports for each gpu individually, increment the test index
    if (this->gpuId == -1)
    {
        testIndex++;
    }
}

void JsonOutput::updatePluginProgress(unsigned int progress, bool clear)
{
	// NO-OP for Json Output
}

void JsonOutput::print()
{
	Json::Value complete;
	complete[LWVS_NAME] = jv;
    if (lwvsCommon.fromDcgm == false)
    {
        complete[LWVS_GLOBAL_WARN] = DEPRECATION_WARNING;
    }
	m_out << complete.toStyledString();
	m_out.flush();
}

void JsonOutput::addInfoStatement(const std::string &info)
{
    if (jv[LWVS_INFO].empty() == true)
    {
        Json::Value infoArray;
        infoArray[globalInfoCount] = RemoveNewlines(info);
        jv[LWVS_INFO] = infoArray;
    }
    else
    {
        jv[LWVS_INFO][globalInfoCount] = RemoveNewlines(info);
    }

    globalInfoCount++;
}

void JsonOutput::AddTrainingResult(const std::string &trainingOut)
{
    if (jv[LWVS_VERSION_STR].empty())
    {
    	jv[LWVS_VERSION_STR] = LWVS_VERSION;
    }

    jv[LWVS_TRAINING_MSG] = trainingOut;
}
