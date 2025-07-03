#include <sstream>
#include <vector>

#include "DcgmStringTokenize.h"
#include "DcgmDiagCommon.h"

bool is_valid_gpu_list(const std::string &gpuList)
{
    std::vector<std::string> gpuIdVec;
    tokenizeString(gpuList, ",", gpuIdVec);
    for (size_t i = 0; i < gpuIdVec.size(); i++)
    {
        for (size_t j = 0; j < gpuIdVec[i].size(); j++)
        {
            if (!isdigit(gpuIdVec[i][j]))
            {
                // found a non-digit in the comma-separated list
                return false;
            }
        }
    }
    return true;
}

dcgmReturn_t dcgm_diag_common_populate_run_diag(dcgmRunDiag_t &drd,
                                                const std::string &testNames,
                                                const std::string &parms,
                                                const std::string &configFileContents,
                                                const std::string &gpuList,
                                                bool verbose,
                                                bool statsOnFail,
                                                const std::string &debugLogFile,
                                                const std::string &statsPath,
                                                unsigned int debugLevel,
                                                const std::string &throttleMask,
                                                const std::string &pluginPath,
                                                bool training,
                                                bool forceTrain,
                                                unsigned int trainingIterations,
                                                unsigned int trainingVariance,
                                                unsigned int trainingTolerance,
                                                const std::string &goldelwaluesFile,
                                                unsigned int groupId,
                                                bool failEarly,
                                                unsigned int failCheckInterval,
                                                std::string &error)
{
    std::stringstream errbuf;
    // testNames might be a simple 1, 2, or 3 for short, medium, or long. Don't both in that case
    if (testNames.size() > 1)
    {
        std::vector<std::string> testNamesVec;
        tokenizeString(testNames, ",", testNamesVec);

        for (size_t namesIndex = 0; namesIndex < testNamesVec.size(); namesIndex++)
        {
            snprintf(drd.testNames[namesIndex], sizeof(drd.testNames[namesIndex]), "%s",
                     testNamesVec[namesIndex].c_str());
        }
        drd.validate = DCGM_POLICY_VALID_NONE;
    }
    else if (testNames == "1")
    {
        drd.validate = DCGM_POLICY_VALID_SV_SHORT;
    }
    else if (testNames == "2")
    {
        drd.validate = DCGM_POLICY_VALID_SV_MED;
    }
    else if (testNames == "3")
    {
        drd.validate = DCGM_POLICY_VALID_SV_LONG;
    }
    else if (training == true)
    {
        if (testNames.size() != 0)
        {
            errbuf << "Invalid value: -r should not be specificied with --train, but found: '" << testNames << "'.";
            error = errbuf.str();
            return DCGM_ST_BADPARAM;
        }
    }
    else
    {
        // The only valid 1 character -r options are 1, 2, or 3
        errbuf << "Invalid value '" << testNames << "' specified for -r.";
        error = errbuf.str();
        return DCGM_ST_BADPARAM;
    }

    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        tokenizeString(parms, ";", parmsVec);

        for (size_t parmsIndex = 0; parmsIndex < parmsVec.size(); parmsIndex++)
        {
            snprintf(drd.testParms[parmsIndex], sizeof(drd.testParms[parmsIndex]), "%s",
                     parmsVec[parmsIndex].c_str());
        }
    }

    if (gpuList.empty() == false)
    {
        if (is_valid_gpu_list(gpuList) == false)
        {
            errbuf << "Invalid gpu list '" << gpuList
                   << "'. The list must be a comma-separated list of numbers only.";
            error = errbuf.str();

            return DCGM_ST_BADPARAM;
        }    
        snprintf(drd.gpuList, sizeof(drd.gpuList), "%s", gpuList.c_str());
    }

    if (statsOnFail)
    {
        drd.flags |= DCGM_RUN_FLAGS_STATSONFAIL;
    }

    if (verbose)
    {
        drd.flags |= DCGM_RUN_FLAGS_VERBOSE;
    }

    if (training)
    {
        drd.flags |= DCGM_RUN_FLAGS_TRAIN;
        if (forceTrain)
        {
            drd.flags |= DCGM_RUN_FLAGS_FORCE_TRAIN;
        }
    }

    drd.debugLevel = debugLevel;

    snprintf(drd.debugLogFile, sizeof(drd.debugLogFile), "%s", debugLogFile.c_str());
    snprintf(drd.statsPath, sizeof(drd.statsPath), "%s", statsPath.c_str());

    drd.groupId = (dcgmGpuGrp_t)(long long)groupId;

    /* Parameters for newer versions of dcgmRunDiag_t */
    dcgm_diag_common_set_config_file_contents(configFileContents, drd);

    if (drd.version > dcgmRunDiag_version2) // v3 and newer
    {
        snprintf(drd.throttleMask, sizeof(drd.throttleMask), "%s", throttleMask.c_str());

        if (drd.version > dcgmRunDiag_version3) // v4 and newer
        {
            snprintf(drd.pluginPath, sizeof(drd.pluginPath), "%s", pluginPath.c_str());
            drd.trainingIterations = trainingIterations;
            drd.trainingVariance = trainingVariance;
            drd.trainingTolerance = trainingTolerance;
            snprintf(drd.goldelwaluesFile, sizeof(drd.goldelwaluesFile), "%s", goldelwaluesFile.c_str());
            if (drd.version > dcgmRunDiag_version4 && failEarly) // v5 and newer
            {
                drd.flags |= DCGM_RUN_FLAGS_FAIL_EARLY;
                drd.failCheckInterval = failCheckInterval;
            }
        }
    }

    return DCGM_ST_OK;
}

void dcgm_diag_common_set_config_file_contents(const std::string &configFileContents, dcgmRunDiag_t &drd)
{
    if (drd.version > (unsigned int)dcgmRunDiag_version1) // v2 and newer
    {
        snprintf(drd.configFileContents, sizeof(drd.configFileContents), "%s", configFileContents.c_str());
    }
}
