#pragma once

/*
Common helper functions and classes relating to DCGM GPU Diagnostics
*/

#include <string>
#include "dcgm_structs.h"

/*****************************************************************************/
dcgmReturn_t dcgm_diag_common_populate_run_diag(dcgmRunDiag_t &drd,
                                                const std::string &testNames, 
                                                const std::string &parms,
                                                const std::string &configFileContents, 
                                                const std::string &gpuList,
                                                bool               verbose, 
                                                bool               statsOnFail, 
                                                const std::string &debugLogFile,
                                                const std::string &statsPath, 
                                                unsigned int debugLevel, 
                                                const std::string &throttleMask,
                                                const std::string &pluginPath,
                                                bool              training,
                                                bool              forceTrain,
                                                unsigned int       trainingIterations,
                                                unsigned int       trainingVariance,
                                                unsigned int       trainingTolerance,
                                                const std::string &goldelwaluesFile,
                                                unsigned int       groupId,
                                                bool               failEarly,
                                                unsigned int       failCheckInterval,
                                                std::string       &error);

/*****************************************************************************/
void dcgm_diag_common_set_config_file_contents(const std::string &configFileContents, dcgmRunDiag_t &drd);

/*****************************************************************************/
