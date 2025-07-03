#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <string>
#include <set>
#include <map>
#include <sysexits.h>
#include <vector>

#include "DcgmError.h"

extern "C"
{
    #include "logging.h"
}

#define DEPRECATION_WARNING "LWVS has been deprecated. Please use dcgmi diag to ilwoke these tests."

#define LWML_CHECK(x,y) do {                 \
    lwmlReturn_t lwmlResult = (x);           \
    if (LWML_SUCCESS != lwmlResult)          \
    {                                        \
        throw std::runtime_error(y);         \
    }                                        \
 } while (0)

/* Main return codes */
#define MAIN_RET_OK     0
#define MAIN_RET_ERROR  1 /* Return a single code for now. In the future, we could use the standard sysexits.h codes,
                             but those are pretty ambiguous as well */

/* Has the user requested a stop? 1=yes. 0=no. Defined in main.cpp */
extern int main_should_stop;

enum suiteNames_enum
{
    LWVS_SUITE_QUICK,
    LWVS_SUITE_MEDIUM,
    LWVS_SUITE_LONG,
    LWVS_SUITE_LWSTOM,
};


/* Logfile output types */
enum logFileType_enum
{
    LWVS_LOGFILE_TYPE_JSON,  /* JSON data without line breaks */
    LWVS_LOGFILE_TYPE_TEXT,  /* Indented plain text */
    LWVS_LOGFILE_TYPE_BINARY /* Binary log format */
    /* Note if you add values here, you must change the ranges where this
     * is used in tp->AddDouble(). Lwrrently LwidiaValidationSuite.cpp */
};

/* Plugin test result states */
typedef enum lwvsPluginResult_enum
{
    LWVS_RESULT_PASS,
    LWVS_RESULT_WARN,
    LWVS_RESULT_FAIL,
    LWVS_RESULT_SKIP
} lwvsPluginResult_t;

// lwvsPluginGpuResults: map GPU IDs to the LWVS Plugin result for the GPU (i.e. Pass | Fail | Warn | Skip)
typedef std::map<unsigned int, lwvsPluginResult_t> lwvsPluginGpuResults_t;

// lwvsPluginGpuMessages: map GPU IDs to vector of (string) messages for that GPU
typedef std::map<unsigned int, std::vector<std::string> > lwvsPluginGpuMessages_t;
typedef std::map<unsigned int, std::vector<DcgmError> > lwvsPluginGpuErrors_t;

/* Logging-related elwironmental variables */
#define LWVS_ELW_DBG_LVL     "__LWVS_DBG_LVL"
#define LWVS_ELW_DBG_APPEND  "__LWVS_DBG_APPEND"
#define LWVS_ELW_DBG_FILE    "__LWVS_DBG_FILE"

/* Internal function return codes */
typedef enum lwvsReturn_enum
{
    LWVS_ST_SUCCESS = 0,
    LWVS_ST_BADPARAM = -1,       //A bad parameter was passed to a function
    LWVS_ST_GENERIC_ERROR = -2,  //A generic, unspecified error
    LWVS_ST_REQUIRES_ROOT = -3,  //This function or one of its children requires root to run
} lwvsReturn_t;

class LwvsCommon
{
public:
    LwvsCommon();
    LwvsCommon(const LwvsCommon &other);
    LwvsCommon &operator=(const LwvsCommon &other);
    void Init();
    void SetStatsPath(const std::string &statsPath);

    // structure for global variables
    std::string logFile;           /* file prefix for statistics */
    std::string m_statsPath;       /* Path where statistics files should be saved */
    logFileType_enum logFileType;  /* format for statistics to log */
    bool parse;                    /* output parseable format */
    bool quietMode;                /* quiet mode prints no output to the console, all results
                                    * are available via logs or return code only */
    bool serialize;                /* serialize tests that would normally be parallel */
    bool overrideMinMax;           /* override parm min/max values */
    bool verbose;                  /* enable verbose metric reporting */
    bool requirePersistenceMode;   /* require persistence mode to be enabled (default true) */
    bool configless;               /* enable configless operation */
    bool statsOnlyOnFail;          /* enable output of statistics files only on an failure */
    unsigned long long errorMask;  /* error mask for inforom */
    lwmlDebugingLevel level;       /* debugging level */
    int mainReturnCode;            /* MAIN_RET_? #define of the error to return or MAIN_RET_OK if no errors have oclwred */
    std::string pluginPath;        /* Path given in command line for plugins */
    std::set<std::string> desiredTest; /* Specific test(s) asked for on the command line */
    std::string indexString;       /* A potentially comma-separated list of GPUs to run LWVS on */
    std::map<std::string, std::map<std::string, std::string> > parms; /* test parameters to set from the command line */
    bool jsonOutput;               /* Produce json output as dolwmented below */
    bool fromDcgm;                 // Note that this run was initiated by DCGM to avoid the deprecation warning
    std::string dcgmHostname;      /* Host name where DCGM is running */
    bool training;                 // Run LWVS in training mode to generate golden values for this configuration
    bool forceTraining;            // Generate golden values despite warnings.
    uint64_t throttleIgnoreMask;   // Mask of throttling reasons to ignore.
    unsigned int trainingIterations; // Number of iterations of each test to perform when training.
    double trainingVariancePcnt;   // Variance allowed as a percentage of the mean in training mode.
    double trainingTolerancePcnt;  // Percentage of tolerance towards meeting the golden value in training mode.
    std::string goldelwaluesFile;  // Filename where golden values should be saved
    bool failEarly;                // enable failure checks throughout test rather than at the end so we stop test sooner
    uint64_t failCheckInterval;    /* how often failure checks should occur when running tests (in seconds). Only
                                       applies if failEarly is enabled. */
};

extern LwvsCommon lwvsCommon;

/* When jsonOutput is set to true, LWVS writes output in the format:
 * {
 *   "DCGM GPU Diagnostic" : {
 *     "test_categories" : [
 *       {
 *         "category" : "<header>",    # One of Deployment|Hardware|Integration|Performance|Custom
 *         "tests" : [
 *           {
 *             "name" : <name>,
 *             "results" : [
 *               {
 *                 "gpu_ids" : <gpu_ids>, # GPU ID - field name is left as "gpu_ids" to maintain backwards compatibility
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

#endif // COMMON_H
