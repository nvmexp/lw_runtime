#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <stdexcept>

#include "common.h"
#include "logging.h"
#include "dcgm_structs.h"

LwvsCommon::LwvsCommon() : logFile(), m_statsPath("./"), logFileType(LWVS_LOGFILE_TYPE_JSON), parse(false),
               quietMode(false), serialize(false), overrideMinMax(false), verbose(false),
               requirePersistenceMode(true), configless(false), statsOnlyOnFail(false),
               errorMask(0), level(LWML_DBG_DISABLED), mainReturnCode(MAIN_RET_OK), pluginPath(),
               desiredTest(), indexString(), parms(), jsonOutput(false), fromDcgm(false),
               dcgmHostname(), training(false), forceTraining(false), throttleIgnoreMask(DCGM_INT64_BLANK), failEarly(false),
               failCheckInterval(5)

{
}

LwvsCommon::LwvsCommon(const LwvsCommon &other) : logFile(other.logFile), m_statsPath(other.m_statsPath),
                                      logFileType(other.logFileType), parse(other.parse),
                                      quietMode(other.quietMode), serialize(other.serialize),
                                      overrideMinMax(other.overrideMinMax),
                                      verbose(other.verbose),
                                      requirePersistenceMode(other.requirePersistenceMode),
                                      configless(other.configless),
                                      statsOnlyOnFail(other.statsOnlyOnFail),
                                      errorMask(other.errorMask), level(other.level),
                                      mainReturnCode(other.mainReturnCode),
                                      pluginPath(other.pluginPath), desiredTest(other.desiredTest),
                                      indexString(other.indexString), parms(other.parms),
                                      jsonOutput(other.jsonOutput), fromDcgm(other.fromDcgm),
                                      dcgmHostname(other.dcgmHostname), training(other.training),
                                      forceTraining(other.forceTraining), throttleIgnoreMask(other.throttleIgnoreMask),
                                      failEarly(other.failEarly),
                                      failCheckInterval(other.failCheckInterval)
{
}

LwvsCommon &LwvsCommon::operator=(const LwvsCommon &other)
{
    logFile = other.logFile;
    m_statsPath = other.m_statsPath;
    logFileType = other.logFileType;
    parse = other.parse;
    quietMode = other.quietMode;
    serialize = other.serialize;
    overrideMinMax = other.overrideMinMax;
    verbose = other.verbose;
    requirePersistenceMode = other.requirePersistenceMode;
    configless = other.configless;
    statsOnlyOnFail = other.statsOnlyOnFail;
    errorMask = other.errorMask;
    level = other.level;
    mainReturnCode = other.mainReturnCode;
    pluginPath = other.pluginPath;
    desiredTest = other.desiredTest;
    indexString = other.indexString;
    parms = other.parms;
    jsonOutput = other.jsonOutput;
    fromDcgm = other.fromDcgm;
    dcgmHostname = other.dcgmHostname;
    training = other.training;
    forceTraining = other.forceTraining;
    throttleIgnoreMask = other.throttleIgnoreMask;
    failEarly = other.failEarly;
    failCheckInterval = other.failCheckInterval;

    return *this;
}

void LwvsCommon::Init()
{
    logFile = "";
    m_statsPath = "./";
    logFileType = LWVS_LOGFILE_TYPE_JSON;
    parse = false;
    quietMode = false;
    serialize = false;
    overrideMinMax = false;
    verbose = false;
    requirePersistenceMode = true;
    configless = false;
    statsOnlyOnFail = false;
    errorMask = 0;
    level = LWML_DBG_DISABLED;
    mainReturnCode = MAIN_RET_OK;
    pluginPath = "";
    desiredTest.clear();
    indexString = "";
    parms.clear();
    jsonOutput = false;
    fromDcgm = false;
    dcgmHostname = "";
    training = false;
    forceTraining = false;
    throttleIgnoreMask = DCGM_INT64_BLANK;
    failEarly = false;
    failCheckInterval = 5;
}

void LwvsCommon::SetStatsPath(const std::string &statsPath)
{
    std::stringstream buf;
    
    if (statsPath.empty())
    {
        return;
    }

    if  (access(statsPath.c_str(), 0) == 0)
    {
        struct stat status;
        stat(statsPath.c_str(), &status);

        if (!(status.st_mode & S_IFDIR)) // not a dir
        {
            buf << "Error: statspath '" << statsPath << "' is not a directory.";
            throw std::runtime_error(buf.str());
        }
    }
    else
    {
        buf << "Error: cannot access statspath '" << statsPath << "': " << strerror(errno);
        throw std::runtime_error(buf.str());
    }

    m_statsPath = statsPath;
}

