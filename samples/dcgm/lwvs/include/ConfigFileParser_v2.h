#ifndef _LWVS_LWVS_ConfigFileParser2_H_
#define _LWVS_LWVS_ConfigFileParser2_H_

#include "yaml-cpp/yaml.h"
#include "GpuSet.h"
#include "TestParameters.h"
#include <string>
#include <vector>
#include <fstream>

enum lwvs_fwcfg_enum
{
    LWVS_FWCFG_GLOBAL_DATAFILE = 0,
    LWVS_FWCFG_GLOBAL_DATAFILETYPE,
    LWVS_FWCFG_GLOBAL_OVERRIDEMINMAX,
    LWVS_FWCFG_GLOBAL_OVERRIDESERIAL,
    LWVS_FWCFG_GLOBAL_SCRIPTABLE,
    LWVS_FWCFG_GLOBAL_PERSISTENCE,

    LWVS_FWCFG_GPUSET_NAME,

    LWVS_FWCFG_GPU_INDEX,
    LWVS_FWCFG_GPU_BRAND,
    LWVS_FWCFG_GPU_NAME,
    LWVS_FWCFG_GPU_BUSID,
    LWVS_FWCFG_GPU_UUID,

    LWVS_FWCFG_TEST_NAME,
};

class LwvsFrameworkConfig
{
public:
    /* GLOBALS */
    std::string dataFile;          /* name of the file to output data */
    logFileType_enum dataFileType; /* type of data output */
    bool overrideMinMax;           /* allow override of the min and max whitelist values */
    bool overrideSerial;           /* force serialization of naturally parallel plugins */
    bool scriptable;               /* give a concise colon-separated output for easy parsing */
    bool requirePersistence;       /* require that persistence mode be on */

    /* GPUSET */
    std::string gpuSetIdentifier;  /* name to identify the gpuset for human readability purposes */

    /* GPUS */
    std::vector<unsigned int> index; /* comma separated list of gpu indexes */
    std::string brand;               /* brand of the GPU */
    std::string name;                /* name of the GPU */
    std::string busid;               /* busID of the GPU */
    std::string uuid;                /* UUID of the GPU */

    /* TESTNAME */
    std::string testname;            /* Name of the test/suite/class that should be exelwted */

    LwvsFrameworkConfig() : dataFile(), dataFileType(LWVS_LOGFILE_TYPE_JSON), overrideMinMax(false),
  						 	overrideSerial(false), scriptable(false), requirePersistence(true),
						   	gpuSetIdentifier(), index(), brand(), name(), busid(), uuid(), testname()
    {
    }

    LwvsFrameworkConfig(const LwvsFrameworkConfig &other) : dataFile(other.dataFile),
															dataFileType(other.dataFileType),
                                                            overrideMinMax(other.overrideMinMax),
                                                            overrideSerial(other.overrideSerial),
                                                            scriptable(other.scriptable),
                                                            requirePersistence(other.requirePersistence),
                                                            gpuSetIdentifier(other.gpuSetIdentifier),
                                                            index(other.index), brand(other.brand),
                                                            name(other.name), busid(other.busid),
                                                            uuid(other.uuid), testname(other.testname)
    {
    }
};

/* Class that represents a run, nominally set via the configuration file */

class FrameworkConfig
{
public:
    /***************************************************************/
    /* ctor is responsible for filling in the default values for the config */
    FrameworkConfig();
    ~FrameworkConfig();
    FrameworkConfig(const FrameworkConfig &other);
    FrameworkConfig &operator=(const FrameworkConfig &other);

    /***************************************************************/
    /* setter 
     * a return of true indicates that the value was set properly 
     */
    template <class T>
    bool SetFrameworkConfigValue(lwvs_fwcfg_enum field, const T& value);

    /***************************************************************/
    /* getter
     */
    template <class T>
    T GetFrameworkConfigValue(lwvs_fwcfg_enum field);

    /* TO BE DELETED */
    LwvsFrameworkConfig GetFWCFG() { return m_config; }

private:
    LwvsFrameworkConfig m_config;
};

/* Class to contain all methods related to parsing the configuration file
 * and represent those values to calling entities.
 */
class ConfigFileParser_v2
{
public:
    /***************************************************************/
    /* ctor/dtor are responsible for entering default values into that object.  
     * The default is a long run on all available GPUs using standard whitelist values
     * It is assumed that a higher layer is responsible for the memory
     * management of the FrameworkConfig object
     */
    ConfigFileParser_v2(const std::string &configFile, FrameworkConfig &fwcfg);
    ~ConfigFileParser_v2();
    
    /***************************************************************/
    /* Open the stringstream for the config file and initialize 
     * YAML to the most upper layer doc
     */
    bool Init();

    /***************************************************************/
    /* Parse the config file for globals and gpu specifications
     * if configFile is empty then return immediate success and assume
     * defaults are fine.
     * This function will throw an exception on error.
     */
    void ParseGlobalsAndGpu();

    /***************************************************************/
    /* Parse the test overrides for a given test and fill in the 
     * appropriate fields in the TestParameters object.  If the
     * configFile is empty then return immediate success and assume
     * the defaults already in the TestParameters object are fine
     * This function will throw an exception on error.
     */
    void ParseTestOverrides(std::string testName, TestParameters& tp);

    /***************************************************************/
    /* Allow the config file to be overridden 
     * This closes the lwrrently opened stream and resets everything 
     */
    void setConfigFile(std::string newConfig)
        { m_configFile = newConfig; Init(); }

    /* TO BE DELETED */
    std::vector<GpuSet *> getGpuSetVec() { return gpuSets; }
    void legacyGlobalStructHelper();

private:
    FrameworkConfig m_fwcfg;
    std::string m_configFile;
    std::ifstream m_inputstream;
    YAML::Parser m_yamlparser;
    YAML::Node m_yamltoplevelnode;
    std::vector<std::string> m_testList;

    /* TO BE DELETED */
    std::vector<GpuSet *> gpuSets;

    /* private functions to relwrsively go through the gpu and globals stanzas looking for known tokens */
    void CheckTokens_globals(const YAML::Node &node);
    void CheckTokens_gpus(const YAML::Node &node);
    void CheckTokens_testDefaults(const YAML::Node &node, std::string testName, TestParameters& tp);
    void handleGpuSetBlock(const YAML::Node &node);
    void handleGpuSetParameters(const YAML::Node &node);
    void handleGpuSetTests(const YAML::Node &node);
    void handleTestDefaults(const YAML::Node &node, TestParameters& tp, bool subTest);
};

class CFPv2Exception: public std::runtime_error 
{
public:
    CFPv2Exception(const YAML::Mark& mark_, const std::string& msg_)
        : std::runtime_error(build_what(mark_, msg_)), mark(mark_), msg(msg_) {}
    virtual ~CFPv2Exception() throw() {}

    YAML::Mark mark;
    std::string msg;
    
private:
    static const std::string build_what(const YAML::Mark& mark, const std::string& msg) 
    {   
        std::stringstream output;
        output << "Config file error at line " << mark.line+1 << ", column " << mark.column+1 << ": " << msg;
        return output.str();
    }   
};

#endif //_LWVS_LWVS_ConfigFileParser2_H_
