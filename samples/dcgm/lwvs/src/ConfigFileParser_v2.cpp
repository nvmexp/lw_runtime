#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "ConfigFileParser_v2.h"
#include "ParsingUtility.h"


#define SET_FWCFG(X,Y)                                                          \
    while (1)                                                                   \
    {                                                                           \
        if (!m_fwcfg.SetFrameworkConfigValue(X,Y))                              \
        {                                                                       \
            PRINT_DEBUG("%d", "Unable to set value %d in FWCFG", X);            \
            throw std::runtime_error("Unable to set value in FrameworkConfig"); \
        }                                                                       \
        break;                                                                  \
    }

/*****************************************************************************/
FrameworkConfig::FrameworkConfig() : m_config()
{
    // set the default config
    m_config.dataFile = "stats";
    m_config.dataFileType = LWVS_LOGFILE_TYPE_JSON;
    m_config.overrideMinMax = false;
    m_config.overrideSerial = false;
    m_config.scriptable = false;
    m_config.requirePersistence = true;

    m_config.index.clear();
    m_config.brand = "";
    m_config.name = "";
    m_config.busid = "";
    m_config.uuid = "";

    m_config.testname = "Long";
}

FrameworkConfig::FrameworkConfig(const FrameworkConfig &other) : m_config(other.m_config)
{
}

FrameworkConfig &FrameworkConfig::operator =(const FrameworkConfig &other)
{
    m_config = other.m_config;
    return *this;
}

/*****************************************************************************/
FrameworkConfig::~FrameworkConfig()
{}

/*****************************************************************************/
template <class T>
T FrameworkConfig::GetFrameworkConfigValue(lwvs_fwcfg_enum field)
{
    switch (field)
    {
        case LWVS_FWCFG_GLOBAL_OVERRIDEMINMAX:
            return m_config.overrideMinMax;
            break;
        case LWVS_FWCFG_GLOBAL_OVERRIDESERIAL:
            return m_config.overrideSerial;
            break;
        case LWVS_FWCFG_GLOBAL_SCRIPTABLE:
            return m_config.scriptable;
            break;
        case LWVS_FWCFG_GLOBAL_PERSISTENCE:
            return m_config.requirePersistence;
            break;
       case LWVS_FWCFG_GLOBAL_DATAFILETYPE:
            return m_config.dataFileType;
            break;
        case LWVS_FWCFG_GPU_INDEX:
            return m_config.index;
            break;
        case LWVS_FWCFG_GLOBAL_DATAFILE:
            return m_config.dataFile;
            break;
        case LWVS_FWCFG_GPU_BRAND:
            return m_config.brand;
            break;
        case LWVS_FWCFG_GPU_NAME:
            return m_config.name;
            break;
        case LWVS_FWCFG_GPU_BUSID:
            return m_config.busid;
            break;
        case LWVS_FWCFG_GPU_UUID:
            return m_config.uuid;
            break;
        case LWVS_FWCFG_TEST_NAME:
            return m_config.testname;
            break;
        default:
            throw std::runtime_error ("Invalid field for the Framework Config object.");
            break;
    }
    return;
}

/*****************************************************************************/
template <class T>
bool FrameworkConfig::SetFrameworkConfigValue(lwvs_fwcfg_enum field, const T& value)
{
    // there is no checking here for the type of T on purpose
    switch (field)
    {
        case LWVS_FWCFG_GLOBAL_OVERRIDEMINMAX:
            m_config.overrideMinMax = value;
            break;
        case LWVS_FWCFG_GLOBAL_OVERRIDESERIAL:
            m_config.overrideSerial = value;
            break;
        case LWVS_FWCFG_GLOBAL_SCRIPTABLE:
            m_config.scriptable = value;
            break;
        case LWVS_FWCFG_GLOBAL_PERSISTENCE:
            m_config.requirePersistence = value;
            break;
       default: 
            return false;
    }
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue (lwvs_fwcfg_enum field, const logFileType_enum& value)
{
    switch (field)
    {
       case LWVS_FWCFG_GLOBAL_DATAFILETYPE:
            m_config.dataFileType = value;
            break;
       default: 
            return false;
    }
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue (lwvs_fwcfg_enum field, const std::vector<unsigned int>& value)
{
    switch (field)
    {
        case LWVS_FWCFG_GPU_INDEX:
            m_config.index = value;
            break;
       default: 
            return false;
    }
    return true;
}

/*****************************************************************************/
template <>
bool FrameworkConfig::SetFrameworkConfigValue (lwvs_fwcfg_enum field, const std::string& value)
{
    // there is no checking here for the type of T on purpose
    switch (field)
    {
        case LWVS_FWCFG_GLOBAL_DATAFILE:
            m_config.dataFile = value;
            break;
        case LWVS_FWCFG_GPU_BRAND:
            m_config.brand = value;
            break;
        case LWVS_FWCFG_GPU_NAME:
            m_config.name = value;
            break;
        case LWVS_FWCFG_GPU_BUSID:
            m_config.busid = value;
            break;
        case LWVS_FWCFG_GPU_UUID:
            m_config.uuid = value;
            break;
        case LWVS_FWCFG_TEST_NAME:
            m_config.testname = value;
            break;
        case LWVS_FWCFG_GPUSET_NAME:
            m_config.gpuSetIdentifier = value;
            break;
        default:
            return false;
    }
    return true;;
}

/*****************************************************************************/
/* ctor saves off the input parameters to local copies/references and opens
 * the config file
 */
ConfigFileParser_v2::ConfigFileParser_v2(const std::string &configFile, FrameworkConfig &fwcfg)
{
    PRINT_DEBUG ("%s", "ConfigFileParser_v2 ctor with configFile %s", configFile.c_str());

    // save the pertinent info and object pointer
    m_configFile = configFile; // initial configuration file
    m_fwcfg = fwcfg;        // initial frameworkconfig object

    // fill the lists of recognized tests into the list vector
    m_testList.push_back("targeted power");
    m_testList.push_back("targeted stress");
    m_testList.push_back("sm stress");
    m_testList.push_back("memory");
    m_testList.push_back("pcie");
    m_testList.push_back("software");

}

/*****************************************************************************/
ConfigFileParser_v2::~ConfigFileParser_v2()
{
    if (m_inputstream.is_open())
        m_inputstream.close();    
}

/*****************************************************************************/
/* Close the stream if needed and initialize an fstream to the config file
 * setting YAML at the top level document
 */
bool ConfigFileParser_v2::Init()
{
    if (m_inputstream.is_open())
        m_inputstream.close();    

    m_inputstream.open(m_configFile.c_str());
    if (!m_inputstream.good())
    {
        return false;
    } 

    m_yamlparser.Load(m_inputstream);
    m_yamlparser.GetNextDolwment(m_yamltoplevelnode);

    return true;
}


/*****************************************************************************/
/* Look for the gpuset, properties, and tests tags and ship those nodes to 
 * the appropriate handler functions
 */
void ConfigFileParser_v2::handleGpuSetBlock(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetBlock");

    const YAML::Node *pName;
    pName = node.FindValue("gpuset");
    if (pName)
    {
        if (pName->Type() == YAML::NodeType::Scalar)
        {
            std::string name;
            pName->GetScalar(name);
            SET_FWCFG(LWVS_FWCFG_GPUSET_NAME, name);
        } else
            throw std::runtime_error("gpuset tag in config file is not a single value");
    }

    pName = node.FindValue("properties");
    if(pName)
    {   
        handleGpuSetParameters(*pName);
    }   

    pName = node.FindValue("tests");
    if(pName)
    {   
        handleGpuSetTests(*pName);
    }    

    PRINT_DEBUG("", "Leaving handleGpuSetBlock");
}

/*****************************************************************************/
/* look for a name tag.  Only support one name tag for now
 */
void ConfigFileParser_v2::handleGpuSetTests(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetTests");

    const YAML::Node *pName;
    std::string tempVal;

    if (node.Type() == YAML::NodeType::Sequence)
    {
        if (node.size() > 1)
            throw std::runtime_error("Only one test name is supported in the gpu stanza at this time");
        else
            handleGpuSetTests(node[0]);
    } else if (node.Type() == YAML::NodeType::Map)
    {
        pName = node.FindValue("name");
        if (pName)
        {
            pName->GetScalar(tempVal);
            SET_FWCFG(LWVS_FWCFG_TEST_NAME, tempVal);
        }
    } else
        throw std::runtime_error("Parsing error in tests section of config file.");
    
    PRINT_DEBUG("", "Leaving handleGpuSetTests");
}

/*****************************************************************************/
/* look for the name, brand, busid, uuid, and index tags
 */
void ConfigFileParser_v2::handleGpuSetParameters(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering handleGpuSetParameters");

    const YAML::Node *pName = 0;
    std::string tempVal = "";

    if (node.Type() != YAML::NodeType::Map)
        throw std::runtime_error ("There is an error in the gpus section of the config file.");

    pName = node.FindValue("name");
    if (pName)
    {   
        pName->GetScalar(tempVal);
        SET_FWCFG(LWVS_FWCFG_GPU_NAME, tempVal);
    }   

    pName = node.FindValue("brand");
    if (pName)
    {   
        pName->GetScalar(tempVal);
        SET_FWCFG(LWVS_FWCFG_GPU_BRAND, tempVal);
    }   

    pName = node.FindValue("busid");
    if (pName)
    {   
        pName->GetScalar(tempVal);
        SET_FWCFG(LWVS_FWCFG_GPU_BUSID, tempVal);
    }   

    pName = node.FindValue("uuid");
    if (pName)
    {   
        pName->GetScalar(tempVal);
        SET_FWCFG(LWVS_FWCFG_GPU_UUID, tempVal);
    }   

    pName = node.FindValue("index");
    if (pName)
    {   
        std::vector<unsigned int> indexVector;
        // potentially a csv
        std::string tempString;
        pName->GetScalar(tempString);
        std::stringstream ss(tempString);
        int i;

        while (ss >>i)
        {   
            indexVector.push_back(i);
            if (ss.peek() == ',')
                ss.ignore();
        }   
        SET_FWCFG(LWVS_FWCFG_GPU_INDEX, indexVector);
    }   

    PRINT_DEBUG("", "Leaving handleGpuSetParameters");
}

/*****************************************************************************/
/* Go through the gpus stanza and find the first map
 */
void ConfigFileParser_v2::CheckTokens_gpus(const YAML::Node &node)
{
    PRINT_DEBUG("", "Entering CheckTokens_gpu");

    /* Dig down until we find a map.
     * This map should be the only one and contain the optional tags: gpuset, properties, and tests
     */

    if (node.Type() == YAML::NodeType::Sequence)
    {   
        if (node.size() > 1)
            throw std::runtime_error ("LWVS does not lwrrently support more than one gpuset.");
        CheckTokens_gpus(node[0]);
    } 
    else if (node.Type() == YAML::NodeType::Map)
    {   
        handleGpuSetBlock(node);
    } 
    else
        throw std::runtime_error ("Could not parse the gpus stanza of the config file.");

    PRINT_DEBUG("", "Leaving CheckTokens_gpu");
}

/*****************************************************************************/
/* go through the "globals" stanza looking for specific keywords and save them
 * to m_fwcfg
 */
void ConfigFileParser_v2::CheckTokens_globals(const YAML::Node &node)
{
    const YAML::Node *pName;

    PRINT_DEBUG("", "Entering CheckTokens_global");

    if (node.Type() == YAML::NodeType::Map)
    {   
        for (YAML::Iterator it=node.begin(); it != node.end(); ++it)
        {   
            std::string key, value, lowerValue;
            it.first() >> key;
            it.second() >> value;

            PRINT_DEBUG("%s %s", "CheckTokens_global key %s, value %s",
                        key.c_str(), value.c_str());

            /* Get a lowercase version of value for case-insensitive operations */
            lowerValue = value;
            std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);

            if (key == "logfile")
            {
                SET_FWCFG(LWVS_FWCFG_GLOBAL_DATAFILE, value);
            }
            if (key == "logfile_type")
            {   
                if(lowerValue == "json")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_DATAFILETYPE, LWVS_LOGFILE_TYPE_JSON);
                }
                else if(lowerValue == "text")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_DATAFILETYPE, LWVS_LOGFILE_TYPE_TEXT);
                }
                else if(lowerValue == "binary")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_DATAFILETYPE, LWVS_LOGFILE_TYPE_BINARY);
                }
                else
                {   
                    stringstream ss; 
                    ss << "Unknown logfile_type \"" << value << "\". Allowed: json, text, or binary";
                    throw std::runtime_error(ss.str());
                }   
            }
            if (key == "overrideMinMax")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_OVERRIDEMINMAX, true);
                }
            }
            if (key == "scriptable")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_SCRIPTABLE, true);
                }
            }
            if (key == "serial_override")
            {
                // default is false
                if (lowerValue == "yes" || lowerValue == "true")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_OVERRIDESERIAL, true);
                }
            }
            if (key == "require_persistence_mode")
            {
                // default is true 
                if (lowerValue == "no" || lowerValue == "false")
                {
                    SET_FWCFG(LWVS_FWCFG_GLOBAL_PERSISTENCE, false);
                }
            }
            if (key == "throttle-mask" && lowerValue.size() > 0)
            {
                /* Note: The mask is directly set in lwvsCommon for colwenience as otherwise we need to add a field 
                to LwvsFrameworkConfig, and update the legacyGlobalStructHelper to copy from LwvsFramworkConfig to 
                lwvsCommon */
                lwvsCommon.throttleIgnoreMask = GetThrottleIgnoreReasonMaskFromString(lowerValue);
            }

        }
    } else
        throw std::runtime_error ("Unable to parse the globals section of the config file.");

    PRINT_DEBUG("", "Leaving CheckTokens_global");
}

/*****************************************************************************/
void ConfigFileParser_v2::ParseGlobalsAndGpu()
{
    // because we now have a sense of what is "default" then neither of 
    // these not being found is an error

    if (const YAML::Node *pName = m_yamltoplevelnode.FindValue("globals"))
        CheckTokens_globals(*pName);
    if (const YAML::Node *pName = m_yamltoplevelnode.FindValue("gpus"))
        CheckTokens_gpus(*pName);

    /* TO BE DELETED */
    legacyGlobalStructHelper();
}

/*****************************************************************************/
/* We are looking for the test name (which can be an individual test, suite, or class
 * and then looking for the "custom" tag.  Wherever that node is, if it exists, drill
 * down from there.
 */
void ConfigFileParser_v2::ParseTestOverrides(std::string testName, TestParameters& tp)
{
    const YAML::Node *pName;
    
    // first look for the test name
    pName = m_yamltoplevelnode.FindValue(testName);
    if (pName) // found something at the top level, leave it to the helper to dig down
       handleTestDefaults(*pName, tp, false);

    /* getting here can mean one of several things
     * 1) the test name and the override section label differ in case
     * 2) it is a legacy config with a "custom" or suite tag
     */

    std::transform(testName.begin(), testName.end(), testName.begin(), ::tolower);
 
    for (YAML::Iterator it=m_yamltoplevelnode.begin(); it != m_yamltoplevelnode.end(); it++)
    {
        std::string key, value;
        it.first() >> key;
        std::transform (key.begin(), key.end(), key.begin(), ::tolower);
        if (key == testName || key == "custom" ||
            key == "long"   || key == "medium" ||
            key == "quick")                        //start the drill down if we find a known legacy tag
            CheckTokens_testDefaults(it.second(), testName, tp);
    }
}

/*****************************************************************************/
/* Initial search function for the test name... for newer config files this
 * will fall straight to handleTestDefaults as the node will be a single
 * entry map with the key == testName.  For legacy configs, we have to 
 * drill past "custom" a bit.
 */
void ConfigFileParser_v2::CheckTokens_testDefaults(const YAML::Node &node, std::string testName, TestParameters& tp)
{
    // no care for anything but maps and maps of maps, ignore everything else
    if (node.Type() == YAML::NodeType::Sequence)
    {
        for (unsigned int i = 0; i < node.size(); i++)
            CheckTokens_testDefaults(node[i], testName, tp);
    } else if (node.Type() == YAML::NodeType::Map)
    {
        for (YAML::Iterator it = node.begin(); it != node.end(); it++)
        {
            std::string key;
            it.first() >> key;
            std::transform (key.begin(), key.end(), key.begin(), ::tolower);
            if (key == testName)
            {
                try {
                    handleTestDefaults(it.second(), tp, false);
                } catch (std::exception &e)
                {
                    std::stringstream ss; 
                    ss << "Test " << key << " parsing failed with: \n\t" << e.what();
                    PRINT_ERROR("%s", "%s", ss.str().c_str());
                    throw std::runtime_error(ss.str());
                }   
            }
            if (it.second().Type() == YAML::NodeType::Map)
                CheckTokens_testDefaults(it.second(), testName, tp);
        }
    } else
    {
        std::stringstream ss;
        ss << "There is an error in the \"" << testName << "\" section of the config file.";
        throw std::runtime_error(ss.str());
    }
}

/*****************************************************************************/
/* handle actually putting the specified parameters in to the TestParms obj
 */
void ConfigFileParser_v2::handleTestDefaults(const YAML::Node &node, TestParameters& tp, bool subTest)
{   
    PRINT_DEBUG("%d", "Entering handleTestDefaults subTest=%d", (int)subTest);
    
    const YAML::Node *pName;
    unsigned int result;
    static std::string subTestName;
    
    if (node.Type() == YAML::NodeType::Map)
    {   
        for (YAML::Iterator it=node.begin(); it != node.end(); ++it)
        {   
            std::string key, value;
            it.first() >> key;
            
            if (it.second().Type() == YAML::NodeType::Map)
            {   
                if (key == "subtests")
                {   
                    handleTestDefaults(it.second(), tp, true);
                } 
                else
                {   
                    if (subTest) 
                        subTestName = key;
                    handleTestDefaults(it.second(), tp, subTest);
                }
            }
            else if(it.second().Type() == YAML::NodeType::Scalar)
            {   
                it.second() >> value;
                
                if (subTest) 
                    result = tp.SetSubTestString(subTestName, key, value);
                else
                    result = tp.SetString(key,value);
                
                if (result)
                {   
                    std::stringstream ss;
                    switch(result)
                    {   
                        case TP_ST_BADPARAM:
                            ss << "The parameter given for \"" << key << "\" caused an internal error .";
                            break;
                        case TP_ST_NOTFOUND:
                            ss << "The key \"" << key << "\" was not found.";
                            break;
                        case TP_ST_ALREADYEXISTS:
                            // should never happen since we are using set not add
                            ss << "The key \"" << key << "\" was added but already exists.";
                            break;
                        case TP_ST_CANTCOERCE:
                            ss << "The parameter given for \"" << key << "\" cannot be coerced to the type needed.";
                            break;
                        case TP_ST_OUTOFRANGE:
                            ss << "The parameter given for \"" << key << "\" is out of the reasonable range for that key.";
                            break;
                        default:
                            ss << "Received an unknown value from the test parameter system.";
                            break;
                    }
                    throw std::runtime_error(ss.str());
                }
            }
            else
            {   
                /* We would be here for a Sequence or Null (whatever Null means) */
                std::stringstream ss;
                ss << "Error in parameters section for  " << key;
                throw CFPv2Exception(node.GetMark(), ss.str());
            }
        }
    }
    else
    {   
        /* We would be here for a Sequence or Null (whatever Null means) */
        std::stringstream ss;
        ss << "error in \"key: value\" pairs";
        throw CFPv2Exception(node.GetMark(), ss.str());
    }
}

/*****************************************************************************/
/* THE BELOW FUNCTIONS ARE ONLY FOR COMPATITBILITY UNTIL HIGHER LAYERS
 * ARE REWRITTEN!
 */
void ConfigFileParser_v2::legacyGlobalStructHelper()
{
    /* gpu stuff */
    GpuSet * gpuSet = new GpuSet();
    LwvsFrameworkConfig fwcfg = m_fwcfg.GetFWCFG();
    
    gpuSet->name = fwcfg.gpuSetIdentifier;
    if (fwcfg.brand.length() || fwcfg.name.length() ||
        fwcfg.busid.length() || fwcfg.uuid.length() ||
        fwcfg.index.size())
        gpuSet->properties.present = true;

    gpuSet->properties.brand = fwcfg.brand;
    gpuSet->properties.name  = fwcfg.name;
    gpuSet->properties.busid = fwcfg.busid;
    gpuSet->properties.uuid  = fwcfg.uuid;
    if (lwvsCommon.indexString.size() > 0)
    {
        std::vector<unsigned int> indexVector;
        // potentially a csv
        std::stringstream ss(lwvsCommon.indexString);
        int i;

        while (ss >>i)
        {   
            indexVector.push_back(i);
            if (ss.peek() == ',')
                ss.ignore();
        }   
        gpuSet->properties.index = indexVector;
        gpuSet->properties.present = true; // so that things are parsed further down
    }
    else
    {
        gpuSet->properties.index = fwcfg.index;
    }
    
    // Ensure that GPU ID vector does not contain duplicates
    if (gpuSet->properties.index.size() > 1)
    {
        std::set<unsigned int> ids; // TODO: Change this to unordered_set once the switch to C++11 is complete
        std:pair<std::set<unsigned int>::iterator, bool> isValueInserted;
        for (size_t i = 0; i < gpuSet->properties.index.size(); i++)
        {
            isValueInserted = ids.insert(gpuSet->properties.index[i]);
            if (!isValueInserted.second)
            {
                throw std::runtime_error("The given GPU ID list contains duplicate IDs. "
                                         "Please remove duplicate entries and verify that the list is correct.");
            }
        }
    }

    if (lwvsCommon.desiredTest.size() > 0) {
        for (std::set<std::string>::iterator it = lwvsCommon.desiredTest.begin();
             it != lwvsCommon.desiredTest.end();
             it++) {
            std::map<std::string, std::string> tempMap;
            tempMap["name"] = *it;
            gpuSet->testsRequested.push_back(tempMap);
        }
    }
    else {
        std::map<std::string, std::string> tempMap;
        tempMap["name"] = fwcfg.testname;
        gpuSet->testsRequested.push_back(tempMap);
    }

    gpuSets.push_back(gpuSet);

    /* globals */
    lwvsCommon.logFile = fwcfg.dataFile;
    lwvsCommon.logFileType = fwcfg.dataFileType;
    lwvsCommon.overrideMinMax = fwcfg.overrideMinMax;
    lwvsCommon.serialize = fwcfg.overrideSerial;
    if (lwvsCommon.parse == false) // if it was turned on in the command line, don't overwrite it
        lwvsCommon.parse = fwcfg.scriptable;
    lwvsCommon.requirePersistenceMode = fwcfg.requirePersistence;
}
