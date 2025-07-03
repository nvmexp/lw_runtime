#include "TestParameters.h"
#include "float.h"
#include "common.h"
#include <sstream>
#include <cstdlib>

/*****************************************************************************/
TestParameterValue::TestParameterValue(std::string defaultValue)
{
    m_doubleValue = 0.0;
    m_stringValue = defaultValue;
    m_valueType = TP_T_STRING;
    m_doubleMilwalue = DBL_MIN;
    m_doubleMaxValue = DBL_MAX;
}

/*****************************************************************************/
TestParameterValue::TestParameterValue(double defaultValue, double milwalue, double maxValue)
{
    m_doubleValue = defaultValue;
    m_stringValue = "";
    m_valueType = TP_T_DOUBLE;
    m_doubleMilwalue = milwalue;
    m_doubleMaxValue = maxValue;
}

/*****************************************************************************/
TestParameterValue::TestParameterValue(const TestParameterValue &copyMe)
{
    m_doubleValue = copyMe.m_doubleValue;
    m_stringValue = copyMe.m_stringValue;
    m_valueType = copyMe.m_valueType;
    m_doubleMilwalue = copyMe.m_doubleMilwalue;
    m_doubleMaxValue = copyMe.m_doubleMaxValue;
}

/*****************************************************************************/
int TestParameterValue::GetValueType()
{
    return m_valueType;
}

/*****************************************************************************/
TestParameterValue::~TestParameterValue()
{
    m_stringValue = "";
}

/*****************************************************************************/
int TestParameterValue::Set(std::string value)
{
    double beforeDoubleValue = m_doubleValue;

    /* Possibly coerce the value into the other type */
    if(m_valueType == TP_T_STRING)
    {
        m_stringValue = value;
        return 0;
    }

    m_doubleValue = atof(value.c_str());
    if(m_doubleValue == 0.0 && value.c_str()[0] != '0')
    {
        m_doubleValue = beforeDoubleValue;
        return TP_ST_CANTCOERCE; /* atof failed. Must be a bad value */
    }
    else if((m_doubleValue < m_doubleMilwalue || m_doubleValue > m_doubleMaxValue) && !lwvsCommon.overrideMinMax)
    {
        m_doubleValue = beforeDoubleValue;
        return TP_ST_OUTOFRANGE;
    }

    return 0;
}

/*****************************************************************************/
int TestParameterValue::Set(double value)
{
    /* Possibly coerce the value into the other type */
    if(m_valueType == TP_T_DOUBLE)
    {
        if((value < m_doubleMilwalue || value > m_doubleMaxValue) && !lwvsCommon.overrideMinMax)
            return TP_ST_OUTOFRANGE;

        m_doubleValue = value;
        return 0;
    }

    std::stringstream ss;
    ss << value;
    m_stringValue = ss.str();
    return 0;
}

/*****************************************************************************/
double TestParameterValue::GetDouble(void)
{
    /* Possibly coerce the value into the other type */
    if(m_valueType == TP_T_DOUBLE)
    {
        return m_doubleValue;
    }

    return atof(m_stringValue.c_str());
}

/*****************************************************************************/
std::string TestParameterValue::GetString(void)
{
    /* Possibly coerce the value into the other type */
    if(m_valueType == TP_T_STRING)
    {
        return m_stringValue;
    }

    std::stringstream ss;
    ss << m_doubleValue;
    return ss.str();
}

/*****************************************************************************/
/*****************************************************************************/
TestParameters::TestParameters()
{
    m_globalParameters.clear();
    m_subTestParameters.clear();
}

/*****************************************************************************/
TestParameters::~TestParameters()
{
    std::map<std::string,TestParameterValue *>::iterator it;

    for(it = m_globalParameters.begin(); it != m_globalParameters.end(); it++)
    {
        if (it->second)
            delete(it->second);
    }
    m_globalParameters.clear();

    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;

    for(outerIt = m_subTestParameters.begin(); outerIt != m_subTestParameters.end(); outerIt++)
    {
        for(it = outerIt->second.begin(); it != outerIt->second.end(); it++)
        {
            delete(it->second);
        }
        outerIt->second.clear();
    }
    m_subTestParameters.clear();
}

/*****************************************************************************/
TestParameters::TestParameters(TestParameters &copyMe)
{
    /* do a deep copy of the source object */
    std::map<std::string,TestParameterValue *>::iterator it;

    for(it = copyMe.m_globalParameters.begin(); it != copyMe.m_globalParameters.end(); it++)
    {
        m_globalParameters[std::string(it->first)] = new TestParameterValue(*(it->second));
    }

    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;

    for(outerIt = copyMe.m_subTestParameters.begin(); outerIt != copyMe.m_subTestParameters.end(); outerIt++)
    {
        for(it = outerIt->second.begin(); it != outerIt->second.end(); it++)
        {
            m_subTestParameters[outerIt->first][it->first] = new TestParameterValue(*(it->second));
        }
    }
}

/*****************************************************************************/
int TestParameters::AddString(std::string key, std::string value)
{
    if(m_globalParameters.find(key) != m_globalParameters.end())
    {
        PRINT_WARNING("%s %s", "Tried to add parameter %s => %s, but it already exists",
                      key.c_str(), value.c_str());
        return TP_ST_ALREADYEXISTS;
    }

    m_globalParameters[key] = new TestParameterValue((std::string)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddDouble(std::string key, double value, double milwalue, double maxValue)
{
    if(m_globalParameters.find(key) != m_globalParameters.end())
    {
        PRINT_WARNING("%s %f %f %f", "Tried to add parameter %s => %f (min %f, max %f), but it already exists",
                      key.c_str(), value, milwalue, maxValue);
        return TP_ST_ALREADYEXISTS;
    }
    else if(value < milwalue || value > maxValue)
    {
        PRINT_WARNING("%s %f %f %f", "Tried to add parameter %s => %f (min %f, max %f) outside of its own range",
                      key.c_str(), value, milwalue, maxValue);
        return TP_ST_OUTOFRANGE; /* Our default value is outside our own range */
    }

    m_globalParameters[key] = new TestParameterValue((double)value, milwalue, maxValue);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddSubTestString(std::string subTest, std::string key, std::string value)
{
    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;
    std::map<std::string,TestParameterValue *>::iterator it;

    outerIt = m_subTestParameters.find(subTest);
    if(outerIt != m_subTestParameters.end())
    {
        it = outerIt->second.find(key);
        if(it != outerIt->second.end())
        {
            PRINT_WARNING("%s %s %s", "Tried to add subtest %s parameter %s => %s, "
                          "but it already exists",
                          subTest.c_str(), key.c_str(), value.c_str());
            return TP_ST_ALREADYEXISTS;
        }
    }

    m_subTestParameters[subTest][key] = new TestParameterValue((std::string)value);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::AddSubTestDouble(std::string subTest, std::string key, double value,
                                     double milwalue, double maxValue)
{
    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;
    std::map<std::string,TestParameterValue *>::iterator it;

    if(value < milwalue || value > maxValue)
    {
        return TP_ST_OUTOFRANGE; /* Our default value is outside our own range */
    }

    outerIt = m_subTestParameters.find(subTest);
    if(outerIt != m_subTestParameters.end())
    {
        it = outerIt->second.find(key);
        if(it != outerIt->second.end())
        {
            PRINT_WARNING("%s %s %f %f %f", "Tried to add subtest %s parameter %s => %f (min %f, max %f), but it already exists",
                          subTest.c_str(), key.c_str(), value, milwalue, maxValue);
            return TP_ST_ALREADYEXISTS;
        }
    }

    m_subTestParameters[subTest][key] = new TestParameterValue((double)value,
                                                               milwalue, maxValue);
    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::SetString(std::string key, std::string value, bool silent)
{
    if(m_globalParameters.find(key) == m_globalParameters.end())
    {
        if (silent == false) {
            PRINT_WARNING("%s %s", "Tried to set unknown parameter %s to %s",
                          key.c_str(), value.c_str());
        }
        return TP_ST_NOTFOUND;
    }

    return m_globalParameters[key]->Set((std::string)value);
}

/*****************************************************************************/
int TestParameters::SetDouble(std::string key, double value)
{
    if(m_globalParameters.find(key) == m_globalParameters.end())
    {
        PRINT_WARNING("%s %f", "Tried to set unknown parameter %s to %f",
                      key.c_str(), value);
        return TP_ST_NOTFOUND;
    }

    return m_globalParameters[key]->Set(value);
}

/*****************************************************************************/
int TestParameters::SetSubTestString(std::string subTest, std::string key, std::string value)
{
    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;
    std::map<std::string,TestParameterValue *>::iterator it;

    outerIt = m_subTestParameters.find(subTest);
    if(outerIt == m_subTestParameters.end())
    {
        PRINT_WARNING("%s %s %s", "Tried to set unknown subtest %s's parameter %s to %s",
                      subTest.c_str(), key.c_str(), value.c_str());
        return TP_ST_NOTFOUND;
    }
    it = outerIt->second.find(key);
    if(it == outerIt->second.end())
    {
        PRINT_WARNING("%s %s %s", "Tried to set subtest %s's unknown parameter %s to %s",
                      subTest.c_str(), key.c_str(), value.c_str());
        return TP_ST_NOTFOUND;
    }

    return m_subTestParameters[subTest][key]->Set((std::string)value);
}

/*****************************************************************************/
int TestParameters::SetSubTestDouble(std::string subTest, std::string key, double value)
{
    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerIt;
    std::map<std::string,TestParameterValue *>::iterator it;

    outerIt = m_subTestParameters.find(subTest);
    if(outerIt == m_subTestParameters.end())
    {
        PRINT_WARNING("%s %s %f", "Tried to set unknown subtest %s's parameter %s to %f",
                      subTest.c_str(), key.c_str(), value);
        return TP_ST_NOTFOUND;
    }
    it = outerIt->second.find(key);
    if(it == outerIt->second.end())
    {
        PRINT_WARNING("%s %s %f", "Tried to set subtest %s's unknown parameter %s to %f",
                      subTest.c_str(), key.c_str(), value);
        return TP_ST_NOTFOUND;
    }

    m_subTestParameters[subTest][key]->Set((double)value);
    return TP_ST_OK;
}

/*****************************************************************************/
std::string TestParameters::GetString(std::string key)
{
    return m_globalParameters[key]->GetString();
}

/*****************************************************************************/
static int bool_string_to_bool(std::string str)
{
    const char *cStr = str.c_str();
    char firstChar = *cStr;

    if(str.size() < 1)
        return 0; /* Empty string is false */

    if(firstChar == 't' || firstChar == 'T' || firstChar == '1' ||
       firstChar == 'Y' || firstChar == 'y')
        return 1;
    else
        return 0; /* Everything else is false */
}

/*****************************************************************************/
int TestParameters::GetBoolFromString(std::string key)
{
    std::string str = m_globalParameters[key]->GetString();
    return bool_string_to_bool(str);
}

/*****************************************************************************/
double TestParameters::GetDouble(std::string key)
{
    return m_globalParameters[key]->GetDouble();
}

/*****************************************************************************/
std::string TestParameters::GetSubTestString(std::string subTest, std::string key)
{
    return m_subTestParameters[subTest][key]->GetString();
}

/*****************************************************************************/
double TestParameters::GetSubTestDouble(std::string subTest, std::string key)
{
    return m_subTestParameters[subTest][key]->GetDouble();
}

/*****************************************************************************/
int TestParameters::GetBoolFromSubTestString(std::string subTest, std::string key)
{
    std::string str = m_subTestParameters[subTest][key]->GetString();
    return bool_string_to_bool(str);
}

/*****************************************************************************/
int TestParameters::OverrideFrom(TestParameters *sourceTp)
{
    std::map<std::string,TestParameterValue *>::iterator destIt, sourceIt;
    TestParameterValue *deleteTpv = 0;

    /* Global parameters */
    for(destIt = m_globalParameters.begin(); destIt != m_globalParameters.end(); destIt++)
    {
        sourceIt = sourceTp->m_globalParameters.find(destIt->first);
        if(sourceIt == sourceTp->m_globalParameters.end())
            continue;

        /* Found the same parameter. Delete ours, clone theirs, and point at the new one */
        deleteTpv = destIt->second;
        m_globalParameters[destIt->first] = new TestParameterValue(*(sourceIt->second));
        PRINT_DEBUG("%s %s %s", "Overrode parameter %s with value %s (previously %s)",
                    destIt->first.c_str(), m_globalParameters[destIt->first]->GetString().c_str(),
                    deleteTpv->GetString().c_str());
        delete(deleteTpv); //Free our old parameter
    }

    /* Subtest parameters */
    std::map<std::string,std::map<std::string,TestParameterValue *> >::iterator outerDestIt, outerSourceIt;

    for(outerDestIt = m_subTestParameters.begin(); outerDestIt != m_subTestParameters.end();
        outerDestIt++)
    {
        /* Does the source even have this subtest? */
        outerSourceIt = sourceTp->m_subTestParameters.find(outerDestIt->first);
        if(outerSourceIt == sourceTp->m_subTestParameters.end())
            continue; /* Nope */

        for(destIt = outerSourceIt->second.begin(); destIt != outerSourceIt->second.end(); destIt++)
        {
            sourceIt = sourceTp->m_globalParameters.find(destIt->first);
            if(sourceIt == sourceTp->m_globalParameters.end())
                continue;

            /* Found the same parameter. Delete ours, clone theirs, and point at the new one */
            deleteTpv = destIt->second;
            m_globalParameters[destIt->first] = new TestParameterValue(*(sourceIt->second));
            PRINT_DEBUG("%s %s %s %s", "Overrode subtest %s parameter %s with value %s (previously %s)",
                        outerDestIt->first.c_str(), destIt->first.c_str(),
                        m_globalParameters[destIt->first]->GetString().c_str(),
                        deleteTpv->GetString().c_str());
            delete(deleteTpv); //Free our old parameter
        }
    }

    return TP_ST_OK;
}

/*****************************************************************************/
int TestParameters::OverrideFromString(const std::string &name, const std::string &value)
{
    int rc = this->SetString(name, value, true);

    if (rc == TP_ST_NOTFOUND) {
        // If name has a '.' this could be a subtest parameter
        size_t dot = name.find('.');
        if (dot != std::string::npos) {
            std::string subtestName(name.substr(0, dot));
            std::string key(name.substr(dot+1));

            rc = this->SetSubTestString(subtestName, key, value);
        } else {
            PRINT_WARNING("%s %s", "Tried to set unknown parameter %s to %s",
                          name.c_str(), value.c_str());
        }
    }

    return rc;
}

/*****************************************************************************/
