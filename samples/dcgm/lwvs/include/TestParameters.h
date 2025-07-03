#ifndef TESTPARAMETERS_H
#define TESTPARAMETERS_H

#include <string>
#include <map>

/* Parameter value types */
#define TP_T_STRING 0
#define TP_T_DOUBLE 1

/*****************************************************************************/
#define TP_ST_OK            0
#define TP_ST_BADPARAM      1 /* Bad parameter to function */
#define TP_ST_NOTFOUND      2 /* The requested TestParameter or sub test was
                                 not found */
#define TP_ST_ALREADYEXISTS 3 /* Tried to add TestParameter more than once */
#define TP_ST_CANTCOERCE    4 /* Was unable to coerce a value from one type
                                 to another (double <=> string) */
#define TP_ST_OUTOFRANGE    5 /* Tried to set a value outside of the value's
                                 allowed range */

/*****************************************************************************/
class TestParameterValue
{
public:
    TestParameterValue(std::string defaultValue);
    TestParameterValue(double defaultValue, double milwalue, double maxValue);
    TestParameterValue(const TestParameterValue& copyMe);
    ~TestParameterValue();

    int GetValueType();

    /*************************************************************************/
    /* Setters. Use this from the configuration file reader. These return a
     * TP_ST_? #define on error (!0)
     */
    int Set(std::string value);
    int Set(double value);

    /*************************************************************************/
    /* Getters. Note that the object also supports direct colwersion to double
     * and std::string
     **/
    double GetDouble();
    std::string GetString();

    /*************************************************************************/

private:
    int m_valueType; /* TP_T_? #define of the value type */

    /* Actual parameter value */
    std::string m_stringValue;
    double m_doubleValue;

    /* Minimum and maximum allowed values for doubles */
    double m_doubleMilwalue;
    double m_doubleMaxValue;
};

/*****************************************************************************/
/* Class for holding all of the parameters that will be passed to a test */
class TestParameters
{
public:
    TestParameters();
    ~TestParameters();
    TestParameters(TestParameters &copyMe);

    /*************************************************************************/
    /* Add a global parameter to the test. Call this from the plugin stub */
    int AddString(std::string key, std::string value);
    int AddDouble(std::string key, double value, double milwalue, double maxValue);

    /* Add a subtest parameter. Call this from the plugin stub  */
    int AddSubTestString(std::string subTest, std::string key, std::string value);
    int AddSubTestDouble(std::string subTest, std::string key, double value, double milwalue, double maxValue);

    /*************************************************************************/
    /* Setters. Call these from the config parser */
    int SetString(std::string key, std::string value, bool silent = false);
    int SetDouble(std::string key, double value);

    /* Add a subtest parameter. Call this from the plugin stub  */
    int SetSubTestString(std::string subTest, std::string key, std::string value);
    int SetSubTestDouble(std::string subTest, std::string key, double value);

    /*************************************************************************/
    /* Getters. Call these from within the plugin */
    std::string GetString(std::string key);
    double GetDouble(std::string key);
    int GetBoolFromString(std::string key);
    std::string GetSubTestString(std::string subTest, std::string key);
    double GetSubTestDouble(std::string subTest, std::string key);
    int GetBoolFromSubTestString(std::string subTest, std::string key);

    /*************************************************************************/
    /*
     * Override the parameters in this class from sourceTp if the corresponding
     * parameters exist in sourceTp
     *
     */
    int OverrideFrom(TestParameters *sourceTp);
    int OverrideFromString(const std::string &name, const std::string &value);

    /*************************************************************************/

private:
    std::map<std::string,TestParameterValue *>m_globalParameters;
    std::map<std::string,std::map<std::string,TestParameterValue *> >m_subTestParameters;
};

/*****************************************************************************/
#endif //TESTPARAMETERS_H
