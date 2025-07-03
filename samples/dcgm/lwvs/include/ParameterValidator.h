#include <string>
#include <map>
#include <set>

typedef struct
{
    std::string testname;
    std::set<std::string> parameters;
} subtestInfo_t;

class TestInfo
{
public:
    void Clear();
    void AddParameter(const std::string &parameter);
    void SetName(const std::string &testname);
    void AddSubtest(const std::string &subtest);
    void AddSubtestParameter(const std::string &subtest, const std::string &parameter);

    bool HasParameter(const std::string &parameter) const;
    bool HasSubtest(const std::string &subtest) const;
    bool HasSubtestParameter(const std::string &subtest, const std::string &parameter);
    subtestInfo_t m_info;
    std::map<std::string, subtestInfo_t> m_subtests;
};

class ParameterValidator
{
public:
    ParameterValidator();
    void Init();

    /*
     * IsValidTestName()
     *
     * Return true if the test name is among the valid choices
     *        false if it isn't
     */
    bool IsValidTestName(const std::string &testname) const;

    /*
     * IsValidParameter()
     *
     * Return true if the parameter is among the valid choices for the specified test
     *        false if it isn't
     */
    bool IsValidParameter(const std::string &testname, const std::string &parameter);

    /*
     * IsValidSubtest()
     *
     * Return true if the subtest is among the valid choices for the specified test
     *        false if it isn't
     */
    bool IsValidSubtest(const std::string &testname, const std::string &subtest);


    /*
     * IsValidSubtestParameter()
     *
     * Return true if the parameter is among the valid choices for the specified test and subtest
     *        false if it isn't
     */
    bool IsValidSubtestParameter(const std::string &testname, const std::string &subtest,
                                 const std::string &parameter);

private:
    std::map<std::string, TestInfo> m_possiblePlugins;

    void AddBusGrind();
    void AddSoftware();
    void AddTargetPower();
    void AddTargetStress();
    void AddMemory();
    void AddSMStress();
    void AddGpuBurn();
    void AddContextCreate();
    void AddMemoryBandwidth();
};

