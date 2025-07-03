#pragma once

#include "gtest/gtest.h"
#include <lwos.h>
#include <vector>

#define SCOPED_TRACE_STREAM(s) SCOPED_TRACE(::testing::Message() << s)

namespace utils
{

class QuietUnitTestResultPrinter : public ::testing::EmptyTestEventListener
{
public:
    explicit QuietUnitTestResultPrinter(std::ostream& stream);
    virtual ~QuietUnitTestResultPrinter();

    void OnTestPartResult(const ::testing::TestPartResult& result) override;
private:
    std::ostream& m_stream;
    // We disallow copying EventListeners
    GTEST_DISALLOW_COPY_AND_ASSIGN_(QuietUnitTestResultPrinter);
};

/* Utility class referencing another process */
class Process
{
public:
    typedef std::vector<Process *> ProcessList;
    Process();
    ~Process();
    int start(const char *programName, std::vector<const char *> args);
    int wait();
    int terminate();
    LWOSPid getProcessId() const;
    /**
    * \brief Waits on all the given \p processes for one to complete
    *
    * \param processes list of processes
    * \param idx Index in \p processes that has completed
    *
    * \return Error code for the completed process
    */
    static int waitAny(ProcessList& processes, size_t& idx);
    /**
    * \brief Waits for all the given \p processes to complete
    *
    * \param processes
    * \param exitCodes
    */
    static void waitAll(ProcessList& processes, std::vector<int>& exitCodes);
private:
    unsigned long long processHandle;
    LWOSthread *thread;
    // We disallow copying Processes
    GTEST_DISALLOW_COPY_AND_ASSIGN_(Process);
};

class BaseElwironment : public ::testing::Environment
{
public:
    virtual void parseArguments(int argc, char **argv) = 0;
};

class UtilsElwironment : public BaseElwironment
{
    bool m_isStressMode;
    bool m_isChildProcess;
    LWOSPid m_parentProcessId;
    std::string m_extra;
    size_t m_processIdx;
public:
    UtilsElwironment(const char *programName);
    virtual ~UtilsElwironment();
    void SetUp() override;
    void TearDown() override;
    void parseArguments(int argc, char **argv) override;
    bool createProcess(Process& process, size_t idx = 0, std::string extra = "") const;
    bool isStressMode() const {
        return m_isStressMode;
    }
    bool isChildProcess() const {
        return m_isChildProcess;
    }
    LWOSPid getParentProcessId() const {
        return m_parentProcessId;
    }
    std::string getExtraOption() const {
        return m_extra;
    }
    size_t getProcessIndex() const {
        return m_processIdx;
    }
    static const UtilsElwironment *getElw();
    const std::string programName;
protected:
    static const char *parseFlagValue(const char *str, const char *flag,
                                      bool def_optional);
private:
    // We disallow copying Elwironments
    GTEST_DISALLOW_COPY_AND_ASSIGN_(UtilsElwironment);
};

}
