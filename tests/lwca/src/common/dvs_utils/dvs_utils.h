#pragma once

#include "gtest/gtest.h"
#include <lwos.h>
#include <ostream>

namespace dvs {

/* This class generates output that is compatible with lwca/apps DVS automation system. */
class DvsLwdaAppsUnitTestResultPrinter : public ::testing::EmptyTestEventListener
{
public:
    explicit DvsLwdaAppsUnitTestResultPrinter(std::ostream& stream);
    virtual ~DvsLwdaAppsUnitTestResultPrinter();

    void OnTestProgramStart(const ::testing::UnitTest& unit_test) override;
    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override;
    void OnTestStart(const ::testing::TestInfo& test_info) override;
    void OnTestPartResult(const ::testing::TestPartResult& result) override;
    void OnTestEnd(const ::testing::TestInfo& test_info) override;
    void OnTestCaseEnd(const ::testing::TestCase& test_case) override;

private:
    std::ostream& m_stream;
    lwosTimer timer;
    /* We disallow copying EventListeners */
    GTEST_DISALLOW_COPY_AND_ASSIGN_(DvsLwdaAppsUnitTestResultPrinter);
};

}
