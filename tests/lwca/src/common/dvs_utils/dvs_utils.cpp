#include "dvs_utils/dvs_utils.h"
#include <string>

namespace dvs {

/**********************************************************
 * DVS lwca/apps test result printer class implementation *
 **********************************************************/

DvsLwdaAppsUnitTestResultPrinter::DvsLwdaAppsUnitTestResultPrinter(std::ostream& s) : m_stream(s), timer()
{
}

DvsLwdaAppsUnitTestResultPrinter::~DvsLwdaAppsUnitTestResultPrinter()
{
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestProgramStart(const ::testing::UnitTest& unit_test)
{
    lwosResetTimer(&timer);
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestProgramEnd(const ::testing::UnitTest& unit_test)
{
    m_stream << "&&&&  ";
    if (unit_test.Failed()) {
        m_stream << "FAILED";
    }
    else if (unit_test.Passed()) {
        m_stream << "PASSED";
    }
    else {
        m_stream << "WAIVED";
    }

    m_stream << " (" << unit_test.elapsed_time() << "ms)" << std::endl;
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestStart(const ::testing::TestInfo& test_info)
{
    if (test_info.should_run()) {
        m_stream << "vvvv Running: " << test_info.test_case_name() << '.' << test_info.name() << std::endl;
    }
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestPartResult(const ::testing::TestPartResult& result)
{
    if (result.passed() /* || result.skipped() */) {
        return;
    }

    m_stream << "FAILURE at " << result.file_name() << ':' << result.line_number() << std::endl
             << '\t' << result.message() << std::endl;
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestEnd(const ::testing::TestInfo& test_info)
{
    m_stream << "^^^^ ";

    if (test_info.result()->Passed()) {
        m_stream << "PASS";
    }
    else if (test_info.result()->Failed()) {
        m_stream << "FAILED";
    }
    else {
        m_stream << "WAIVED";
    }

    m_stream << ": " << test_info.test_case_name() << '.' << test_info.name()
             << " (" << test_info.result()->elapsed_time() << "ms)" << std::endl;
}

void DvsLwdaAppsUnitTestResultPrinter::OnTestCaseEnd(const ::testing::TestCase &)
{
    // Useful for checking how long the full test run has been taking without having
    // to sum up all the individual test cases
    m_stream << "**** Total running time: " << lwosGetTimer(&timer) << "ms" << std::endl;
}

}
