#ifndef SYSTEM_CHECKER_H
#define SYSTEM_CHECKER_H

#include <string>

extern const double LOADAVG_THRESHOLD;

class LwvsSystemChecker
{
public:
    std::string CheckSystemInterference();

private:
    std::string m_error;

    void CheckCpuActivity();
};

#endif
