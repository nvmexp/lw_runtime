#include <iostream> 
#include <fstream> 
#include <sstream>

#include "LwvsSystemChecker.h"
#include "logging.h"

// Observed measurements show that the diagnostic will monopolize the CPUs at times, so we 
// should recommend against training at anything .5 and above.
const double LOADAVG_THRESHOLD = .5;

std::string LwvsSystemChecker::CheckSystemInterference()
{
    m_error.clear();

    CheckCpuActivity();

    return m_error;
}
    
void LwvsSystemChecker::CheckCpuActivity()
{
    std::ifstream loadfile("/proc/loadavg");
    std::string line;
    double data[3];
    if (std::getline(loadfile, line))
    {
        std::istringstream iss(line);

        for (int i = 0; i < 3; i++)
            iss >> data[i];

        if (data[0] >= LOADAVG_THRESHOLD)
        {
            std::stringstream error;
            error << "Loadavg should be below " << LOADAVG_THRESHOLD <<
                     " to train the diagnostic, but is " << data[0] << ".";
            m_error = error.str();
        }
    }
    else
    {
        PRINT_DEBUG("", "Unable to read a line from /proc/loadavg: please check the syslog.");
    }
}

