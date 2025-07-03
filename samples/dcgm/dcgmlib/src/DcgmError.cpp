#include <string>

#include "DcgmError.h"
#include "DcgmGPUHardwareLimits.h"

#define DEF_ERROR(code, nextSteps, priority)                                                                           \
    case code:                                                                                                         \
    {                                                                                                                  \
        return DcgmError(code, code##_MSG, nextSteps, priority);                                                       \
    }


/*****************************************************************************/
DcgmError::DcgmError(const dcgmError_t code,
                     const char* fmtMsg,
                     const char* nextSteps,
                     const dcgmErrorPriority_t priority)
    : m_code(code)
    , m_formatMessage(fmtMsg)
    , m_nextSteps(nextSteps)
    , m_priority(priority)
{}

