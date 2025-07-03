#ifndef DCGM_ERROR_H
#define DCGM_ERROR_H

#include <string>
#include <sstream>

#include "dcgm_errors.h"
#include "dcgm_structs.h"

/*****************************************************************************/
/* 
 * Priority levels for errors. FAILURE indicates that isolation is needed.
 */
typedef enum dcgmErrorPriority_enum
{
    DCGM_FR_LVL_WARNING     = 1,
    DCGM_FR_LVL_FAILURE     = 2
} dcgmErrorPriority_t;


/*****************************************************************************/
class DcgmError
{
/***************************PUBLIC***********************************/
public:
    /*****************************************************************************/
    DcgmError() : m_code(), m_formatMessage(NULL), m_nextSteps(), m_priority(), m_message(),
                  m_errorDetail()
    {}

    /*****************************************************************************/
    ~DcgmError() {}

    /* Getters and Setters */
    /*****************************************************************************/
    dcgmError_t GetCode() const
    {
        return m_code;
    }

    void SetCode(dcgmError_t code)
    {
        m_code = code;
    }

    /*****************************************************************************/
    const char* GetFormatMessage() const
    {
        return m_formatMessage;
    }

    /*****************************************************************************/
    const std::string& GetMessage() const
    {
        return m_fullError;
    }

    /*****************************************************************************/
    void SetMessage(const std::string& msg)
    {
        m_message = msg;
    
        SetFullError();
    }

    /*****************************************************************************/
    void SetFullError()
    {
        std::stringstream buf;
        buf << m_message;

        if (m_errorDetail.empty() == false)
        {
            buf << m_errorDetail;
        }

        if (m_nextSteps.empty() == false)
        {
            buf << " " << m_nextSteps;
        }

        if (m_details.empty() == false)
        {
            buf << " " << m_details;
        }

        m_fullError = buf.str();
    }
    
    /*****************************************************************************/
    void AddDcgmError(dcgmReturn_t ret)
    {
        std::stringstream buf;
        const char *msg = errorString(ret);
        if (msg != NULL)
        {
            buf << ": '" << errorString(ret) << "' (" << ret << ").";
        }
        else
        {
            buf << ": unknown error code (" << ret << ").";
        }

        m_errorDetail = buf.str();
    
        SetFullError();
    }

    /*****************************************************************************/
    void SetNextSteps(const std::string &nextSteps)
    {
        m_nextSteps = nextSteps;
    
        SetFullError();
    }

    void AddDetail(const std::string &additionalDetail)
    {
        m_details = additionalDetail;

        SetFullError();
    }

    /*****************************************************************************/
    const char* GetNextSteps() const
    {
        return m_nextSteps.c_str();
    }

    /*****************************************************************************/
    dcgmErrorPriority_t GetPriority() const
    {
        return m_priority;
    }

/***************************PRIVATE**********************************/
private:
    /* Private constructor */
    DcgmError(const dcgmError_t code, const char* fmtMsg, const char* nextSteps, const dcgmErrorPriority_t priority);
    
    /* Variables */
    dcgmError_t m_code;
    const char* m_formatMessage;
    std::string m_nextSteps;
    dcgmErrorPriority_t m_priority;

    std::string m_message;
    std::string m_errorDetail;
    std::string m_fullError;
    std::string m_details;
};


// Colwenience macro for formatting and setting the error message
#define DCGM_ERROR_FORMAT_MESSAGE(errCode, errorObj, ...)           \
do                                                                  \
{                                                                   \
    char buf[1024];                                                 \
    const char* fmt = dcgmErrorMeta[errCode].msgFormat;             \
    snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__);                 \
    errorObj.SetMessage(buf);                                       \
    errorObj.SetNextSteps(dcgmErrorMeta[errCode].suggestion);       \
    errorObj.SetCode(errCode);                                      \
} while (0)

// Colwenience macro for formatting and setting the error message and adding an lwml error
#define DCGM_ERROR_FORMAT_MESSAGE_DCGM(errCode, errorObj, dcgmRet, ...)   \
do                                                                        \
{                                                                         \
    char buf[1024];                                                       \
    const char* fmt = errCode##_MSG;                                      \
    snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__);                       \
    errorObj.SetMessage(std::string(buf));                                \
    errorObj.AddDcgmError(dcgmRet);                                       \
    errorObj.SetCode(errCode);                                            \
} while (0)


#endif // DCGM_ERROR_H
