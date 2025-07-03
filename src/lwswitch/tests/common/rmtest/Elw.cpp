/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "Elw.h"

using namespace Log;

/**
 * @param[in] log General logs will be written to this object
 * @param[in] logRmApi RmApiWrap object logs will be written to this object
 */
LeastPrivilegedElwironment::LeastPrivilegedElwironment(ILogger& log, ILogger& logRmApi)
    : m_log(log), m_logRmApi(logRmApi)
{
}

void LeastPrivilegedElwironment::SetUp()
{
    if (osIsUserAdmin())
        osDropAdminPrivileges();
}

void LeastPrivilegedElwironment::TearDown()
{
    if (osIsUserAdmin())
        osDropAdminPrivileges();
}

const ILogger& LeastPrivilegedElwironment::getLogger() const
{
    return m_log;
}

const ILogger& LeastPrivilegedElwironment::getRmApiLogger() const
{
    return m_logRmApi;
}

/**
 * @param[in] log General logs will be written to this object
 * @param[in] logRmApi RmApiWrap object logs will be written to this object
 */
NonPrivilegedElwironment::NonPrivilegedElwironment(ILogger& log, ILogger& logRmApi)
    : m_log(log), m_logRmApi(logRmApi)
{
}

void NonPrivilegedElwironment::SetUp()
{
}

void NonPrivilegedElwironment::TearDown()
{
}

const ILogger& NonPrivilegedElwironment::getLogger() const
{
    return m_log;
}

const ILogger& NonPrivilegedElwironment::getRmApiLogger() const
{
    return m_logRmApi;
}

