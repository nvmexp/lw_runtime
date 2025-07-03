/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
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

