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

#ifndef _ELW_H_
#define _ELW_H_

#include "gtest/gtest.h"
#include "UtilOS.h"
#include "Logger.h"

/**
 * Interface for an environment that has a logger.
 */
class ILoggedElwironment : public ::testing::Environment
{
public:
    virtual void SetUp()=0;
    virtual void TearDown()=0;
    virtual const Log::ILogger& getLogger() const=0;
    virtual const Log::ILogger& getRmApiLogger() const=0;
};

/**
 * Tests that use this environment should require no admin privileges.
 */
class NonPrivilegedElwironment : public ILoggedElwironment
{
public:
    NonPrivilegedElwironment(Log::ILogger& log, Log::ILogger& logRmApi);
    virtual void SetUp();
    virtual void TearDown();
    virtual const Log::ILogger& getLogger() const;
    virtual const Log::ILogger& getRmApiLogger() const;

private:
    Log::ILogger& m_log;
    Log::ILogger& m_logRmApi;
};

/**
 * Downgrades the user privileges to non-elevated by default.
 * If a test fixture or a test case wants to run as admin it can call
 * RestoreAdminPrivileges();
 */
class LeastPrivilegedElwironment : public ILoggedElwironment
{
public:
    LeastPrivilegedElwironment(Log::ILogger& log, Log::ILogger& logRmApi);
    virtual void SetUp();
    virtual void TearDown();
    virtual const Log::ILogger& getLogger() const;
    virtual const Log::ILogger& getRmApiLogger() const;

private:
    Log::ILogger& m_log;
    Log::ILogger& m_logRmApi;
};

/**
 * Pointer to the logged environment so that the test can access it. main() is
 * responsible for initializing this
 */
extern ILoggedElwironment* g_pElw;


#endif // _ELW_H_
