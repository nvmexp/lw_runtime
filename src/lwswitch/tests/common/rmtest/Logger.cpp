/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <iostream>
#include "Logger.h"

using namespace Log;

/**
 * Logger interface ctor
 */
ILogger::ILogger()
{
}

/**
 * Logger interface dtor
 */
ILogger::~ILogger()
{
}

/**
 * NullLogger ctor
 */
NullLogger::NullLogger()
{
}

/**
 * Discard log message
 */
void NullLogger::write(const std::string& msg) const
{
}

/**
 * Discard log message
 */
void NullLogger::write(const boost::format& fmt) const
{
}

/**
 * Discard flush
 */
void NullLogger::flush() const
{
}

/**
 * StdoutLogger ctor
 */
StdoutLogger::StdoutLogger()
{
}

/**
 * Log a message to stdout
 */
void StdoutLogger::write(const std::string& msg) const
{
    std::cout << msg;
}

/**
 * Log a message to stdout
 */
void StdoutLogger::write(const boost::format& fmt) const
{
    std::cout << fmt.str();
}

/**
 * Flush stdout
 */
void StdoutLogger::flush() const
{
    std::cout.flush();
}

/**
 * StderrLogger ctor
 */
StderrLogger::StderrLogger()
{
}

/**
 * Log a message to stderr
 */
void StderrLogger::write(const std::string& msg) const
{
    std::cerr << msg;
}

/**
 * Log a message to stderr
 */
void StderrLogger::write(const boost::format& fmt) const
{
    std::cerr << fmt.str();
}

/**
 * Flush stderr
 */
void StderrLogger::flush() const
{
    std::cerr.flush();
}
