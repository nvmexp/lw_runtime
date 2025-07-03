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
