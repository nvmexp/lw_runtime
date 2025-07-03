/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <iostream>
#include <boost/format.hpp>

namespace Log
{
    /**
     * Generic logging interface
     */
    class ILogger
    {
        public:
            ILogger();
            virtual ~ILogger()=0;

            virtual void write(const std::string& msg) const=0;
            virtual void write(const boost::format& fmt) const=0;
            virtual void flush() const=0;

        private:
            ILogger(const ILogger&);
            ILogger& operator=(const ILogger&);
    };

    /**
     * Logging implementation that does nothing
     */
    class NullLogger : public ILogger
    {
        public:
            NullLogger();

            virtual void write(const std::string& msg) const;
            virtual void write(const boost::format& fmt) const;
            virtual void flush() const;
    };

    /**
     * Logging implementation that writes to stdout
     */
    class StdoutLogger : public ILogger
    {
        public:
            StdoutLogger();

            virtual void write(const std::string& msg) const;
            virtual void write(const boost::format& fmt) const;
            virtual void flush() const;
    };

    /**
     * Logging implementation that writes to stderr
     */
    class StderrLogger : public ILogger
    {
    public:
      StderrLogger();

      virtual void write(const std::string& msg) const;
      virtual void write(const boost::format& fmt) const;
      virtual void flush() const;
    };
}

#endif // _LOGGER_H_
