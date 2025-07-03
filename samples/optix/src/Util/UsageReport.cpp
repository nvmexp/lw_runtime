// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <corelib/system/System.h>
#include <corelib/system/Timer.h>

#include <Util/UsageReport.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

using namespace optix;


UsageReport::IndentFrame::IndentFrame( UsageReport& ur_ )
    : ur( ur_ )
{
    ur.pushIndent();
}


UsageReport::IndentFrame::~IndentFrame()
{
    ur.popIndent();
}


//------------------------------------------------------------------------------
//
// UsageReport definition: simply forwards all functions on to Impl class
//
//------------------------------------------------------------------------------

class UsageReport::UsageReportImpl
{
  public:
    UsageReportImpl();

    bool isActive( int level ) const;

    void setUserCallback( UserCB cb, int level, void* cbdata );
    UserCB getUserCallback() const;
    int    getLevel() const;

    void appendToPreamble( const char* msg );

    void pushIndent();
    void popIndent();

    std::ostream& getStream( int level, const char* tag );
    std::ostream& getPreambleStream();

  private:
    class NullStream : public std::ostream
    {
      public:
        NullStream()
            : std::ostream( &m_buffer )
        {
        }

      private:
        class NullBuffer : public std::streambuf
        {
          public:
            int overflow( int c ) override { return c; }
        };

        NullBuffer m_buffer;
    };


    class CBStream : public std::ostream
    {
      public:
        CBStream()
            : std::ostream( &m_buffer )
        {
        }

        void setUserCallback( UserCB cb, void* cbdata ) { m_buffer.setUserCallback( cb, cbdata ); }

        UserCB getUserCallback() const { return m_buffer.getUserCallback(); }

        void setMessageMetaData( int lvl, const char* tag ) { m_buffer.setMessageMetaData( lvl, tag ); }

        void pushIndent() { m_buffer.pushIndent(); }

        void popIndent() { m_buffer.popIndent(); }
      private:
        class StreamBuf : public std::stringbuf
        {
          public:
            StreamBuf() {}

            void setUserCallback( UserCB cb, void* cbdata )
            {
                m_cb     = cb;
                m_cbdata = cbdata;
            }

            UserCB getUserCallback() const { return m_cb; }

            void setMessageMetaData( int lvl, const char* tag )
            {
                m_msgLvl = lvl;
                m_msgTag = tag;
            }

            void pushIndent() { ++m_indentLvl; }

            void popIndent() { m_indentLvl = std::max<int>( 0, m_indentLvl - 1 ); }

            int sync() override
            {
                std::string indent( m_indentLvl * 4, ' ' );
                if( m_cb )
                {
                    const std::string line = indent + str();
                    m_cb( m_msgLvl, m_msgTag.c_str(), line.c_str(), m_cbdata );
                }
                str( "" );
                return 0;
            }

          private:
            UserCB      m_cb        = nullptr;
            void*       m_cbdata    = nullptr;
            int         m_indentLvl = 0;
            int         m_msgLvl    = 0;
            std::string m_msgTag;
        };

        StreamBuf m_buffer;
    };

    int                m_level;
    CBStream           m_stream;
    NullStream         m_nullStream;
    std::ostringstream m_preamble;
};


UsageReport::UsageReportImpl::UsageReportImpl()
    : m_level( 0 )
{
}


bool UsageReport::UsageReportImpl::isActive( int level ) const
{
    return level <= m_level && getUserCallback() != nullptr;
}


void UsageReport::UsageReportImpl::setUserCallback( UserCB cb, int level, void* cbdata )
{
    m_stream.setUserCallback( cb, cbdata );
    m_level = level;
    getStream( 1, "SYS INFO" ) << m_preamble.str() << std::endl;
}


UsageReport::UserCB UsageReport::UsageReportImpl::getUserCallback() const
{
    return m_stream.getUserCallback();
}


int UsageReport::UsageReportImpl::getLevel() const
{
    return m_level;
}


void UsageReport::UsageReportImpl::pushIndent()
{
    m_stream.pushIndent();
}


void UsageReport::UsageReportImpl::popIndent()
{
    m_stream.popIndent();
}


std::ostream& UsageReport::UsageReportImpl::getStream( int level, const char* tag )
{
    if( isActive( level ) )
    {
        m_stream.setMessageMetaData( level, tag );
        return m_stream;
    }
    return m_nullStream;
}


std::ostream& UsageReport::UsageReportImpl::getPreambleStream()
{
    return m_preamble;
}


//------------------------------------------------------------------------------
//
// UsageReport definition: simply forwards all functions on to Impl class
//
//------------------------------------------------------------------------------

UsageReport::UsageReport()
    : m_impl( new UsageReportImpl )
{
}


UsageReport::~UsageReport()
{
}


bool UsageReport::isActive( int level ) const
{
    return m_impl->isActive( level );
}


void UsageReport::setUserCallback( UserCB cb, int level, void* cbdata )
{
    m_impl->setUserCallback( cb, level, cbdata );
}


optix::UsageReport::UserCB UsageReport::getUserCallback() const
{
    return m_impl->getUserCallback();
}


int UsageReport::getLevel() const
{
    return m_impl->getLevel();
}


void UsageReport::pushIndent()
{
    m_impl->pushIndent();
}


void UsageReport::popIndent()
{
    m_impl->popIndent();
}


std::ostream& UsageReport::getStream( int level, const char* tag )
{
    return m_impl->getStream( level, tag );
}


std::ostream& UsageReport::getPreambleStream()
{
    return m_impl->getPreambleStream();
}
