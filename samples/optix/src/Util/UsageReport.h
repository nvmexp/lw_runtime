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

#pragma once

#include <iosfwd>
#include <memory>

namespace optix {

class UsageReport
{
  public:
    /// Signature of user callback function.  The message level is the first
    /// param.  A descriptive tag string (eg STAT, ERROR, WARN) will be passed
    /// in the 2nd argument, the 3rd arg will be the message itself and the 4th
    /// arg is a user-defined callback data pointer.
    typedef void ( *UserCB )( int, const char*, const char*, void* );

    UsageReport();
    ~UsageReport();

    bool isActive( int level ) const;

    void setUserCallback( UserCB cb, int level, void* cbdata );
    UserCB getUserCallback() const;
    int    getLevel() const;

    void pushIndent();
    void popIndent();

    std::ostream& getStream( int level, const char* tag );

    // The preamble is the header message(s) printed at the top of the usage
    // report.  Use the preamble stream to create this header.  Anything in the
    // header will be sent to the callback when setUserCallback is ilwoked.
    std::ostream& getPreambleStream();

    struct IndentFrame
    {
        IndentFrame( UsageReport& ur );
        ~IndentFrame();
        UsageReport& ur;
    };

  private:
    class UsageReportImpl;
    std::unique_ptr<UsageReportImpl> m_impl;

    // Non-copyable
    UsageReport( const UsageReport& ) = delete;
    UsageReport& operator=( const UsageReport& ) = delete;
};

}  // namespace optix


#define ureport_if_active( usage_report, level )                                                                       \
    if( !( usage_report ).isActive( level ) )                                                                          \
        ;                                                                                                              \
    else

#define ureport0( usage_report, tag ) ureport_if_active( usage_report, 0 )( usage_report ).getStream( 0, tag )
#define ureport1( usage_report, tag ) ureport_if_active( usage_report, 1 )( usage_report ).getStream( 1, tag )
#define ureport2( usage_report, tag ) ureport_if_active( usage_report, 2 )( usage_report ).getStream( 2, tag )
