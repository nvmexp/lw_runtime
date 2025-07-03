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

#include <iomanip>
#include <sstream>

#if defined( _WIN32 )
#include <support/lwapi/lwapi.h>
#else
#include <Util/LWML.h>
#endif

#include <Device/DriverVersion.h>
#include <corelib/misc/String.h>

namespace optix {

DriverVersion::DriverVersion()
{
#if defined( _WIN32 )
    LwAPI_Status      ret;
    LwU32             driverVersion;
    LwAPI_ShortString buildBranch;

    ret = LwAPI_SYS_GetDriverAndBranchVersion( &driverVersion, buildBranch );
    if( ret == LWAPI_OK )
    {
        m_major = driverVersion / 100;
        m_minor = driverVersion % 100;
    }
#else
    const std::string version = LWML::driverVersion();
    if( version != "unknown" )
    {
        // The version string might be either major.minor or major.minor.patch.
        // We are ignoring the patch version for now since it is not available
        // in Windows.
        std::vector<std::string> tokens = corelib::tokenize( version, "." );
        if( tokens.size() == 2 || tokens.size() == 3 )
        {
            bool okMajor, okMinor;
            int  major = corelib::from_string<int>( tokens[0], &okMajor );
            int  minor = corelib::from_string<int>( tokens[1], &okMinor );
            if( okMajor && okMinor )
            {
                m_major = major;
                m_minor = minor;
            }
        }
    }
#endif

    if( isValid() )
    {
        std::stringstream ss;
        ss << m_major << "." << std::setfill( '0' ) << std::setw( 2 ) << m_minor;
        m_version = ss.str();
    }
}

DriverVersion::DriverVersion( unsigned int majorVersion, unsigned int minorVersion )
    : m_major( majorVersion )
    , m_minor( minorVersion )
{
    if( isValid() )
    {
        std::stringstream ss;
        ss << m_major << "." << std::setfill( '0' ) << std::setw( 2 ) << m_minor;
        m_version = ss.str();
    }
}

bool DriverVersion::isValid() const
{
    return m_major != 0 || m_minor != 0;
}

unsigned int DriverVersion::getMajorVersion() const
{
    return m_major;
}

unsigned int DriverVersion::getMinorVersion() const
{
    return m_minor;
}

std::string DriverVersion::toString() const
{
    return m_version;
}

bool DriverVersion::operator<( const DriverVersion& c ) const
{
    return m_major < c.m_major || ( m_major == c.m_major && m_minor < c.m_minor );
}

bool DriverVersion::operator<=( const DriverVersion& c ) const
{
    return m_major < c.m_major || ( m_major == c.m_major && m_minor <= c.m_minor );
}

bool DriverVersion::operator>( const DriverVersion& c ) const
{
    return m_major > c.m_major || ( m_major == c.m_major && m_minor > c.m_minor );
}

bool DriverVersion::operator>=( const DriverVersion& c ) const
{
    return m_major > c.m_major || ( m_major == c.m_major && m_minor >= c.m_minor );
}

bool DriverVersion::operator==( const DriverVersion& c ) const
{
    return m_major == c.m_major && m_minor == c.m_minor;
}

bool DriverVersion::operator!=( const DriverVersion& c ) const
{
    return m_major != c.m_major || m_minor != c.m_minor;
}
}
