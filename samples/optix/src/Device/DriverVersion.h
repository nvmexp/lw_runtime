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

#include <string>

namespace optix {

class DriverVersion
{
  public:
    /// Initializes the instance with the installed driver version.
    DriverVersion();

    /// Initializes the instance with the given major and minor version numbers.
    /// A value of 0 for major and minor version number can used to construct an invalid instance.
    DriverVersion( unsigned int major, unsigned int minor );

    /// Indicates whether the instance represents a valid driver version.
    bool isValid() const;

    /// Returns the major version of the driver, or 0 if not valid
    unsigned int getMajorVersion() const;

    /// Returns the minor version of the driver, or 0 if not valid
    unsigned int getMinorVersion() const;

    /// Returns the driver version in format MMM.mm, or the empty string if not valid.
    std::string toString() const;

    /// \name Lexicographic comparison of driver versions
    //@{

    bool operator<( const DriverVersion& c ) const;
    bool operator<=( const DriverVersion& c ) const;
    bool operator>( const DriverVersion& c ) const;
    bool operator>=( const DriverVersion& c ) const;
    bool operator==( const DriverVersion& c ) const;
    bool operator!=( const DriverVersion& c ) const;

    //@}

  private:
    std::string  m_version = "0.0";
    unsigned int m_major   = 0;
    unsigned int m_minor   = 0;
};

}  // end namespace optix
