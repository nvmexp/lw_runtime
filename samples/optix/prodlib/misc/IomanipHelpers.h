
// Copyright LWPU Corporation 2012
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
#include <ios>


namespace prodlib {

// Saves the current set of stream flags, then restores them when going out of scope.
class IOSSaver
{
  public:
    explicit IOSSaver( std::ios& ios )
        : m_ios( ios )
    {
        m_flags     = m_ios.flags();
        m_precision = m_ios.precision();
        m_fill      = m_ios.fill();
    }
    ~IOSSaver()
    {
        m_ios.flags( m_flags );
        m_ios.precision( m_precision );
        m_ios.fill( m_fill );
    }
    std::ios_base::fmtflags flags() const { return m_flags; }
    std::streamsize         precision() const { return m_precision; }
    char                    fill() const { return m_fill; }
  private:
    std::ios&               m_ios;
    std::ios_base::fmtflags m_flags;
    std::streamsize         m_precision;
    char                    m_fill;

  protected:
    // Don't copy objects of this class
    IOSSaver( IOSSaver& );
    IOSSaver& operator=( IOSSaver& );
};

}  // end namespace corelib
