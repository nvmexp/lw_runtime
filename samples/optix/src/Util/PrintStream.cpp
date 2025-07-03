// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES


#include <Util/PrintStream.h>

#include <iostream>


namespace {

template <typename T>
void printArray( std::ostream& out, char* data, size_t size )
{
    T*        t_data  = reinterpret_cast<T*>( data );
    const int t_count = size / sizeof( T );
    for( int i = 0; i < t_count; ++i )
    {
        out << t_data[i];
        if( i != t_count - 1 )
            out << ",";
    }
}

}  // anon namespace


optix::PrintStream::PrintStream( std::ostream& out )
    : PersistentStream( PersistentStream::Writing )
    , m_out( out )
{
}


void optix::PrintStream::pushLabel( const char* label, const char* classname )
{
    m_out << m_scope_indent << label << ": " << classname << std::endl;
    m_scope_indent.push_back( '\t' );
}


void optix::PrintStream::popLabel()
{
    m_scope_indent.pop_back();
}


void optix::PrintStream::readOrWriteObjectVersion( const unsigned int* version )
{
    optix::readOrWrite( this, &version[0], "version[0]" );
    optix::readOrWrite( this, &version[1], "version[1]" );
    optix::readOrWrite( this, &version[2], "version[2]" );
    optix::readOrWrite( this, &version[3], "version[3]" );
}


void optix::PrintStream::readOrWrite( char* data, size_t size, const char* label, Format format )
{
    // WAR for bug 3059486. 'label' can sometimes be NULL, which causes a segfault.
    m_out << m_scope_indent << ( label ? label : "[NULL]" ) << ": ";
    switch( format )
    {
        case Opaque:
        case None:
        {
            for( int i = 0; i < size; ++i )
                m_out << std::hex << reinterpret_cast<unsigned char*>( data )[i];
        }
        break;

        case String:
        {
            m_out << data;
        }
        break;

        case Bool:
        {
            printArray<bool>( m_out, data, size );
        }
        break;

        case Char:
        {
            printArray<char>( m_out, data, size );
        }
        break;

        case Int:
        {
            printArray<int>( m_out, data, size );
        }
        break;

        case UInt:
        {
            printArray<unsigned int>( m_out, data, size );
        }
        break;

        case Short:
        {
            printArray<short>( m_out, data, size );
        }
        break;

        case UShort:
        {
            printArray<unsigned short>( m_out, data, size );
        }
        break;

        case ULong:
        {
            printArray<unsigned long>( m_out, data, size );
        }
        break;

        case LongLong:
        {
            printArray<long long>( m_out, data, size );
        }
        break;

        case ULongLong:
        {
            printArray<unsigned long long>( m_out, data, size );
        }
        break;
    };

    m_out << std::endl;
}
