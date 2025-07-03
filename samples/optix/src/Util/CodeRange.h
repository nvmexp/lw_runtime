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

#include <string>

namespace optix {

// We're not using CodeRange at them moment. I'm turning it into NOPs
// until we decide whether to keep it or not to make sure it's as low
// overhead as it can get without touching the client code. --Martin
#if 1

class CodeRange
{
  public:
    CodeRange() {}
    CodeRange( const char* ) {}
    void operator()( const char* ) {}
    void start( const char* ) {}
    void end() {}

    class IHandler
    {
      public:
        virtual ~IHandler() {}
        virtual void push( const char* ) = 0;
        virtual void pop()               = 0;
    };

    static void setHandler( IHandler* handler ) {}
    static void                       push( const char* ) {}
    static void                       pop() {}
};

#else
/// Generic interface for marking a region of code.
///
/// A CodeRange object provides an RAII-like idiom that ensures that pushed ranges
/// get popped. The CodeRange class also has a static interface for directly pushing
/// and popping named ranges. A CodeRange::IHandler provides an interface for
/// handling pushes and pops.
class CodeRange
{
  public:
    /// Default constructor. Does not push a range.
    CodeRange()
        : m_active( false )
    {
    }

    /// Starts a new range
    CodeRange( const std::string& name )
        : m_active( false )
    {
        start( name );
    }

    ~CodeRange() { end(); }

    /// Syntactic sugar on start().
    ///
    /// Example:
    /// @code
    ///   CodeRange range( "Range 1" )
    ///   ...
    ///   range( "Range 2" );
    ///   ...
    /// @endcode
    void operator()( const std::string& name ) { start( name ); }

    /// Starts a new range and ends any previous one.
    void start( const std::string& name )
    {
        end();
        m_active = true;
        push( name );
    }

    /// Ends the range.
    void end()
    {
        if( m_active )
        {
            pop();
            m_active = false;
        }
    }

  private:
    bool m_active;


    ////////////////////////////////
    //
    // STATIC API
    //
    ////////////////////////////////
  public:
    class IHandler
    {
      public:
        virtual ~IHandler() {}
        virtual void push( const std::string& name ) = 0;
        virtual void pop()                           = 0;
    };

    static void setHandler( IHandler* handler );
    static void push( const std::string& name );
    static void pop();

  private:
    static IHandler* ms_handler;
};


inline void CodeRange::setHandler( IHandler* handler )
{
    ms_handler = handler;
}

inline void CodeRange::push( const std::string& name )
{
    if( ms_handler )
        ms_handler->push( name );
}

inline void CodeRange::pop()
{
    if( ms_handler )
        return ms_handler->pop();
}

#endif

}  // namespace optix
