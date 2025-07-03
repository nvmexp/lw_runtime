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

// Helper for profiling with LWPU Nsight tools.
//
// If Nsight is installed, find lwToolsExt headers and libs path in
// environment variable LWTOOLSEXT_PATH. We expect that LWTX_AVAILABLE is defined then.

// Define LWTX_ENABLE before including this header to enable lwToolsExt markers and ranges.
//#define LWTX_ENABLE


#ifndef LWTX_AVAILABLE
#ifdef LWTX_ENABLE
#undef LWTX_ENABLE
#endif
#endif


#ifdef _WIN32
// Emulate stdint.h header.
#ifndef __stdint_h__
#define __stdint_h__
typedef signed char      int8_t;
typedef unsigned char    uint8_t;
typedef short            int16_t;
typedef unsigned short   uint16_t;
typedef __int32          int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64          int64_t;
typedef unsigned __int64 uint64_t;
#endif  // file guard
// Control inclusion of stdint.h in lwToolsExt.h
#define LWTX_STDINT_TYPES_ALREADY_DEFINED

#else

#include <stdint.h>
// Control inclusion of stdint.h in lwToolsExt.h
#define LWTX_STDINT_TYPES_ALREADY_DEFINED

#endif  //_WIN32


#ifdef LWTX_ENABLE

#include "lwToolsExt.h"

#ifdef _WIN64
#pragma comment( lib, "lwToolsExt64_1.lib" )
#else
#pragma comment( lib, "lwToolsExt32_1.lib" )
#endif

#define LWTX_MarkEx lwtxMarkEx
#define LWTX_MarkA lwtxMarkA
#define LWTX_MarkW lwtxMarkW
#define LWTX_RangeStartEx lwtxRangeStartEx
#define LWTX_RangeStartA lwtxRangeStartA
#define LWTX_RangeStartW lwtxRangeStartW
#define LWTX_RangeEnd lwtxRangeEnd
#define LWTX_RangePushEx lwtxRangePushEx
#define LWTX_RangePushA lwtxRangePushA
#define LWTX_RangePushW lwtxRangePushW
#define LWTX_RangePop lwtxRangePop
#define LWTX_NameOsThreadA lwtxNameOsThreadA
#define LWTX_NameOsThreadW lwtxNameOsThreadW

#else

struct lwtxEventAttributes_t
{
};
typedef uint64_t lwtxRangeId_t;

#ifndef _MSC_VER
#define __noop( ... )
#endif

#define LWTX_MarkEx __noop
#define LWTX_MarkA __noop
#define LWTX_MarkW __noop
#define LWTX_RangeStartEx __noop
#define LWTX_RangeStartA __noop
#define LWTX_RangeStartW __noop
#define LWTX_RangeEnd __noop
#define LWTX_RangePushEx __noop
#define LWTX_RangePushA __noop
#define LWTX_RangePushW __noop
#define LWTX_RangePop __noop
#define LWTX_NameOsThreadA __noop
#define LWTX_NameOsThreadW __noop


#endif

// C++ function templates to enable LwToolsExt functions
namespace lwtx {
#ifdef LWTX_ENABLE

class Attributes
{
  public:
    Attributes() { clear(); }
    Attributes& category( uint32_t category )
    {
        m_event.category = category;
        return *this;
    }
    Attributes& color( uint32_t argb )
    {
        m_event.colorType = LWTX_COLOR_ARGB;
        m_event.color     = argb;
        return *this;
    }
    Attributes& payload( uint64_t value )
    {
        m_event.payloadType      = LWTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        m_event.payload.ullValue = value;
        return *this;
    }
    Attributes& payload( int64_t value )
    {
        m_event.payloadType     = LWTX_PAYLOAD_TYPE_INT64;
        m_event.payload.llValue = value;
        return *this;
    }
    Attributes& payload( double value )
    {
        m_event.payloadType    = LWTX_PAYLOAD_TYPE_DOUBLE;
        m_event.payload.dValue = value;
        return *this;
    }
    Attributes& message( const char* message )
    {
        m_event.messageType   = LWTX_MESSAGE_TYPE_ASCII;
        m_event.message.ascii = message;
        return *this;
    }
    Attributes& message( const wchar_t* message )
    {
        m_event.messageType     = LWTX_MESSAGE_TYPE_UNICODE;
        m_event.message.unicode = message;
        return *this;
    }
    Attributes& clear()
    {
        memset( &m_event, 0, LWTX_EVENT_ATTRIB_STRUCT_SIZE );
        m_event.version = LWTX_VERSION;
        m_event.size    = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
        return *this;
    }
    const lwtxEventAttributes_t* out() const { return &m_event; }
  private:
    lwtxEventAttributes_t m_event;
};


class ScopedRange
{
  public:
    ScopedRange( const char* message ) { lwtxRangePushA( message ); }
    ScopedRange( const wchar_t* message ) { lwtxRangePushW( message ); }
    ScopedRange( const lwtxEventAttributes_t* attributes ) { lwtxRangePushEx( attributes ); }
    ScopedRange( const lwtx::Attributes& attributes ) { lwtxRangePushEx( attributes.out() ); }
    ~ScopedRange() { lwtxRangePop(); }
};

inline void Mark( const lwtx::Attributes& attrib )
{
    lwtxMarkEx( attrib.out() );
}
inline void Mark( const lwtxEventAttributes_t* eventAttrib )
{
    lwtxMarkEx( eventAttrib );
}
inline void Mark( const char* message )
{
    lwtxMarkA( message );
}
inline void Mark( const wchar_t* message )
{
    lwtxMarkW( message );
}
inline lwtxRangeId_t RangeStart( const lwtx::Attributes& attrib )
{
    return lwtxRangeStartEx( attrib.out() );
}
inline lwtxRangeId_t RangeStart( const lwtxEventAttributes_t* eventAttrib )
{
    return lwtxRangeStartEx( eventAttrib );
}
inline lwtxRangeId_t RangeStart( const char* message )
{
    return lwtxRangeStartA( message );
}
inline lwtxRangeId_t RangeStart( const wchar_t* message )
{
    return lwtxRangeStartW( message );
}
inline void RangeEnd( lwtxRangeId_t id )
{
    lwtxRangeEnd( id );
}
inline int RangePush( const lwtx::Attributes& attrib )
{
    return lwtxRangePushEx( attrib.out() );
}
inline int RangePush( const lwtxEventAttributes_t* eventAttrib )
{
    return lwtxRangePushEx( eventAttrib );
}
inline int RangePush( const char* message )
{
    return lwtxRangePushA( message );
}
inline int RangePush( const wchar_t* message )
{
    return lwtxRangePushW( message );
}
inline void RangePop()
{
    lwtxRangePop();
}
inline void NameCategory( uint32_t category, const char* name )
{
    lwtxNameCategoryA( category, name );
}
inline void NameCategory( uint32_t category, const wchar_t* name )
{
    lwtxNameCategoryW( category, name );
}
inline void NameOsThread( uint32_t threadId, const char* name )
{
    lwtxNameOsThreadA( threadId, name );
}
inline void NameOsThread( uint32_t threadId, const wchar_t* name )
{
    lwtxNameOsThreadW( threadId, name );
}
inline void NameLwrrentThread( const char* name )
{
    lwtxNameOsThreadA(::GetLwrrentThreadId(), name );
}
inline void NameLwrrentThread( const wchar_t* name )
{
    lwtxNameOsThreadW(::GetLwrrentThreadId(), name );
}

#else

class Attributes
{
  public:
    Attributes() {}
    Attributes& category( uint32_t category ) { return *this; }
    Attributes& color( uint32_t argb ) { return *this; }
    Attributes& payload( uint64_t value ) { return *this; }
    Attributes& payload( int64_t value ) { return *this; }
    Attributes& payload( double value ) { return *this; }
    Attributes& message( const char* message ) { return *this; }
    Attributes& message( const wchar_t* message ) { return *this; }
    Attributes&                         clear() { return *this; }
    const lwtxEventAttributes_t*        out() { return 0; }
};

class ScopedRange
{
  public:
    ScopedRange( const char* message ) { (void)message; }
    ScopedRange( const wchar_t* message ) { (void)message; }
    ScopedRange( const lwtxEventAttributes_t* attributes ) { (void)attributes; }
    ScopedRange( const Attributes& attributes ) { (void)attributes; }
    ~ScopedRange() {}
};

inline void Mark( const lwtx::Attributes& attrib )
{
    (void)attrib;
}
inline void Mark( const lwtxEventAttributes_t* eventAttrib )
{
    (void)eventAttrib;
}
inline void Mark( const char* message )
{
    (void)message;
}
inline void Mark( const wchar_t* message )
{
    (void)message;
}
inline lwtxRangeId_t RangeStart( const lwtx::Attributes& attrib )
{
    (void)attrib;
    return 0;
}
inline lwtxRangeId_t RangeStart( const lwtxEventAttributes_t* eventAttrib )
{
    (void)eventAttrib;
    return 0;
}
inline lwtxRangeId_t RangeStart( const char* message )
{
    (void)message;
    return 0;
}
inline lwtxRangeId_t RangeStart( const wchar_t* message )
{
    (void)message;
    return 0;
}
inline void RangeEnd( lwtxRangeId_t id )
{
    (void)id;
}
inline int RangePush( const lwtx::Attributes& attrib )
{
    (void)attrib;
    return -1;
}
inline int RangePush( const lwtxEventAttributes_t* eventAttrib )
{
    (void)eventAttrib;
    return -1;
}
inline int RangePush( const char* message )
{
    (void)message;
    return -1;
}
inline int RangePush( const wchar_t* message )
{
    (void)message;
    return -1;
}
inline int RangePop()
{
    return -1;
}
inline void NameCategory( uint32_t category, const char* name )
{
    (void)category;
    (void)name;
}
inline void NameCategory( uint32_t category, const wchar_t* name )
{
    (void)category;
    (void)name;
}
inline void NameOsThread( uint32_t threadId, const char* name )
{
    (void)threadId;
    (void)name;
}
inline void NameOsThread( uint32_t threadId, const wchar_t* name )
{
    (void)threadId;
    (void)name;
}
inline void NameLwrrentThread( const char* name )
{
    (void)name;
}
inline void NameLwrrentThread( const wchar_t* name )
{
    (void)name;
}

#endif
}  //lwtx
