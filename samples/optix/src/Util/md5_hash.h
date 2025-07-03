/***************************************************************************************************
 * Copyright (c) 2017, LWPU CORPORATION. All rights reserved.
 **************************************************************************************************/
/**
   \file
   \brief        MD5 hash value class.
*/

#ifndef BASE_LIB_DIGEST_MD5_HASH_H
#define BASE_LIB_DIGEST_MD5_HASH_H

typedef unsigned long long Uint64;
#include <iomanip>
#include <iosfwd>

#include <memory.h>

namespace MI {
namespace DIGEST {

/// MD5 digest generator class, see digest_md5.h
class MD5;


/// Simple MD5 hash value class, essentially a 128 bit integer.
class MD5_hash
{
  public:
    /// Default constructor creates invalid hash.
    MD5_hash() { m_value[0] = m_value[1] = 0; }

    /// Construct with hash from MD5 generator.
    MD5_hash( MD5 const& );

    /// Construct with hash from data
    MD5_hash( Uint64 lo, Uint64 hi )
    {
        m_value[0] = lo;
        m_value[1] = hi;
    }

    /// Test validity, a hash of (0,0) is invalid.
    bool is_valid() const { return ( m_value[0] | m_value[1] ) != Uint64( 0 ); }

    Uint64 get_lo_value() const { return m_value[0]; }
    Uint64 get_hi_value() const { return m_value[1]; }

  private:
    Uint64 m_value[2];
};

/// Stream out operator
template <class Stream_T>
Stream_T& operator<<( Stream_T& os, const MD5_hash& v )
{
    std::ios_base::fmtflags originalFormat = os.flags();
    os << std::hex << std::uppercase << std::setfill( '0' ) << std::setw( 16 ) << v.get_hi_value() << '.'
       << std::setfill( '0' ) << std::setw( 16 ) << v.get_lo_value();
    os.flags( originalFormat );

    return os;
}


/// Three-way comparison: -1 for l < r; 1 for l > r; 0 for l == r.
inline int compare( MD5_hash const& l, MD5_hash const& r )
{
    if( l.get_hi_value() < r.get_hi_value() )
        return -1;
    if( l.get_hi_value() > r.get_hi_value() )
        return 1;
    if( l.get_lo_value() < r.get_lo_value() )
        return -1;
    if( l.get_lo_value() > r.get_lo_value() )
        return 1;
    return 0;
}


/// \name MD5_hash comparison operators.
//@{
inline bool operator<( MD5_hash const& l, MD5_hash const& r )
{
    if( l.get_hi_value() == r.get_hi_value() )
        return l.get_lo_value() < r.get_lo_value();
    else
        return l.get_hi_value() < r.get_hi_value();
}

inline bool operator<=( MD5_hash const& l, MD5_hash const& r )
{
    if( l.get_hi_value() == r.get_hi_value() )
        return l.get_lo_value() <= r.get_lo_value();
    else
        return l.get_hi_value() <= r.get_hi_value();
}

inline bool operator>( MD5_hash const& l, MD5_hash const& r )
{
    if( l.get_hi_value() == r.get_hi_value() )
        return l.get_lo_value() > r.get_lo_value();
    else
        return l.get_hi_value() > r.get_hi_value();
}

inline bool operator>=( MD5_hash const& l, MD5_hash const& r )
{
    if( l.get_hi_value() == r.get_hi_value() )
        return l.get_lo_value() >= r.get_lo_value();
    else
        return l.get_hi_value() >= r.get_hi_value();
}

inline bool operator==( MD5_hash const& l, MD5_hash const& r )
{
    return ( l.get_hi_value() == r.get_hi_value() ) && ( l.get_lo_value() == r.get_lo_value() );
}

inline bool operator!=( MD5_hash const& l, MD5_hash const& r )
{
    return ( l.get_hi_value() != r.get_hi_value() ) || ( l.get_lo_value() != r.get_lo_value() );
}
//@}
}
}

#endif  // header guard
