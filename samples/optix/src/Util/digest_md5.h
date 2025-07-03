/******************************************************************************
 * Copyright (c) 2017, LWPU CORPORATION. All rights reserved.
 ******************************************************************************
 * Author:      dimfair
 * Created:     24.2.2009
 *
 * Purpose:     Implementation of MD5 digest algorithm
 *
 ******************************************************************************
 *
 *  Derived from the RSA Data Security, Inc. MD5 Message-Digest Algorithm
 *
 *  Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
 *  rights reserved.
 *
 *  License to copy and use this software is granted provided that it
 *  is identified as the "RSA Data Security, Inc. MD5 Message-Digest
 *  Algorithm" in all material mentioning or referencing this software
 *  or this function.
 *
 *  License is also granted to make and use derivative works provided
 *  that such works are identified as "derived from the RSA Data
 *  Security, Inc. MD5 Message-Digest Algorithm" in all material
 *  mentioning or referencing the derived work.
 *
 *  RSA Data Security, Inc. makes no representations concerning either
 *  the merchantability of this software or the suitability of this
 *  software for any particular purpose. It is provided "as is"
 *  without express or implied warranty of any kind.
 *
 *****************************************************************************/

#ifndef __MD5HASH_200902241045_H
#define __MD5HASH_200902241045_H

#include <stdio.h>
#include <string.h>

typedef unsigned char      Uint8;
typedef unsigned short     Uint16;
typedef unsigned int       Uint32;
typedef unsigned long long Uint64;

namespace MI {
namespace DIGEST {

class MD5
{
  public:
    // simple initializer
    MD5();
    // constructors for special cirlwmstances.  All these constructors finalize
    // the MD5 context.
    MD5( FILE* file );                              // digest file, close, finalize
    MD5( const char* input, size_t input_length );  // digest bffr, finalize

    // methods for controlled operation:
    bool update( const char* input, size_t input_length );
    bool update( FILE* file );
    bool finalize();

    // methods to acquire finalized result
    const char* raw_digest() const;         // digest as a 16-byte binary array
    const char* hex_digest( char* ) const;  // digest as a 32-byte ascii-hex string


  private:
    // internal types
    typedef Uint32 uint4;
    typedef Uint16 uint2;
    typedef Uint8  uint1;

    // private data:
    uint4 state[4];
    uint4 count[2];    // number of *bits*, mod 2^64
    uint1 buffer[64];  // input buffer
    uint1 digest[16];
    uint1 finalized;

    // private methods:
    void init();                      // called by all constructors
    void transform( uint1* buffer );  // does the real update work.  Note
                                      // that length is implied to be 64.

    static void encode( uint1* dest, uint4* src, uint4 length );
    static void decode( uint4* dest, uint1* src, uint4 length );

    static inline uint4 rotate_left( uint4 x, uint4 n );
    static inline uint4 F( uint4 x, uint4 y, uint4 z );
    static inline uint4 G( uint4 x, uint4 y, uint4 z );
    static inline uint4 H( uint4 x, uint4 y, uint4 z );
    static inline uint4 I( uint4 x, uint4 y, uint4 z );
    static inline void FF( uint4& a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac );
    static inline void GG( uint4& a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac );
    static inline void HH( uint4& a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac );
    static inline void II( uint4& a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac );
};


// Utility for incremental hash update with arbitrary plain data.
template <typename T>
inline void update_raw( MD5& hasher, T const& data )
{
    hasher.update( (const char*)&data, sizeof( T ) );
}

// Utility for incremental hash update with c-string.
inline void update_str( MD5& hasher, const char* str )
{
    if( str )
    {
        hasher.update( str, strlen( str ) );
    }
}

}  // namespace DIGEST
}  // namespace MI

#endif  // __MD5HASH_200902241045_H
