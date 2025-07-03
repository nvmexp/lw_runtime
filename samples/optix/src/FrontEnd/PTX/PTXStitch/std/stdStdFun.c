/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdStdFun.c
 *
 *  Description              :
 *     
 *               This module defines hash-, equality- and comparison
 *               functions on the standard types, and it defines
 *               the corresponding function types.
 * 
 *               These functions and function types are colwenient
 *               for building various hash tables, and for traversing
 *               data structures.
 */

/*------------------------------- Includes -----------------------------------*/

#include "stdStdFun.h"
#include "stdLocal.h"

/*--------------------------------- Functions --------------------------------*/

typedef union { uInt32 p; Float f; } PFType;

static uInt32 cftoi( Float value )
{
    PFType pf;
    pf.f= value;
    return pf.p;
}

uInt32 STD_CDECL stdStringHash( cString s )
{
    /* Use murmur3 hash for strings,
     * as faster and better distribution than checksum hashes.
     * See https://stackoverflow.com/questions/114085/fast-string-hashing-algorithm-with-low-collision-rates-with-32-bit-integer
     * code from https://en.wikipedia.org/wiki/MurmurHash,
     * tweaked to avoid strlen.
     */
    uInt32 h = 0;
    const uInt8 *key = (const uInt8*) s;
    size_t len = 0;
    int remainingBytes;
    if (key[0] != 0 && key[1] != 0 && key[2] != 0 && key[3] != 0) {
        // use 4-byte chunks
        const uInt32* key_x4 = (const uInt32*) key;
        do {
            uInt32 k = *key_x4++;
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xe6546b64;
            key = (const uInt8*) key_x4;
            len += 4;
            // check if have 4 more bytes to process
        } while (key[0] != 0 && key[1] != 0 && key[2] != 0 && key[3] != 0);
    }
    remainingBytes = (int)(key[0] != 0);
    remainingBytes += remainingBytes == 1 ? (int)(key[1] != 0) : 0;
    remainingBytes += remainingBytes == 2 ? (int)(key[2] != 0) : 0;
    if (remainingBytes) {
        size_t i = remainingBytes;
        uInt32 k = 0;
        do {
            k <<= 8;
            k |= key[i - 1];
            ++len;
        } while (--i);
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uInt32 STD_CDECL stdIntHash       ( Int32 e )                 { return _stdIntHash(e);           }
Bool STD_CDECL   stdIntEqual      ( Int32 e1, Int32 e2 )      { return e1 == e2;                 }
Bool STD_CDECL   stdIntLessEq     ( Int32 e1, Int32 e2 )      { return e1 <= e2;                 }
uInt32 STD_CDECL stdInt64Hash     ( Int64 e )                 { return _stdInt64Hash(e);         }
Bool STD_CDECL   stdInt64Equal    ( Int64 e1, Int64 e2 )      { return e1 == e2;                 }
Bool STD_CDECL   stdInt64LessEq   ( Int64 e1, Int64 e2 )      { return e1 <= e2;                 }

uInt32 STD_CDECL stdFloatHash     ( Float e )                 { return _stdIntHash(cftoi(e)>>4); }
Bool STD_CDECL   stdFloatEqual    ( Float e1, Float e2 )      { return e1 == e2;                 }
Bool STD_CDECL   stdFloatLessEq   ( Float e1, Float e2 )      { return e1 <= e2;                 }

Bool STD_CDECL   stdStringEqual   ( cString e1, cString e2 )    { return stdEQSTRING (e1,e2);      }
Bool STD_CDECL   stdStringLessEq  ( cString e1, cString e2 )    { return stdLEQSTRING(e1,e2);      }

#ifndef DETERMINISTIC_POINTER_HASHING
uInt32 STD_CDECL stdPointerHash   ( Pointer e )               { return STD_POINTER_HASH(e);      }
#endif

Bool STD_CDECL   stdPointerEqual  ( Pointer e1, Pointer e2 )  { return e1 == e2;                 }
Bool STD_CDECL   stdPointerLessEq ( Pointer e1, Pointer e2 )  { return e1 <= e2;                 }

uInt32 STD_CDECL stdpInt32Hash    ( Int32 *e )                { return _stdIntHash(*e);          }
Bool STD_CDECL   stdpInt32Equal   ( Int32 *e1, Int32 *e2 )    { return *e1 == *e2;               }
Bool STD_CDECL   stdpInt32LessEq  ( Int32 *e1, Int32 *e2 )    { return *e1 <= *e2;               }

uInt32 STD_CDECL stdpInt64Hash    (  Int64 *e )               { return _stdInt64Hash(*e);        }
Bool STD_CDECL   stdpInt64Equal   (  Int64 *e1,  Int64 *e2 )  { return *e1 == *e2;               }
Bool STD_CDECL   stdpInt64LessEq  (  Int64 *e1,  Int64 *e2 )  { return *e1 <= *e2;               }

uInt32 STD_CDECL stdpuInt64Hash   ( uInt64 *e )               { return _stdInt64Hash(*e);        }
Bool STD_CDECL   stdpuInt64Equal  ( uInt64 *e1, uInt64 *e2 )  { return *e1 == *e2;               }
Bool STD_CDECL   stdpuInt64LessEq ( uInt64 *e1, uInt64 *e2 )  { return *e1 <= *e2;               }

Bool STD_CDECL   stdNStringEqual  ( cString e1, cString e2 )    { return (e1 && e2 && stdEQSTRING(e1,e2)) || (e1 == e2); }
uInt32 STD_CDECL stdNStringHash   ( cString e)                 { return e ? stdStringHash(e) : 0xffffffff; }


    static inline Bool isLower(Char c)
    { return ('a' <= c && c <= 'z'); } 

    static inline Char toUppercase(Char c)
    { return isLower(c) ? (c - ('a'-'A')) : c; }

Bool STD_CDECL   stdCIStringEqual ( cString e1, cString e2 ) 
{
    while (*e1 && *e2) {
        Char c1= *(e1++);
        Char c2= *(e2++);
        
        if (c1 != c2) { 
          if ( (c1^c2) & ~('a'-'A')               ) { return False; }
          if ( toUppercase(c1) != toUppercase(c2) ) { return False; }
        }
    }
    
    return !*e1 && !*e2;
}

Bool STD_CDECL   stdCIStringLessEq ( cString e1, cString e2 ) 
{
    while (*e1 && *e2) {
        Char c1= *(e1++);
        Char c2= *(e2++);
        
        if ( toUppercase(c1) > toUppercase(c2) ) { return False; }
    }
    
    return !*e1;
}

uInt32 STD_CDECL stdCIStringHash( cString e )
{
    uInt32 result= 0;
    Byte  *p= (Byte*)e;        
        
    while ( *p ) {
        result = stdStreamHash( result, toUppercase(*p) );
        
        p++;       
    }
    
    return result;
}



cString STD_CDECL stdStringIsPrefix ( cString e1, cString e2 )
{
    while (*e1 && *e2) {
        Char c1= *(e1++);
        Char c2= *(e2++);
        
        if (c1^c2) { 
          if ( (c1^c2) & ~('a'-'A') ) { return Nil; }
          if ( c1 != c2             ) { return Nil; }
        }
    }
    
    if (*e1) { return Nil; }
        else { return e2;   }
}



cString STD_CDECL stdCIStringIsPrefix ( cString e1, cString e2 )
{
    while (*e1 && *e2) {
        Char c1= *(e1++);
        Char c2= *(e2++);
        
        if (c1^c2) { 
          if ( (c1^c2) & ~('a'-'A')               ) { return Nil; }
          if ( toUppercase(c1) != toUppercase(c2) ) { return Nil; }
        }
    }
    
    if (*e1) { return Nil; }
        else { return e2;   }
}

