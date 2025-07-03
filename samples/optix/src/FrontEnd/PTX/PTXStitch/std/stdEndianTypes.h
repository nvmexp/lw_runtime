/*
 *  Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 * 
 *  NOTICE TO USER: The source code, and related code and software
 *  ("Code"), is copyrighted under U.S. and international laws.  
 * 
 *  LWPU Corporation owns the copyright and any patents issued or 
 *  pending for the Code.  
 * 
 *  LWPU CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY 
 *  OF THIS CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS-IS" WITHOUT EXPRESS
 *  OR IMPLIED WARRANTY OF ANY KIND.  LWPU CORPORATION DISCLAIMS ALL
 *  WARRANTIES WITH REGARD TO THE CODE, INCLUDING NON-INFRINGEMENT, AND 
 *  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE.  IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 *  WHATSOEVER ARISING OUT OF OR IN ANY WAY RELATED TO THE USE OR
 *  PERFORMANCE OF THE CODE, INCLUDING, BUT NOT LIMITED TO, INFRINGEMENT,
 *  LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 *  NEGLIGENCE OR OTHER TORTIOUS ACTION, AND WHETHER OR NOT THE 
 *  POSSIBILITY OF SUCH DAMAGES WERE KNOWN OR MADE KNOWN TO LWPU
 *  CORPORATION.
 * 
 *  Module name              : stdEndianTypes.h
 *
 *  Last update              :
 *
 *  Description              :
 *
 *        This module defines the following integral types with
 *        prescribed endian representation:
 *
 *              Int64_LE,  Int32_LE,  Int16_LE    (Signed Little Endian)
 *             uInt64_LE, uInt32_LE, uInt16_LE    (Unsigned Little Endian)
 *
 *              Int64_BE,  Int32_BE,  Int16_BE    (Signed Big Endian)
 *             uInt64_BE, uInt32_BE, uInt16_BE    (Unsigned Big Endian)
 *
 *        These types behave like normal integer types. That is, they can
 *        be assigned the result of integer- valued expressions, and they
 *        can be used in expressions where any integer is allowed. 
 *
 *     EXAMPLE:
 *       
 *        uInt32_BE  bigEndianInt= 43 + fibonacci(8);  // implicit colwersion from big endian at load
 *
 *        uInt32 normalInt = bigEndianInt + 45;        // implicit colwersion to big endian at store
 */

#ifndef vpEndianTypes_INCLUDED
#define vpEndianTypes_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include <stdTypes.h>
#include <stdLocal.h>

/*------------------------------ Definitions ---------------------------------*/

#define stdDefEndianType(IntType,size,endian,Endian) \
    class IntType##size##_##endian {                                                        \
        IntType##size value;                                                                \
      public:                                                                               \
        IntType##size##_##endian(IntType##size v) { value= std##Endian##ENDIAN##size(v); }  \
                                                                                            \
        operator IntType##size () { return std##Endian##ENDIAN##size(value); }              \
    };               

    stdDefEndianType(  Int,64,LE,LITTLE );
    stdDefEndianType(  Int,32,LE,LITTLE );
    stdDefEndianType(  Int,16,LE,LITTLE );

    stdDefEndianType( uInt,64,LE,LITTLE );
    stdDefEndianType( uInt,32,LE,LITTLE );
    stdDefEndianType( uInt,16,LE,LITTLE );

    stdDefEndianType(  Int,64,BE,BIG );
    stdDefEndianType(  Int,32,BE,BIG );
    stdDefEndianType(  Int,16,BE,BIG );

    stdDefEndianType( uInt,64,BE,BIG );
    stdDefEndianType( uInt,32,BE,BIG );
    stdDefEndianType( uInt,16,BE,BIG );




/*
 * The following defines a filter for describing memory ranges in terms of 
 * endian groups (of sizes 1 .. 8 bytes), and addressable in units (of sizes 1 .. 4 bytes).
 * ET is the type of the endian group encountered in the memory area,
 * that is, before endian colwersion. TT is the same type, but after normalization to
 * the corresponding internal integer type. T is the surface pixel type.
 *
 * Three implementation classes are distinguished here:
 *
 *       stdEqEndianGroupFilter : T is identical to TT
 *       stdLtEndianGroupFilter : T is narrower than TT
 *       stdGtEndianGroupFilter : T is wider than TT
 */

class stdEndianGroupFilter {
  public :  
    virtual ~stdEndianGroupFilter(){}
    
    virtual uInt getUnit( uInt i             )= 0;
    virtual void putUnit( uInt i, uInt value )= 0;
};


template < typename ET, typename TT, typename T >
class stdEqEndianGroupFilter : public stdEndianGroupFilter {

    ET *base;
  
  public :
    stdEqEndianGroupFilter( Pointer base ) : base ((ET*) base) {}
      
    uInt getUnit( uInt address )
    {
        TT group= base[address / sizeof(T)];
        
        return group;
    }
      
    void putUnit( uInt address, uInt value )
    {
        base[address / sizeof(T)]= value;
    }
};


template < typename ET, typename TT, typename T >
class stdLtEndianGroupFilter : public stdEndianGroupFilter {

    ET *base;
  
  public :
    stdLtEndianGroupFilter( Pointer base ) : base ((ET*) base) {}
      
    uInt getUnit( uInt address )
    {
        uInt grpOffset = address / sizeof(TT);
        uInt grpElt    = address % sizeof(TT);
        TT   group     = base[grpOffset];
        TT   eltMask   = (((TT)1) << stdBITSIZEOF(T)) - 1;
        uInt eltShift  = (grpElt*stdBITSIZEOF(T));
        return (group >> eltShift) & eltMask;
    }
      
    void putUnit( uInt address, uInt value )
    {
        uInt grpOffset = address / sizeof(TT);
        uInt grpElt    = address % sizeof(TT);
        TT   group     = base[grpOffset];
        TT   eltMask   = (((TT)1) << stdBITSIZEOF(T)) - 1;
        uInt eltShift  = (grpElt*stdBITSIZEOF(T));
       
        group &= ~(eltMask     << eltShift);
        group |=  (((TT)value) << eltShift);
        
        base[grpOffset]= group;
    }
};


template < typename ET, typename TT, typename T >
class stdGtEndianGroupFilter : public stdEndianGroupFilter {

    ET *base;
  
  public :
    stdGtEndianGroupFilter( Pointer base ) : base ((ET*) base) {}
      
    uInt getUnit( uInt address )
    {
        uInt grpOffset = address / sizeof(TT);
         Int scale     = sizeof(T)/sizeof(TT);
        
        T result= 0;
        
        for (Int i=scale-1; i>=0; i--) {
            result = (result<<stdBITSIZEOF(TT)) + base[ grpOffset + i ];
        }
        
        return result;
    }
      
    void putUnit( uInt address, uInt value )
    {
        uInt grpOffset = address / sizeof(TT);
        TT   eltMask   = (TT)( (((T)1) << stdBITSIZEOF(TT)) - 1 );
         Int scale     = sizeof(T)/sizeof(TT);
        
        for (Int i=0; i<scale; i++) {
            base[ grpOffset + i ]= (TT)(value & eltMask);
            value >>= stdBITSIZEOF(TT);
        }
    }
};


#define __COMBINE_ENDIANMEM(b,bpe,bpp)  (  ((b&1)<<0) + ((bpe)<<1) + ((bpp)<<5)  )

static inline stdEndianGroupFilter* stdCreateEndianGroupFilter ( Bool bigEndian, uInt bytesPerEndianGroup, uInt bytesPerUnit, Pointer base )
{
    switch ( __COMBINE_ENDIANMEM(bigEndian,bytesPerEndianGroup,bytesPerUnit) ) {
    case __COMBINE_ENDIANMEM(True, 1,1) : return new stdEqEndianGroupFilter< uInt8,     uInt8,  uInt8  >(base);
    case __COMBINE_ENDIANMEM(True, 2,1) : return new stdLtEndianGroupFilter< uInt16_BE, uInt16, uInt8  >(base); 
    case __COMBINE_ENDIANMEM(True, 4,1) : return new stdLtEndianGroupFilter< uInt32_BE, uInt32, uInt8  >(base); 
    case __COMBINE_ENDIANMEM(True, 8,1) : return new stdLtEndianGroupFilter< uInt64_BE, uInt64, uInt8  >(base); 
    case __COMBINE_ENDIANMEM(True, 1,2) : return new stdGtEndianGroupFilter< uInt8,     uInt8,  uInt16 >(base);
    case __COMBINE_ENDIANMEM(True, 2,2) : return new stdEqEndianGroupFilter< uInt16_BE, uInt16, uInt16 >(base);
    case __COMBINE_ENDIANMEM(True, 4,2) : return new stdLtEndianGroupFilter< uInt32_BE, uInt32, uInt16 >(base);
    case __COMBINE_ENDIANMEM(True, 8,2) : return new stdLtEndianGroupFilter< uInt64_BE, uInt64, uInt16 >(base);
    case __COMBINE_ENDIANMEM(True, 1,4) : return new stdGtEndianGroupFilter< uInt8,     uInt8,  uInt32 >(base);
    case __COMBINE_ENDIANMEM(True, 2,4) : return new stdGtEndianGroupFilter< uInt16_BE, uInt16, uInt32 >(base);
    case __COMBINE_ENDIANMEM(True, 4,4) : return new stdEqEndianGroupFilter< uInt32_BE, uInt32, uInt32 >(base);
    case __COMBINE_ENDIANMEM(True, 8,4) : return new stdLtEndianGroupFilter< uInt64_BE, uInt64, uInt32 >(base);
    case __COMBINE_ENDIANMEM(False,1,1) : return new stdEqEndianGroupFilter< uInt8,     uInt8,  uInt8  >(base);
    case __COMBINE_ENDIANMEM(False,2,1) : return new stdLtEndianGroupFilter< uInt16_LE, uInt16, uInt8  >(base);
    case __COMBINE_ENDIANMEM(False,4,1) : return new stdLtEndianGroupFilter< uInt32_LE, uInt32, uInt8  >(base);
    case __COMBINE_ENDIANMEM(False,8,1) : return new stdLtEndianGroupFilter< uInt64_LE, uInt64, uInt8  >(base);
    case __COMBINE_ENDIANMEM(False,1,2) : return new stdGtEndianGroupFilter< uInt8,     uInt8,  uInt16 >(base);
    case __COMBINE_ENDIANMEM(False,2,2) : return new stdEqEndianGroupFilter< uInt16_LE, uInt16, uInt16 >(base);
    case __COMBINE_ENDIANMEM(False,4,2) : return new stdLtEndianGroupFilter< uInt32_LE, uInt32, uInt16 >(base);
    case __COMBINE_ENDIANMEM(False,8,2) : return new stdLtEndianGroupFilter< uInt64_LE, uInt64, uInt16 >(base);
    case __COMBINE_ENDIANMEM(False,1,4) : return new stdGtEndianGroupFilter< uInt8,     uInt8,  uInt32 >(base);
    case __COMBINE_ENDIANMEM(False,2,4) : return new stdGtEndianGroupFilter< uInt16_LE, uInt16, uInt32 >(base);
    case __COMBINE_ENDIANMEM(False,4,4) : return new stdEqEndianGroupFilter< uInt32_LE, uInt32, uInt32 >(base);
    case __COMBINE_ENDIANMEM(False,8,4) : return new stdLtEndianGroupFilter< uInt64_LE, uInt64, uInt32 >(base);
    default : return NULL;
    }
}


#endif
