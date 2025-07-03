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
 *  Module name              : stdLocal.c
 *
 *  Description              :
 *     
 */

/*------------------------------- Includes ----------------------------------*/

#include "stdLocal.h"
#include "stdMessageDefs.h"
#include <stdarg.h>

/*------------------------------- Variables ---------------------------------*/

Bool stdEnableAsserts = False;

/*------------------------------ Memory Allocation -------------------------*/

void STD_CDECL stdOutOfMemory() 
{
    msgReport(stdMsgMemoryOverFlow);
}

Bool STD_CDECL stdIsOOMReported()
{
    return (stdGetThreadContext()->raisedException == stdMsgMemoryOverFlow);
}

String STD_CDECL stdGetOOMErrorMessage()
{
    return stdCOPYSTRING((stdGetThreadContext()->raisedException)->repr);
}

void STD_CDECL stdFreeFun( Pointer  p )
{
    stdFREE(p);
}

Pointer STD_CDECL  stdMallocFun( size_t size )
{
    return stdMALLOC(size);
}

Pointer STD_CDECL  stdReallocFun( Pointer p, size_t size )
{
    return stdREALLOC(p, size);
}

#if defined(USE_MEMALIGN)
Pointer STD_CDECL stdMemAlignFun( uInt align, size_t size )
{
    return stdMEMALIGN(align,size);
}
#endif

stdMemSpace_t STD_CDECL stdSwapMemSpace( stdMemSpace_t newSpace )
{
    stdThreadContext_t msgEXCT =  stdGetThreadContext();
    stdMemSpace_t      result  =  msgEXCT->lwrrentMemspace;
    msgEXCT->lwrrentMemspace   =  newSpace;
    return result;
}

/*------------------------- Missing libc functions -------------------------*/

#if defined (STD_OS_win32)

       int STD_CDECL strcasecmp (const char *s1, const char *s2)
       {
           while (True) {
               if (!(*s1)    ) { if (*s2) return -1; else return 0; } else
               if (!(*s2)    ) {          return  1;                } else {
                   char us1 = stdToUpperCase(*s1);
                   char us2 = stdToUpperCase(*s2);
                   if (us1 < us2 ) {     return -1;                } else
                   if (us1 > us2 ) {     return  1;                } else {
                       s1++; s2++;
                   }
               }
           }
           return 0;
       }

       int STD_CDECL strncasecmp(const char *s1, const char *s2, size_t n)
       {
           while (n--) {
               if (!(*s1)    ) { if (*s2) return -1; else return 0; } else
               if (!(*s2)    ) {          return  1;                } else {
                   char us1 = stdToUpperCase(*s1);
                   char us2 = stdToUpperCase(*s2);
                   if (us1 < us2 ) {     return -1;                } else
                   if (us1 > us2 ) {     return  1;                } else {
                       s1++; s2++;
                   }
               }
           }
           return 0;
       }
#endif

/*-------------------------------- Etcetera --------------------------------*/

Int STD_CDECL stdLOG2U(uInt x)
{
    Int result= -!(x&(x-1));
    while (x) { x>>=1; result++; }
    return result;  
}


Int STD_CDECL stdLOG2(uInt x)
{
    if (x==0) {
        return -1;
    } else {
        uInt px;
        do { px= x; x=x&(x-1); } while (x);
        return stdFirstBit32(px); 
    } 
}

Int STD_CDECL stdLOG2_64(uInt64 x)
{
    if (x==0) {
        return -1;
    } else {
        uInt64 px;
        do { px= x; x=x&(x-1); } while (x);
        return stdFirstBit64(px); 
    } 
}

uInt STD_CDECL stdGCD(uInt x, uInt y)
{
    while (y != 0)
    {
        x = x % y;
        if (x==0) { return y; }
        y = y % x;
    }
    return x;
}


uInt STD_CDECL stdLCM(uInt x, uInt y)
{
    uInt gcd= stdGCD(x,y);
    return (x/gcd)*y;
}



stdOverlapKind STD_CDECL stdBitsOverlapsHow32( uInt32 l, uInt32 r )
{
    if ( (l &  r) == 0 ) { return stdOverlapsNone;  } else
    if (  l == r       ) { return stdOverlapsEqual; } else
    if ( (l &  r) == l ) { return stdOverlapsLT;    } else
    if ( (l &  r) == r ) { return stdOverlapsGT;    } else
                         { return stdOverlapsSome;  } 
}


stdOverlapKind STD_CDECL stdBitsOverlapsHow64( uInt64 l, uInt64 r )
{
    if ( (l &  r) == 0 ) { return stdOverlapsNone;  } else
    if (  l == r       ) { return stdOverlapsEqual; } else
    if ( (l &  r) == l ) { return stdOverlapsLT;    } else
    if ( (l &  r) == r ) { return stdOverlapsGT;    } else
                         { return stdOverlapsSome;  } 
}


/*
 * Function        : Determine if the specified 32/64 bit integer value can be represented 
 *                   by a signed/unsigned word of the specified amount of representation bits,
 *                   under the condition that this value will further be used in two complement
 *                   arithmetic in the specified amount of computation bits.
 *                   The latter will e.g. allow the value of 16 be represented as a signed, 4 bit
 *                   quantity, provided that it is further used in 4 bit arithmetic. In contrast,
 *                   this would not be possible if e.g. 8 bit arithmetic is further used, since 
 *                   these 8 bits then would be interpreted as -16.
 * Parameters      : value         (I)  integer value.                      
 *                   nrofReprBits  (I)  Number of bits to represent value in.
 *                   nrofCompBits  (I)  Number of bits for further computations of represented value,
 *                                      or 0 if unspecified.
 *                   isSigned      (I)  If "value" is signed.               
 *                   isSSigned     (I)  If true, then value 0x80000000      
 *                                      is considered not in range.         
 *                                      This is used for checking whether   
 *                                      the number can be represented using 
 *                                      a sign bit plus absolute value      
 * Function Result : True iff. the value can be represented as described
 */
Bool STD_CDECL stdInRange32( Int32 value, uInt nrofReprBits, uInt nrofCompBits, Bool isSigned, Bool isSSigned )
{
    uInt32 mask;

    if (nrofReprBits == nrofCompBits) {
        isSigned = True;
    } else
    if (isSigned && !isSSigned) {
        nrofReprBits--;
    }
        
    mask = ((uInt32)-1) >> (stdBITSPERINT - nrofReprBits);

    if (isSSigned && (uInt32)value==0x80000000) { return False;   }
    if (isSigned  && value < 0        ) { value = ~value; }

    return (mask & (uInt32)value) == (uInt32)value;
}


Bool STD_CDECL stdInRange64( Int64 value, uInt nrofReprBits, uInt nrofCompBits, Bool isSigned, Bool isSSigned )
{
    uInt64 mask;

    if (nrofReprBits == nrofCompBits) {
        isSigned = True;
    } else
    if (isSigned && !isSSigned) {
        nrofReprBits--;
    }
    
    mask = ((uInt64)-1) >> (stdBITSPERINT64 - nrofReprBits);

    if (isSSigned && (Int64)value==S64_CONST(0x8000000000000000)) { return False;   }
    if (isSigned  && value < 0                           ) { value = ~value; }

    return (mask & (uInt64)value) == (uInt64)value;
}


/*
 * For debugging
 */
void STD_CDECL STOP( Pointer i )
{
}


/*
 * Function        : Skip initial white space and trunc trailing white space from string
 * Parameters      : str (I) String to shrinkwrap
 * Function Result : Pointer to first nonwhite space character in str
 * Note            : This function modifies the original string
 */
String STD_CDECL stdShrinkWrap( String s )
{
    uInt len= strlen(s);

    while (stdIsSpace(*s,True)) { s++; len--; }

    while (len && stdIsSpace(s[--len],True)) { s[len]= 0; }

    return s;
}


/*
 * Function        : Skips initial comments if any from string
 * Parameters      : str (I) String to skip
 * Note            : This function modifies the original string
 */
void STD_CDECL stdSkipComments(String *str)
{
    if (stdIS_PREFIX("//", *str)) {
        while (**str && **str != '\n') { (*str)++; }
        if(**str) {
            (*str)++;
        }
    } else if (stdIS_PREFIX("/*", *str)) {
        while (**str && !stdIS_PREFIX("*/", *str)) { (*str)++; }
        if (**str) {
            (*str) += 2;
        }
    }
}

/*
 * Function        : Translate non C identifier characters in string to canonical characters
 * Parameters      : from (I) Source string buffer
 *                   to   (I) Destination string buffer.
 *                            May be identical to 'from'
 */
void STD_CDECL stdTranslateNonIdChars( Char *from, Char *to)
{
    while (True) {    
        Char c= *(from++);   
             
        switch (c) {
        case '-' :
        case ' ' :
        case '.' : *(to++)= '_'; 
                   break;                  
        
        case '*' :
        case '@' :
        case '#' : *(to++)= '$'; 
                   break;                  
        
        case  0 : *(to++)= 0; 
                   return;   
        
        default :  *(to++)= c; 
                   break;            
        }
    }
}



void STD_CDECL stdPrintString ( cString  s )
{
    if (s) { printf("%s\n", s); }
      else { printf("<Nil>\n"); }
}
void STD_CDECL stdPrintInt    ( uInt    i ) { printf("%d\n",                   i); }
void STD_CDECL stdPrintInt64  ( uInt64  i ) { printf("%" stdFMT_LLD "\n",      i); }
void STD_CDECL stdPrintuInt   ( uInt    i ) { printf("%u\n",                   i); }
void STD_CDECL stdPrintuInt64 ( uInt64  i ) { printf("%" stdFMT_LLU "\n",      i); }
void STD_CDECL stdPrintHex    ( uInt    i ) { printf("0x%08x\n",               i); }
void STD_CDECL stdPrintHex64  ( uInt64  i ) { printf("0x%016" stdFMT_LLX "\n", i); }
void STD_CDECL stdPrintPInt64 ( uInt64 *i ) { printf("%" stdFMT_LLD "\n",     *i); }
void STD_CDECL stdPrintPuInt64( uInt64 *i ) { printf("%" stdFMT_LLU "\n",     *i); }
void STD_CDECL stdPrintPHex64 ( uInt64 *i ) { printf("0x%016" stdFMT_LLX "\n",*i); }


/*------------------------ Bit Manipulation Functions ------------------------*/

/* Bit manipulations in terms of 64 bit words */
#define ENDIAN_COLW(x)        stdLITTLEENDIAN64(x)
#define ALLFFFFFFFF           U64_CONST(0xffffffffffffffff)

void STD_CDECL stdPutBits( uInt64 value, uInt startInCode, uInt length, uInt64 *code)
{
    uInt64 valueMask           = (length==64) ? ALLFFFFFFFF : ~(ALLFFFFFFFF << length);
    uInt   codeWordOffset      = startInCode / stdBITSIZEOF(*code);
    uInt   codeBitInWordOffset = startInCode % stdBITSIZEOF(*code);
    uInt   bitShift            = codeBitInWordOffset;

    uInt64 maskedValue         = value & valueMask;

    uInt64 valueHWord, valueLWord, maskHWord, maskLWord;

    if ( (bitShift+length) > stdBITSIZEOF(*code)) {
        valueLWord = maskedValue >> (stdBITSIZEOF(*code) - bitShift);
        valueHWord = maskedValue << bitShift;

        maskLWord  = valueMask   >> (stdBITSIZEOF(*code) - bitShift);
        maskHWord  = valueMask   << bitShift;

        code[codeWordOffset  ]= ENDIAN_COLW((ENDIAN_COLW(code[codeWordOffset  ]) & ~maskHWord) | (valueHWord & maskHWord));
        code[codeWordOffset+1]= ENDIAN_COLW((ENDIAN_COLW(code[codeWordOffset+1]) & ~maskLWord) | (valueLWord & maskLWord));

    } else {
        valueHWord = maskedValue << bitShift;
        maskHWord  = valueMask   << bitShift;

        code[codeWordOffset  ]= ENDIAN_COLW((ENDIAN_COLW(code[codeWordOffset  ]) & ~maskHWord) | (valueHWord & maskHWord));
    }
}




uInt64 STD_CDECL stdGetBits( uInt startInCode, uInt length, uInt64 *code)
{
    uInt64 valueMask           = (length==64) ? ALLFFFFFFFF : ~(ALLFFFFFFFF << length);
    uInt   codeWordOffset      = startInCode / stdBITSIZEOF(*code);
    uInt   codeBitInWordOffset = startInCode % stdBITSIZEOF(*code);
    uInt    bitShift           = codeBitInWordOffset;

    uInt64 valueHWord          = 0;
    uInt64 valueLWord          = 0;

    uInt64 maskHWord, maskLWord;

    if ( (bitShift+length) > stdBITSIZEOF(*code)) {
        maskHWord  = valueMask >> (stdBITSIZEOF(*code) - bitShift);
        maskLWord  = valueMask << bitShift;

        valueLWord = ENDIAN_COLW(code[codeWordOffset  ]) & maskLWord;
        valueHWord = ENDIAN_COLW(code[codeWordOffset+1]) & maskHWord;

        return (valueLWord >> bitShift)
             | (valueHWord << (stdBITSIZEOF(*code) - bitShift) );

    } else {
        maskHWord  = valueMask << bitShift;
        valueHWord = ENDIAN_COLW(code[codeWordOffset  ]) & maskHWord;

        return (valueHWord >> bitShift);
    }
}


/*------------------------ Section Handling Functions ------------------------*/

#ifdef STD_OS_win32

    void STD_CDECL stdGetwin32SectionBoundaries(cString section, Pointer *start, Pointer *end)
    {
        Byte                   *Base            = (Byte *)GetModuleHandle(Nil);
        PIMAGE_NT_HEADERS       pNtHeaders      = (PIMAGE_NT_HEADERS)(Base + *(Int *)(Base + 0x3c));
        PIMAGE_SECTION_HEADER   pSectionHeaders = (PIMAGE_SECTION_HEADER)((PBYTE)&pNtHeaders->OptionalHeader + pNtHeaders->FileHeader.SizeOfOptionalHeader);
        
        Int i;

        for (i=0; i<pNtHeaders->FileHeader.NumberOfSections; i++)
        {
            if ( stdEQSTRINGN( (String)pSectionHeaders[i].Name, section, 8) ) {
                Byte *addr = Base + pSectionHeaders[i].VirtualAddress;
                Int   size = pSectionHeaders[i].Misc.VirtualSize;
                
               *start= (Pointer)(addr);
               *end  = (Pointer)(addr + size);
               
                return;
            }
        }

        stdASSERT(False, ("Section %s not found", section) );
    }

#endif


/*------------------------- Cleanup Handler Functions ------------------------*/

    typedef struct _CleanupRec  *CleanupRec;

    struct _CleanupRec {
       stdDataFun   cleanup;
       Pointer      data; 
       CleanupRec   tail;
    };

    static CleanupRec cleanupList;

void STD_CDECL stdProcessCleanupHandlers(void)
{
    while (cleanupList) {
        CleanupRec t= cleanupList;
        cleanupList = cleanupList->tail;

        t->cleanup(t->data);

        free(t);
    }
}

void STD_CDECL stdSetCleanupHandler( stdDataFun cleanup, Pointer data )
{
    CleanupRec t = (CleanupRec)malloc(sizeof(*t));

    t->cleanup   = cleanup;
    t->data      = data;

    stdGlobalEnter();

    t->tail      = cleanupList;
    cleanupList  = t;

    stdGlobalExit();
}


/*---------------------------- System Timers -------------------------*/

#ifdef STD_OS_win32
    static double        ilwfMS;    // Clock ilwerse frequency factor in milliseconds
    static double        ilwfNS;    // Clock ilwerse frequency factor in nanoseconds

    void STD_CDECL stdTimerInit(void)
    {
        LARGE_INTEGER freq;

        // Timer initializations
        QueryPerformanceFrequency(&freq);
        ilwfMS = 1000.0 / freq.QuadPart;
        ilwfNS = 1e6 * ilwfMS;
    }

    void STD_CDECL stdResetTimer(stdTimer *timer) {
        QueryPerformanceCounter(&timer->t);
    }

    Float STD_CDECL stdGetTimer(stdTimer *timer) {
        LARGE_INTEGER s;
        QueryPerformanceCounter(&s);
        return (Float) (ilwfMS * (s.QuadPart - timer->t.QuadPart));
    }
#endif

#ifdef STD_OS_FAMILY_Unix
    void stdTimerInit(void)
    {
        // No initialization required
    }

    void stdResetTimer(stdTimer *timer)
    {
        gettimeofday(&timer->t, Nil);
    }

    Float stdGetTimer(stdTimer * timer)
    {
        struct timeval s;
        gettimeofday(&s, Nil);

        return (int)(s.tv_sec - timer->t.tv_sec)*1000.0f +  
             (int)(s.tv_usec - timer->t.tv_usec)/1000.0f;
    }
#endif

#ifdef STD_OS_CygWin
    void stdTimerInit(void)
    {
        // No initialization required
    }

    void stdResetTimer(stdTimer *timer)
    {
        gettimeofday(&timer->t, Nil);
    }

    Float stdGetTimer(stdTimer * timer)
    {
        struct timeval s;
        gettimeofday(&s, Nil);

        return (int)(s.tv_sec - timer->t.tv_sec)*1000.0f +  
             (int)(s.tv_usec - timer->t.tv_usec)/1000.0f;
    }
#endif

#ifdef STD_OS_Darwin
    static Float         cvMS;  // Clock colwersion factor in milliseconds
    static Double        cvNS;  // Clock colwersion factor in nanoseconds

    void stdTimerInit(void)
    {
        mach_timebase_info_data_t tTBI;

        // Timer initializations
        mach_timebase_info(&tTBI);
        cvNS = ((Double)tTBI.numer) / ((Double)tTBI.denom);
        cvMS = (Float)(1e-6 * cvNS);
}

    void stdResetTimer(stdTimer *timer) {
        timer->t = mach_absolute_time();
    }

    // Returns ms
    Float stdGetTimer(stdTimer *timer) {
        uint64_t lwrtime = mach_absolute_time();
        return (Float)(lwrtime - timer->t) * cvMS;
    }
#endif

/*---------------------------- Identifier ----------------------------*/

cString STD_CDECL stdIDENT(void)
{
#if defined(RELEASE_IDENT)
    return stdSTRING(RELEASE_IDENT)
#else   
    return "unofficial build by " stdSTRING(STD_USER) " on " stdSTRING(STD_DATE);
#endif  
}

#ifdef STD_OS_MinGW
int STD_CDECL random(void) 
{
    static int srandom_called=0;
    if(!srandom_called) {
        srand((unsigned int)time(NULL));
        srandom_called = 1;
    }

    return rand();
}
#endif

void STD_CDECL stdASSERTReport(const char *fileName, int lineNum, const char *p)
{
    msgMessageRec msg;
    static char buf[10000];
    sprintf(buf, "Assertion failure at %s, line %d: %s", fileName, lineNum, p);
    msg.level    = msgFatal;
    msg.disabled = False;
    msg.copied   = False;
    msg.repr     = buf;
    msgReport(&msg);
}

const char * STD_CDECL stdASSERTAssembleString(const char *p, ...)
{
    static char buf[10000];
    va_list args;
    va_start(args, p);
    vsprintf(buf, p, args);
    va_end(args);

    return &buf[0];
}

void STD_CDECL stdSetAssertState(Bool enDisAsserts)
{
    stdEnableAsserts = enDisAsserts;
}
