/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdUtils.c
 *
 *  Description              :
 *     
 */

/*---------------------------------- Includes --------------------------------*/

#include "stdUtils.h"
#include "stdBitSet.h"
#include "stdMessageDefs.h"

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Discard variously structured maps.
 * Parameters      : map  (I) Map to discard.
 * Function Result : 
 */
// x --> SET(y)
void STD_CDECL mapDeleteSetMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)setDelete, Nil );
    mapDelete       ( map );
}

// x --> BITSET(y)
void STD_CDECL mapDeleteBitSetMap   ( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)bitSetDelete, Nil );
    mapDelete       ( map );
}

// x --> (y-->z)
void STD_CDECL mapDeleteMapMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)mapDelete, Nil );
    mapDelete       ( map );
}


// x --> (y-->(z-->w))
void STD_CDECL mapDeleteMapMapMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)mapDeleteMapMap, Nil );
    mapDelete       ( map );
}


/*
 * Function        : Empty variously structured maps.
 * Parameters      : map  (I) Map to discard.
 * Function Result : 
 */
// x --> SET(y)
void STD_CDECL mapEmptySetMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)setDelete, Nil );
    mapEmpty        ( map );
}

// x --> BITSET(y)
void STD_CDECL mapEmptyBitSetMap   ( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)bitSetDelete, Nil );
    mapEmpty        ( map );
}

// x --> (y-->z)
void STD_CDECL mapEmptyMapMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)mapDelete, Nil );
    mapEmpty        ( map );
}


// x --> (y-->(z-->w))
void STD_CDECL mapEmptyMapMapMap( stdMap_t map )
{
    mapRangeTraverse( map, (stdEltFun)mapDeleteMapMap, Nil );
    mapEmpty        ( map );
}


/*--------------------------------- Functions --------------------------------*/


static Char *skipNext( Char *s, Char **p, Bool doEscapes, Bool inString, 
                                          Bool keepQuote, Bool keepBraces)
        {
            if (doEscapes && *s=='\\') {
                stdCHECK( *(++s), (stdMsgStrayBackSlash) ) {
                   *((*p)++)= *(s++);
                }
            } else
            if (!inString && *s == '[') {
                if (keepBraces) {
                    // check that brackets match, but allow brackets in string
                    *((*p)++)= *(s++);
                } else {
                    s++;
                }
                while (*s && *s != ']') { s=skipNext(s,p,doEscapes,False, keepQuote, keepBraces); }
                stdCHECK( *s==']', (stdMsgStrayBracket) ) { 
                    if (keepBraces) *((*p)++)= *(s++);
                    else s++;
                }
            } else
            if (*s == '"') {
		        /* handling double quotes */
                if (!inString && (doEscapes || keepQuote)) { *((*p)++)= *(s++);  } else { s++; }
                while (*s && *s != '"') { s=skipNext(s,p,doEscapes,True, keepQuote, keepBraces); }
                stdCHECK( *s=='"', (stdMsgStrayQuote) ) { 
                    if (!inString && (doEscapes || keepQuote)) { *((*p)++)= *(s++);  } else { s++; }
                }

            } else {
               *((*p)++)= *(s++);
            }
            
            return s;
        }

static String my_strtok( Char **start, cString separators, Bool doEscapes,
                                              Bool keepQuote, Bool keepBraces )
    {
        String result = *start;
        
       /* 
        * Enter separation loop, if there is anything left:
        */
        if (!*result) {
            return Nil;
        } else {
           Char *p    = result;
           Char *next = result;
                                             
           while (*next && !strchr(separators,*next)) {
            next = skipNext(next,&p,doEscapes,False, keepQuote, keepBraces);
           }
           
          /* 
           * Skip encountered separator:
           */
           if (*next) { *start= next+1; }
                 else { *start= next;   }
                 
          /* 
           * Skip off encountered token and remove leading and trailing white space.
           */
          *p = 0;

           while (*result && strchr("\t ",*(result))) { result++; }

           if (*result) {
               while (strchr("\t ",*(p-1))) { p--; }
           }

          *p = 0;


          /* 
           * Return result
           */
           return result;
        }
    }

/*
 * Function        : Tokenize string according to specified separator characters.
 * Parameters      : value      (I) String to split.
 *                   separators (I) String containing separator characters to split on.
 *                   emptyFields(I) Pass False to skip empty tokens
 *                   doEscapes  (I) Pass False iff backslash characters need to be filtered.
 *                   fun        (I) Callback function for feeding each encountered token.
 *                   data       (I) Additional user specified data element to callback function.
 *                   keepQuote  (I) False: obeying quote characters
 *                                  True:  keeping quote characters even if not do escape
 *                   keepBraces (I) False: skipping brace characters
 *                                  True:  keeping brace characters even if not do escape
 * Function Result : 
 */
 void STD_CDECL stdTokenizeString(String value, cString separators, 
                                  Bool emptyFields, Bool doEscapes, 
                                  stdEltFun fun, Pointer data, 
                                  Bool keepQuote, Bool keepBraces)
{
    if (value != Nil) {
        String s;
    
        value= stdCOPYSTRING(value);
    
        s = my_strtok(&value, separators, doEscapes, keepQuote, keepBraces);

        while ( s != Nil ) {
             if (emptyFields || *s) { fun(s, data); }
             s = my_strtok(&value, separators, doEscapes, keepQuote, keepBraces);
        }
    }
}

