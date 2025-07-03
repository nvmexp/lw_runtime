/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2008-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdReader.h
 *
 *  Description              :
 *
 */

/*------------------------------- Includes -----------------------------------*/

#include "stdReader.h"
#include "stdObfuscate.h"
#include "stdMessageDefs.h"

/*--------------------------------- Types ------------------------------------*/

struct stdReader  {
    rdrReaderFun    read;
    rdrResetFun     reset;
    rdrCleanupFun   cleanup;
    Pointer         data;
};

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new reader object.
 * Parameters      : read    (I) Reader function to encapsulate
 *                   reset   (I) Reset function to encapsulate
 *                   cleanup (I) Function to cleanup state upon reader deletion, 
 *                                or Nil when not appropriate
 *                   data    (I) Reader state to read from
 * Function Result : Requested (empty) string.
 */
stdReader_t STD_CDECL rdrCreate( rdrReaderFun read, rdrResetFun reset, rdrCleanupFun cleanup, Pointer data )
{
    stdReader_t result;
    
    stdNEW(result);
    
    result->read    = read;
    result->reset   = reset;
    result->cleanup = cleanup;
    result->data    = data;
    
    return result;
}


/*
 * Function        : Discard reader object.
 * Parameters      : r       (I) Reader to discard.
 * Function Result : 
 */
void STD_CDECL  rdrDelete( stdReader_t r )
{
    if (r->cleanup) { r->cleanup(r->data); }
    stdFREE(r);
}


/*
 * Function        : Reset reader object to set read position to '0'.
 * Parameters      : r       (I)  Reader to reset.
 * Function Result : 
 */
void STD_CDECL  rdrReset( stdReader_t r )
{
    r->reset(r->data);
}


/*
 * Function        : Read block of data from reader object.
 * Parameters      : r       (I)  Reader to read from.
 *                   buffer  (I)  Buffer to read into.
 *                   amount  (I)  Maximal number of bytes to read.
 * Function Result : Actual number of bytes read
 */
uInt STD_CDECL  rdrRead( stdReader_t r, Byte *buffer, uInt amount )
{
    return r->read(r->data, buffer, amount);
}


/*--------------------------------- Utilities --------------------------------*/


    static uInt STD_CDECL fileRead( FILE *f, Byte *buffer, uInt amount )
    {
        if (f) {
            return fread(buffer,1,amount,f);
        } else {
            uInt i;
            
            for (i=0; i<amount; i++) { 
                *(buffer++)= getchar();
            }
            
            return amount;
        }
    }

/*
 * Function        : Wrap reader object around opened file.
 * Parameters      : f       (I)  File handle to wrap.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateFileReader(FILE *f)
{
    return rdrCreate( (rdrReaderFun)fileRead, (rdrResetFun)rewind, Nil, f );
}


/*
 * Function        : Open the specified file, and return 
 *                   a wrapped reader object around it.
 * Parameters      : name    (I)  Name of file to open.
 * Function Result : Wrapping new reader object, 
 *                   OR Nil in case of error in opening the
 *                   file, in which case an error is issued
 *                   via msgReport.
 */
stdReader_t STD_CDECL rdrCreateFileNameReader( cString name )
{
    if (stdEQSTRING(name,"-")) {
       /*
        * Create stdin reader.
        * No rewind, no close:
        */
        return rdrCreate( (rdrReaderFun)fileRead, Nil, Nil, Nil );
    } else {
        FILE *f= fopen(name,"r");

        if (f) {
            return rdrCreate( (rdrReaderFun)fileRead, (rdrResetFun)rewind, (rdrCleanupFun)fclose, f );
        } else {
            msgReport(stdMsgOpenInputFailed,name);
            return Nil;
        }
    }
}



    typedef struct StringReadStateRec {
        Int real_size;
        cString start;
        cString current;
    } *StringReadState;

    static uInt STD_CDECL stringRead( StringReadState state, Byte *buffer, uInt amount )
    {
        Byte *bp= buffer;
        
        while (amount && *(state->current)) {
           *(bp++)= *(state->current++);
            amount--;
        }
        
        return bp - buffer;
    }

    static uInt STD_CDECL sizedStringRead( StringReadState state, Byte *buffer, uInt amount )
    {
        Byte *bp = buffer;

        while (amount && ((state->current - state->start) < state->real_size)) {
           *(bp++)= *(state->current++);
            amount--;
        }
        
        return bp - buffer;
    }

    static void STD_CDECL stringRewind( StringReadState state )
    {
        state->current= state->start;
    }

/*
 * Function        : Wrap reader object around text string.
 * Parameters      : s       (I)  String to wrap.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateStringReader(cString s)
{
    StringReadState stringState;
    
    stdNEW(stringState);

    stringState->real_size = -1;
    stringState->start   = s;
    stringState->current = s;
    
    return rdrCreate( (rdrReaderFun)stringRead, (rdrResetFun)stringRewind, (rdrCleanupFun)stdFreeFun, stringState );
}

/*
 * Function        : Wrap reader object around text string.
 * Parameters      : s       (I)  String to wrap.
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateSizedStringReader(cString s, uInt size)
{
    StringReadState stringState;
    
    stdNEW(stringState);

    stringState->real_size = (Int) size;
    stdASSERT( stringState->real_size >= 0,("Optional rdrCreateStringReader size argument must not exceed 2^31-1"));
    stringState->start   = s;
    stringState->current = s;
    
    return rdrCreate( (rdrReaderFun)sizedStringRead, (rdrResetFun)stringRewind, (rdrCleanupFun)stdFreeFun, stringState );
}


    typedef struct {
        uInt32                seed;
        stdReader_t           reader;
        stdObfuscationState   state;
    } ObfuscatingReadStateRec;
    typedef ObfuscatingReadStateRec *ObfuscatingReadState;

    static uInt STD_CDECL obfRead( ObfuscatingReadState state, Byte *buffer, uInt amount )
    {
        uInt result = rdrRead( state->reader, buffer, amount );
        
        stdDeobfuscateBuffer( state->state, (Char*)buffer, result );
        
        return result;
    }

    static void STD_CDECL obfRewind( ObfuscatingReadState state )
    {
        stdDeleteObfuscation( state->state );
        state->state = stdCreateObfuscation( state->seed );

        rdrReset( state->reader );
    }

    static void STD_CDECL obfDelete( ObfuscatingReadState state )
    {
        stdDeleteObfuscation( state->state );
        stdFREE(state);
    }

/*
 * Function        : Wrap deobfuscating reader around existing reader object
 * Parameters      : reader  (I)  Reader object to wrap.
 *                   seed    (I)  Obfuscation seed
 * Function Result : Wrapping new reader object.
 */
stdReader_t STD_CDECL rdrCreateObfuscatedReader(stdReader_t reader, uInt32 seed)
{
    ObfuscatingReadState state;
    
    stdNEW(state);
    state->seed   = seed;
    state->reader = reader;
    state->state  = stdCreateObfuscation( seed );
     
    return rdrCreate( (rdrReaderFun)obfRead, (rdrResetFun)obfRewind, (rdrCleanupFun)obfDelete, state );
}
