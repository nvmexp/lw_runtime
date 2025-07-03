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
 *  Module name              : stdWriter.h
 *
 *  Description              :
 *
 */

/*------------------------------- Includes -----------------------------------*/

#include "stdWriter.h"
#include "stdObfuscate.h"
#include "stdMessageDefs.h"

/*--------------------------------- Types ------------------------------------*/

typedef enum {
     NotSpecial,
     NullWriter,
     StringWriter,
     FileWriter,
     RawWriter
} SpecialType;

struct stdWriter  {
    SpecialType     special;
    wtrWriterFun    write;
    wtrResetFun     reset;
    wtrCleanupFun   cleanup; 
    Pointer         data;
};

/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new writer object.
 * Parameters      : write   (I) Writer function to encapsulate
 *                   reset   (I) Reset function to encapsulate
 *                   cleanup (I) Function to cleanup state upon writer deletion, 
 *                                or Nil when not appropriate
 *                   data    (I) Writer state to print to
 * Function Result : Requested (empty) string.
 */
stdWriter_t STD_CDECL wtrCreate( wtrWriterFun write, wtrResetFun reset, wtrCleanupFun cleanup, Pointer data )
{
    stdWriter_t result;
    
    stdNEW(result);
    
    result->special = NotSpecial;
    result->write   = write;
    result->reset   = reset;
    result->cleanup = cleanup;
    result->data    = data;
    
    return result;
}


/*
 * Function        : Discard writer object.
 * Parameters      : w       (I) Writer to discard, 
 *                               or Nil for trivial stdout writer
 * Function Result : 
 */
void STD_CDECL  wtrDelete( stdWriter_t w )
{
    if (w) {
        if (w->cleanup) { w->cleanup(w->data); }
        stdFREE(w);
    }
}


/*
 * Function        : Reset writer object to set print position to '0'.
 * Parameters      : w       (I)  Writer to reset.
 * Function Result : 
 */
void STD_CDECL  wtrReset( stdWriter_t w )
{
    if (w->reset) {
        w->reset(w->data);
    }
}


/*
 * Function        : Write block of data to writer object.
 * Parameters      : w       (I)  Writer to write to.
 *                   buffer  (I)  Buffer to write from.
 *                   amount  (I)  Maximal number of bytes to write.
 * Function Result : Actual number of bytes written
 */
SizeT STD_CDECL  wtrWrite( stdWriter_t w, Byte *buffer, SizeT amount )
{
    if (!w) {
       /*
        * Special case: write to stdout
        */
        return fwrite(buffer,1,amount,stdout);
        
    } else {
        switch (w->special) {
        case NotSpecial   :
              return w->write(w->data,buffer,amount);
              
        case NullWriter   :
              return amount;

        case FileWriter   :
              if (w->data) {
                  return fwrite(buffer,1,amount,(FILE*)w->data);
              } else {
                  SizeT i;
                  
                  for (i=0; i<amount; i++) { 
                      putchar(*(buffer++));
                  }
                  
                  return amount;
              }

        case StringWriter : 
          {
              stringAddBufLen((stdString_t)w->data, (String)buffer, amount);

              return amount;
          }

        case RawWriter : 
          {
            stdMEMCOPY_S(w->data, buffer, amount);
            w->data = (Byte*)w->data + amount;
            return amount;      
          }
        }

        return (SizeT)(-1);
    }
}


/*
 * Function        : Print formatted text to writer object.
 * Parameters      : w       (I)  Writer to print to.
 *                   format  (I)  The 'sprintf' format.
 *                   ...     (I)  Format data.
 * Function Result : Number of characters printed
 */
SizeT STD_CDECL  wtrPrintf( stdWriter_t w, cString format, ... )
{
    SizeT     result;
    va_list  ap;
    
    va_start (ap, format);

    result= wtrVPrintf(w,format,ap);

    va_end (ap);  
    
    return result;
}


/*
 * Function        : Print formatted text to writer object.
 * Parameters      : w       (I)  Writer to print to.
 *                   format  (I)  The 'sprintf' format.
 *                   arg     (I)  Format data.
 * Function Result : Number of characters printed
 */
SizeT STD_CDECL  wtrVPrintf( stdWriter_t w, cString format, va_list arg )
{
    if (!w) {
       /*
        * Special case: write to stdout
        */
        return vfprintf(stdout,format,arg);
        
    } else {
        switch (w->special) {
        case NotSpecial    :
        case NullWriter    :
          {
              SizeT result;
              Byte *buf;
              stdString_t tmp= stringNEW();

              stringAddVFormat(tmp,format,arg);

              result = stringSize(tmp);
              buf    = (Byte*)stringStripToBuf(tmp);
              result = wtrWrite(w,buf,result);

              stdFREE(buf);
              return result;
          }

        case FileWriter   :
              if (w->data) {
                  return vfprintf((FILE*)w->data,format,arg);
              } else {
                  return  vprintf(        format,arg);
              }

        case StringWriter : 
              return stringAddVFormat((stdString_t)w->data,format,arg);

        case RawWriter    :
          {
              SizeT result = vsprintf((char*)w->data, format, arg);
              w->data = (Byte*)w->data + result;
              return result;
          }
        }

        return (SizeT)(-1);
    }
}


/*--------------------------------- Utilities --------------------------------*/

/*
 * Function        : Return writer that sinks its output
 * Function Result : Requested new writer object.
 */
stdWriter_t STD_CDECL wtrCreateNullWriter(void)
{
    stdWriter_t result;
    
    stdNEW(result);
    
    result->special = NullWriter;
    
    return result;
}


/*
 * Function        : Wrap writer object around opened file.
 * Parameters      : f       (I)  File handle to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateFileWriter(FILE *f)
{
    stdWriter_t result;
    
    stdNEW(result);
    
    result->special = FileWriter;
    result->write   = Nil;
    result->reset   = (wtrResetFun)rewind;
    result->cleanup = Nil;
    result->data    = f;
    
    return result;
}


/*
 * Function        : Open the specified file, and return 
 *                   a wrapped writer object around it.
 * Parameters      : name    (I)  Name of file to open.
 * Function Result : Wrapping new writer object, 
 *                   OR Nil in case of error in opening the
 *                   file, in which case an error is issued
 *                   via msgReport.
 */
stdWriter_t STD_CDECL wtrCreateFileNameWriter( cString name )
{
    if (stdEQSTRING(name,"-")) {
       /*
        * Create stdout writer.
        * No rewind, no close:
        */
        stdWriter_t result;

        stdNEW(result);

        result->special = FileWriter;

        return result;
    
    } else {
        FILE *f= fopen(name,"w");

        if (f) {
            stdWriter_t result;

            stdNEW(result);

            result->special = FileWriter;
            result->write   = Nil;
            result->reset   = (wtrResetFun)rewind;
            result->cleanup = (wtrCleanupFun)fclose;
            result->data    = f;

            return result;

        } else {
            msgReport(stdMsgOpenOutputFailed,name);
            return Nil;
        }
    }
}


/*
 * Function        : Wrap writer object around string.
 * Parameters      : s       (I)  String to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateStringWriter(stdString_t s)
{
    stdWriter_t result;
    
    stdNEW(result);
    
    result->special = StringWriter;
    result->write   = Nil;
    result->reset   = (wtrResetFun)stringEmpty;
    result->cleanup = Nil;
    result->data    = s;
    
    return result;
}

/*
 * Function        : Wrap writer object around raw pointer.
 * Parameters      : ptr       (I)  Pointer to wrap.
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateRawWriter(Pointer ptr)
{
    stdWriter_t result;
    
    stdNEW(result);
    
    result->special = RawWriter;
    result->write   = Nil;
    result->reset   = Nil;
    result->cleanup = Nil;
    result->data    = ptr;
    
    return result;
}

    typedef struct {
        uInt32                seed;
        stdWriter_t           writer;
        stdObfuscationState   state;
    } ObfuscatingWriteStateRec;
    typedef ObfuscatingWriteStateRec *ObfuscatingWriteState;

    static SizeT STD_CDECL obfWrite( ObfuscatingWriteState state, Byte *buffer, SizeT amount )
    {
        Byte *tmp= (Byte*)alloca(amount);

        memcpy(tmp,buffer,amount);
        stdObfuscateBuffer( state->state, (Char*)tmp, amount );
                
        return wtrWrite( state->writer, tmp, amount );
    }

    static void STD_CDECL obfRewind( ObfuscatingWriteState state )
    {
        stdDeleteObfuscation( state->state );
        state->state = stdCreateObfuscation( state->seed );

        wtrReset( state->writer );
    }

    static void STD_CDECL obfDelete( ObfuscatingWriteState state )
    {
        stdDeleteObfuscation( state->state );
        stdFREE(state);
    }

/*
 * Function        : Wrap obfuscating writer around existing writer object
 * Parameters      : writer  (I)  Writer object to wrap.
 *                   seed    (I)  Obfuscation seed
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateObfuscatedWriter(stdWriter_t writer, uInt32 seed)
{
    ObfuscatingWriteState state;
    
    stdNEW(state);
    state->seed   = seed;
    state->writer = writer;
    state->state  = stdCreateObfuscation( seed );
     
    return wtrCreate( (wtrWriterFun)obfWrite, (wtrResetFun)obfRewind, (wtrCleanupFun)obfDelete, state );
}



    typedef struct TabColwWriteState_t {
        SizeT                  pos;
        uInt                  tablen;
        stdWriter_t           writer;
    } *TabColwWriteState;

    static SizeT STD_CDECL tcolwWrite( TabColwWriteState state, Byte *buffer, SizeT amount )
    {
        SizeT result = 0;
        
        while (amount--) {
            Byte c      = *(buffer++);
            uInt repeat = 1;
            
            if (c == '\t') {
                c      = ' ';
                repeat = stdROUNDUP(state->pos+1,state->tablen) - state->pos;
            } else 
            if (c == '\n') {
                state->pos = (SizeT)-1;
            }
            
            state->pos += repeat;
            result     += repeat;
            
            while (repeat--) {
                wtrWrite(state->writer,&c,1);
            } 
        }
        
        return result;
    }

    static void STD_CDECL tcolwRewind( TabColwWriteState state )
    {
        state->pos = 0;

        wtrReset( state->writer );
    }

    static void STD_CDECL tcolwDelete( TabColwWriteState state )
    {
        stdFREE(state);
    }

/*
 * Function        : Wrap tab colwerting writer around existing writer object
 * Parameters      : writer  (I)  Writer object to wrap.
 *                   tablen  (I)  Tab length to be used
 * Function Result : Wrapping new writer object.
 */
stdWriter_t STD_CDECL wtrCreateTabColwWriter(stdWriter_t writer, uInt tablen)
{
    TabColwWriteState state;
    
    stdNEW(state);
    state->tablen = tablen;
    state->writer = writer;
     
    return wtrCreate( (wtrWriterFun)tcolwWrite, (wtrResetFun)tcolwRewind, (wtrCleanupFun)tcolwDelete, state );
}

